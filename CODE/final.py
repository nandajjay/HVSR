# Author:       [Nandaj Jayaraj]
# Date:         October 30, 2025
# Contact:      [nandajofc@gmail.com]

# Description:
# This script performs a full HVSR analysis on seismic data (SEED format)
# in compliance with SESAME guidelines. It includes custom-built functions for
# windowed spectral analysis, geometric mean of horizontal components,
# custom Konno-Ohmachi smoothing, and SESAME-compliant peak validation.

# Copyright (c) 2025 [Nandaj Jayaraj]



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import read, Stream
# REMOVED: from obspy.signal.konnoohmachismoothing import konno_ohmachi_smoothing
from matplotlib.ticker import FixedLocator, ScalarFormatter, NullLocator
from scipy.signal import find_peaks
from scipy.signal.windows import hann, tukey
from scipy.fft import fft
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ----------------------------
# SESAME COMPLIANT PARAMETERS
# ----------------------------
fmin = 0.20      # Hz
fmax = 40.0      # Hz
hp_cut = 1.0     # Hz - High-pass filter
window_len = 60.0 # seconds
overlap = 0.40   ## CHANGED: 50% is a common and robust choice
bandwidth = 9.0 # Konno-Ohmachi bandwidth (b-value). Smaller = more smoothing.

# SESAME reliability criteria parameters
MIN_SIGNIFICANT_CYCLES = 200
MIN_WINDOWS = 10

# ----------------------------
# HELPER & PREPROCESSING FUNCTIONS (Largely Unchanged)
# ----------------------------

def identify_components(stream):
    """Identifies Z, N, E components from the stream."""
    component_map = {
        'Z': ['Z', 'BHZ', 'HHZ', 'LHZ', 'V', 'VERT'],
        'N': ['N', 'BHN', 'HHN', 'LHN', 'NS'],
        'E': ['E', 'BHE', 'HHE', 'LHE', 'EW']
    }
    identified = {'Z': None, 'N': None, 'E': None}
    for tr in stream:
        channel = tr.stats.channel.upper()
        for comp, codes in component_map.items():
            if any(code in channel for code in codes):
                if identified[comp] is None:
                    identified[comp] = tr
                    break
    if None in identified.values():
        missing = [k for k, v in identified.items() if v is None]
        raise ValueError(f"Missing components: {', '.join(missing)}")
    return identified['Z'], identified['N'], identified['E']

def apply_preprocessing(stream, highpass_freq=0.5):
    """Applies standard preprocessing steps."""
    stream.merge(method=1, fill_value='interpolate')
    trZ, trN, trE = identify_components(stream)
    
    # Ensure all traces have the same length after merging
    min_len = min(len(trZ.data), len(trN.data), len(trE.data))
    trZ.data = trZ.data[:min_len]
    trN.data = trN.data[:min_len]
    trE.data = trE.data[:min_len]

    ordered_stream = Stream([trZ, trN, trE])
    ordered_stream.detrend('linear')
    ordered_stream.detrend('demean')
    ordered_stream.taper(max_percentage=0.05, type='cosine') # Apply taper
    ordered_stream.filter('highpass', freq=highpass_freq, corners=4, zerophase=True)
    return ordered_stream

# ----------------------------
# CORE HVSR CALCULATION (WITH CUSTOM SMOOTHING)
# ----------------------------

def custom_konno_ohmachi_smoothing(data, freqs, bandwidth=40.0, normalize=True):
    """  
    Applies Konno-Ohmachi smoothing to spectral data.
    This window is triangular in the log-frequency domain.
    
    :param data: 1D numpy array of spectral amplitudes.
    :param freqs: 1D numpy array of frequencies corresponding to data.
    :param bandwidth: Konno-Ohmachi bandwidth coefficient (b-value).
                      This controls the 'width' of the log-triangular window.
                      A *smaller* value (e.g., 15-20) gives a *wider* window (more smoothing).
                      A *larger* value (e.g., 40) gives a *narrower* window (less smoothing).
    :param normalize: If True, normalizes the window sum (standard).
    :return: 1D numpy array of smoothed spectral amplitudes.
    """
    smoothed_data = np.zeros_like(data)
    
    # Find the first non-zero frequency index. Smoothing is not defined at f=0.
    idx_f_min = np.where(freqs > 0)[0]
    if len(idx_f_min) == 0:
        # All frequencies are 0, return original data
        return data
        
    idx_f_min = idx_f_min[0]
    
    # Work only with positive frequencies in log space
    log_freqs = np.log10(freqs[idx_f_min:])
    data_to_smooth = data[idx_f_min:]
    
    # Iterate over each frequency point to calculate its smoothed value
    for i in range(len(log_freqs)):
        log_f_c = log_freqs[i] # Center frequency (log)
        
        # Calculate log-ratios relative to the center frequency
        # log_ratios = log10(f_i) - log10(f_c) = log10(f_i / f_c)
        log_ratios = log_freqs - log_f_c
        
        # Apply the triangular window function in log-space
        # W(f) = 1 - b * |log10(f/f_c)|  for |log10(f/f_c)| <= 1/b
        # W(f) = 0 otherwise
        window_vals = np.maximum(0, 1.0 - bandwidth * np.abs(log_ratios))
        
        # Normalize and apply the window
        if normalize:
            sum_window = np.sum(window_vals)
            if sum_window > 0:
                smoothed_data[idx_f_min + i] = np.sum(window_vals * data_to_smooth) / sum_window
            else:
                # This should not happen if i is in range, but as a fallback
                smoothed_data[idx_f_min + i] = data_to_smooth[i]
        else:
             # This isn't standard, but for completeness
            smoothed_data[idx_f_min + i] = np.sum(window_vals * data_to_smooth)

    # Handle the zero-frequency component(s)
    if idx_f_min > 0:
         # Just copy the original value(s) for f=0
        smoothed_data[:idx_f_min] = data[:idx_f_min]
        
    return smoothed_data

def calculate_hvsr_with_windows(stream):
    """
    Calculates HVSR by averaging spectra from multiple windows, the standard method.
    """
    trZ, trN, trE = stream[0], stream[1], stream[2]
    fs = trZ.stats.sampling_rate
    
    nperseg = int(window_len * fs)
    noverlap = int(nperseg * overlap)
    step = nperseg - noverlap

    win = tukey(nperseg, alpha=0.10) # Tapered window
    all_hv_curves = []
    
    # Iterate through the data with the specified overlap
    for i in range(0, len(trZ.data) - nperseg + 1, step):
        # Slice and window data for all three components
        z_win, n_win, e_win = [comp.data[i : i + nperseg] * win for comp in [trZ, trN, trE]]

        # Compute FFT and then Power Spectra for each component
        fZ, fN, fE = fft(z_win), fft(n_win), fft(e_win)
        n_fft = len(fZ)
        freqs = np.fft.fftfreq(n_fft, d=1/fs)[:n_fft//2]
        
        PxxZ, PxxN, PxxE = [np.abs(f[:n_fft//2])**2 for f in [fZ, fN, fE]]
        
        # Combine horizontals using geometric mean of amplitude (sqrt of power)
        AmpN = np.sqrt(PxxN)
        AmpE = np.sqrt(PxxE)
        AmpZ = np.sqrt(PxxZ)
        AmpH_geom = np.sqrt(AmpN * AmpE)
        
        # Calculate H/V ratio for this window, avoiding division by zero
        hv_win = AmpH_geom / (AmpZ + 1e-12)
        all_hv_curves.append(hv_win)

    if not all_hv_curves:
        raise ValueError("No valid windows found. The recording might be too short.")

    # Calculate the geometric mean and standard deviation across all windows
    all_hv_curves = np.array(all_hv_curves)
    log_hv = np.log(all_hv_curves)
    mean_log_hv = np.mean(log_hv, axis=0)
    std_log_hv = np.std(log_hv, axis=0)
    
    # Convert back to linear scale for plotting and analysis
    mean_hv = np.exp(mean_log_hv)
    std_hv_upper = np.exp(mean_log_hv + std_log_hv)
    std_hv_lower = np.exp(mean_log_hv - std_log_hv)
    
    # ---!!! CHANGED !!!---
    # Use the custom Konno-Ohmachi smoothing function
    mean_hv_smooth = custom_konno_ohmachi_smoothing(mean_hv, freqs, bandwidth=bandwidth, normalize=True)
    # ---!!! END CHANGE !!!---
    
    num_windows = len(all_hv_curves)
    print(f"   Processed {num_windows} windows.")
    
    return freqs, mean_hv_smooth, (std_hv_lower, std_hv_upper), num_windows

# ----------------------------
# SESAME CHECKS & PEAK DETECTION (IMPROVED)
# ----------------------------

def assess_sesame_peak_criteria(f0, A0, freqs, hv_curve, std_curves, num_windows):
    """
    A more robust SESAME assessment function.
    Checks the reliability and clarity of a given peak (f0, A0).
    """
    # 1. Reliability Checks
    check1 = f0 > 10.0 / window_len
    significant_cycles = window_len * num_windows * f0
    check2 = significant_cycles >= MIN_SIGNIFICANT_CYCLES
    
    # Find std deviation at the peak frequency f0
    f0_idx = np.argmin(np.abs(freqs - f0))
    # Using log-space standard deviation is more robust
    std_log_A0 = np.log(std_curves[1][f0_idx]) - np.log(hv_curve[f0_idx])
    threshold_theta = 2.0 if f0 > 0.5 else 3.0
    check3 = std_log_A0 < threshold_theta

    is_reliable = all([check1, check2, check3])

    # 2. Clarity Checks
    clarity1 = A0 > 2.0
    low_freq_mask = (freqs >= f0/4) & (freqs < f0)
    clarity2 = np.any(hv_curve[low_freq_mask] < A0/2) if np.any(low_freq_mask) else False
    high_freq_mask = (freqs > f0) & (freqs <= 4*f0)
    clarity3 = np.any(hv_curve[high_freq_mask] < A0/2) if np.any(high_freq_mask) else False
    
    is_clear = sum([clarity1, clarity2, clarity3]) >= 2

    return {
        'is_reliable': is_reliable,
        'is_clear': is_clear,
        'significant_cycles': significant_cycles,
    }

def find_and_assess_peaks(freqs, hv_curve, std_curves, num_windows):
    """
    Simplified and more robust peak finding and assessment.
    """
    mask = (freqs >= fmin) & (freqs <= fmax)
    f_plot, hv_plot = freqs[mask], hv_curve[mask]
    std_plot = (std_curves[0][mask], std_curves[1][mask])

    if len(f_plot) == 0:
        return []

    # Find all peaks with reasonable constraints
    peaks, _ = find_peaks(hv_plot, height=1.5, prominence=0.1, distance=5)
    
    if len(peaks) == 0 and np.max(hv_plot) > 1.5:
        peaks = np.array([np.argmax(hv_plot)]) # Fallback to absolute max if no peaks found
        print("   No prominent peaks found, falling back to absolute maximum.")

    peak_assessments = []
    for peak_idx in peaks:
        f0, A0 = f_plot[peak_idx], hv_plot[peak_idx]
        
        # Calculate peak width (this is a simple FWHM-like estimate)
        try:
            # Find indices where curve is above half-amplitude
            above_half_amp = np.where(hv_plot >= A0 / 2)[0]
            
            # Find left-side crossing
            left_crossings = above_half_amp[above_half_amp < peak_idx]
            left_idx = left_crossings[-1] if len(left_crossings) > 0 else 0
            
            # Find right-side crossing
            right_crossings = above_half_amp[above_half_amp > peak_idx]
            right_idx = right_crossings[0] if len(right_crossings) > 0 else len(f_plot) - 1

            width = f_plot[right_idx] - f_plot[left_idx]
        except Exception:
            width = 0.1 # Fallback width

        assessment = assess_sesame_peak_criteria(f0, A0, f_plot, hv_plot, std_plot, num_windows)
        assessment.update({
            'frequency': f0,
            'amplitude': A0,
            'width': width,
        })
        peak_assessments.append(assessment)
    
    # Sort peaks by amplitude, descending
    peak_assessments.sort(key=lambda x: x['amplitude'], reverse=True)
    
    # Mark the absolute maximum peak
    if peak_assessments:
        peak_amps = [p['amplitude'] for p in peak_assessments]
        abs_max_idx = np.argmax(peak_amps)
        for i, p in enumerate(peak_assessments):
            p['is_absolute_max'] = (i == abs_max_idx)

    print(f"   Found {len(peak_assessments)} potential peaks.")
    return peak_assessments [:4] # Return top 4 peaks

# ----------------------------
# PLOTTING FUNCTION (IMPROVED)
# ----------------------------

def plot_sesame_compliant_hvsr(result):
    """
    Plots the standard deviation as a shaded blue area.
    """
    plt.figure(figsize=(12, 8))
    
    # Plot standard deviation as a shaded area
    plt.fill_between(result['frequencies'], result['HV_std_lower'], result['HV_std_upper'],
                     color='lightblue', alpha=0.7)

    # Main H/V curve
    plt.plot(result['frequencies'], result['HV_ratio'], 'k-', linewidth=2, label='Mean H/V Ratio (Smoothed)')
    
    # Plot peaks
    for i, assessment in enumerate(result['peak_assessments']):
        freq, amp = assessment['frequency'], assessment['amplitude']
        
        if assessment['is_reliable'] and assessment['is_clear']:
            color, marker, label_suffix = 'green', 'o', ' (Reliable & Clear)'
        elif assessment['is_reliable']:
            color, marker, label_suffix = 'orange', 's', ' (Reliable)'
        else:
            color, marker, label_suffix = 'red', '^', ' (Not Reliable)'

        if assessment.get('is_absolute_max', False):
             label_suffix += ' ★ Abs Max'

        plt.plot(freq, amp, marker=marker, color=color, markersize=10, linestyle='None',
                 label=f'Peak {i+1}: {freq:.2f} Hz{label_suffix}')
        plt.annotate(f'{freq:.2f} Hz', xy=(freq, amp), xytext=(5, 5),
                     textcoords='offset points', fontsize=9, color='blue')
    
    # Axis labels and Title
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('H/V Amplitude Ratio', fontsize=12)
    plt.title(f'H/V Analysis - {result["file"]}', fontsize=14)
    plt.grid(True, which="both", linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    
    # Set the scale and limits
    plt.xscale('log')
    plt.xlim(fmin, fmax)

    # Manually set x-axis ticks to match Geopsy
    ax = plt.gca() 
    ticks = [0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8, 10, 20, 30]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())

    # Dynamic y-axis
    if result['peak_assessments']:
        max_peak_amp = max([p['amplitude'] for p in result['peak_assessments']])
    else:
        max_peak_amp = 0
    max_curve_amp = np.max(result['HV_ratio']) if len(result['HV_ratio']) > 0 else 0
    max_y = max(max_peak_amp, max_curve_amp)
    upper_limit = max(max_y * 1.2, 3.0)
    plt.ylim(0, upper_limit)

    plt.tight_layout()
    plt.show()
        
# ----------------------------
# MAIN PROCESSING & VALIDATION (INTEGRATED)
# ----------------------------

def process_sesame_compliant(file_path):
    """Main processing function using the corrected, window-based approach."""
    try:
        st = read(str(file_path))
        processed_stream = apply_preprocessing(st, highpass_freq=hp_cut)
        
        freqs, hv_mean, hv_std_curves, num_windows = calculate_hvsr_with_windows(processed_stream)
        
        peak_assessments = find_and_assess_peaks(freqs, hv_mean, hv_std_curves, num_windows)
        
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        
        # Package results with all necessary info for plotting and analysis
        result = {
            'file': file_path.name,
            'frequencies': freqs[freq_mask],
            'HV_ratio': hv_mean[freq_mask],
            'HV_std_lower': hv_std_curves[0][freq_mask],
            'HV_std_upper': hv_std_curves[1][freq_mask],
            'peak_assessments': peak_assessments,
            'num_windows': num_windows,
            'sampling_rate': processed_stream[0].stats.sampling_rate
        }
        return result
    except Exception as e:
        print(f"   ERROR processing {file_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def validate_with_manual_results(seed_files, manual_peaks):
    """Compare with manual Geopsy measurements using the corrected processing."""
    print("VALIDATING CODE AGAINST MANUAL PEAKS")
    print("=" * 70)
    comparison_data = []
    
    for i, file in enumerate(seed_files):
        if i >= len(manual_peaks): break
        manual_freq = manual_peaks[i]
        
        print(f"\n[{i+1}/{len(manual_peaks)}] {file.name} | Manual Peak: {manual_freq:.2f} Hz")
        result = process_sesame_compliant(file)
        
        if result and result['peak_assessments']:
            # Use the highest amplitude peak for comparison
            primary_peak = result['peak_assessments'][0]
            code_freq = primary_peak['frequency']
            difference = manual_freq - code_freq
            
            print(f"   Code Peak: {code_freq:.3f} Hz | Diff: {difference:+.3f} Hz")
            print(f"   Reliability: {'✓' if primary_peak['is_reliable'] else '✗'}, Clarity: {'✓' if primary_peak['is_clear'] else '✗'}")

            comparison_data.append({'file': file.name, 'manual_Hz': manual_freq, 'code_Hz': code_freq, 'difference_Hz': difference})
            
            # Plot the result for visual validation
            plot_sesame_compliant_hvsr(result)
        else:
            print("   ❌ No peaks detected by the code.")
            comparison_data.append({'file': file.name, 'manual_Hz': manual_freq, 'code_Hz': np.nan, 'difference_Hz': np.nan})
    return comparison_data

def main():
    """Main execution function"""
    # Define your manual results and data folder
    manual_peaks = [
        22.9, 13.94, 14.17, 17.3, 19.34, 22.45, 14.87, 1.34, 
        10.64, 17.26, 14.72, 12.38, 11.94, 19.01, 2.01, 1.33, 17.55, 21.98
    ]
    folder = Path("D:\IISC-Internship\Data\VV_SAGAR\pYTHON\twohndrd") #folder path
    seed_files = sorted(list(folder.glob("*.seed")))

    if not seed_files:
        print(f"No .seed files found in {folder}. Please check the path.")
        return

    # 1. Run validation with plotting for the first set of files
    validate_with_manual_results(seed_files, manual_peaks)
    
    # 2. Run batch processing for ALL files and save to CSV
    print(f"\n\n{'='*70}\nBATCH PROCESSING ALL FILES TO CSV\n{'='*70}")
    
    all_results_for_df = []
    for i, file in enumerate(seed_files):
        print(f"\n[{i+1}/{len(seed_files)}] Processing: {file.name}")
        result = process_sesame_compliant(file)
        
        if result:
            row = {'file': result['file'], 'num_windows': result['num_windows']}
            for i_peak in range(4): # Store up to 4 peaks
                if i_peak < len(result['peak_assessments']):
                    peak = result['peak_assessments'][i_peak]
                    for key in ['frequency', 'amplitude', 'width', 'is_reliable', 'is_clear', 'is_absolute_max']:
                        row[f'peak{i_peak+1}_{key}'] = peak[key]
                else:
                    for key in ['frequency', 'amplitude', 'width']:
                        row[f'peak{i_peak+1}_{key}'] = np.nan
                    for key in ['is_reliable', 'is_clear', 'is_absolute_max']:
                        row[f'peak{i_peak+1}_{key}'] = False
            all_results_for_df.append(row)

    if all_results_for_df:
        df = pd.DataFrame(all_results_for_df)
        output_file = folder / "hvsr_analysis_CUSTOM_SMOOTHING_B15.csv"
        df.to_csv(output_file, index=False, float_format='%.4f')
        print(f"\n✅ Batch processing complete! Results saved to: {output_file}")

if __name__ == "__main__":
    main()