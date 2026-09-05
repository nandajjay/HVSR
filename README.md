# SESAME-Compliant HVSR Analysis Tool

A Python-based seismic signal processing tool for automated Horizontal-to-Vertical Spectral Ratio (HVSR) analysis of ambient vibration data.

## Overview

This project implements an end-to-end HVSR analysis pipeline for extracting dominant resonance frequencies from low-SNR ambient seismic recordings. The workflow follows SESAME guidelines and includes preprocessing, windowed spectral analysis, custom spectral smoothing, automated peak detection, and reliability assessment.

The tool was developed and validated using real seismic data for research applications in collaboration with an IISc Bangalore research group.

## Features

* Supports seismic data in SEED format
* Automatic identification of Z, N, and E components
* Signal preprocessing including detrending, demeaning, tapering, and filtering
* Windowed FFT-based spectral analysis with overlap
* Geometric mean calculation of horizontal components
* Log-space averaging of HVSR curves
* Custom implementation of Konno–Ohmachi spectral smoothing
* Automated peak detection using SciPy
* SESAME-based peak reliability and clarity assessment
* Batch processing of multiple recordings
* HVSR visualization with confidence bounds
* CSV export of detected peak parameters

## Processing Pipeline

```text
SEED Seismic Data
        ↓
Component Identification (Z, N, E)
        ↓
Preprocessing & Filtering
        ↓
Windowing with Overlap
        ↓
FFT-Based Spectral Estimation
        ↓
Horizontal / Vertical Spectral Ratio
        ↓
Log-Space Averaging
        ↓
Konno–Ohmachi Smoothing
        ↓
Peak Detection
        ↓
SESAME Reliability Validation
        ↓
Plots & CSV Output
```

## HVSR Calculation

The horizontal spectral component is calculated using the geometric mean:

H(f) = √(HN(f) × HE(f))

The Horizontal-to-Vertical Spectral Ratio is then computed as:

HVSR(f) = H(f) / V(f)

Multiple window-wise HVSR curves are combined using geometric averaging in log space to improve statistical stability.

## Technologies Used

* Python
* NumPy
* SciPy
* ObsPy
* Pandas
* Matplotlib

## Validation

The automated peak detection results were compared against manually analyzed HVSR results obtained using Geopsy. Detected resonance frequencies were assessed using SESAME reliability and clarity criteria.

## Applications

* Seismic site characterization
* Site resonance frequency estimation
* Ambient vibration analysis
* Earthquake engineering studies
* Structural and subsurface dynamic characterization

## Author

**Nandaj Jayaraj**
Developed as part of seismic signal processing research work.
