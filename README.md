# Isolated Word Recognition Project

This project implements isolated word recognition for Recorded Persian spoken digits (0–9) using various feature extraction techniques. The primary objective is to analyze the performance of different feature extraction methods on recognizing spoken digits (0–9). The project includes Python scripts for feature extraction and models like Wav2Vec 2.0 and HuBERT.

---

## Table of Contents
- [Features](#features)
- [Feature Extraction Methods](#feature-extraction-methods)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)

---

## Features
1. **Feature Extraction**:
   - Real Cepstrum
   - MFCC (Mel Frequency Cepstral Coefficients)
   - MFCC with Energy
   - Delta and Delta-Delta MFCCs
   - Wav2Vec 2.0 features
   - HuBERT features
2. **Isolated Word Recognition**:
   - Recognizes digits (0–9) spoken twice for testing and training.
3. **Analysis and Results**:
   - Comparative analysis of feature extraction techniques.
   - Recognition accuracy for each method.

---

## Feature Extraction Methods
The following methods are implemented to extract features from speech:
1. **Real Cepstrum**: Captures spectral envelope.
2. **MFCC**: Extracts features based on the human auditory system.
3. **MFCC with Energy**: Adds overall signal energy to MFCC features.
4. **Delta and Delta-Delta MFCC**: Incorporates temporal derivatives for dynamic features.
5. **Wav2Vec 2.0**: Uses transformer-based contextual speech representations.
6. **HuBERT**: Another transformer-based model for robust speech representation.

---

## Prerequisites
Ensure you have the following installed:
- Python 3.7 or higher
- NumPy
- SciPy
- Matplotlib
- PyTorch (for Wav2Vec 2.0 and HuBERT)
- Librosa

## Project Structure
**realCepstrum.py**: Extracts real cepstrum features.\
**mfcc.py**: Extracts MFCC features.\
**mfcc_energy.py**: Extracts MFCC features with energy.\
**delta_mfcc_energy.py**: Extracts delta MFCC features with energy.\
**delta2_mfcc_energy.py**: Extracts delta-delta MFCC features with energy.\
**WAV2VEC2.py**: Extracts features using the Wav2Vec 2.0 model.\
**Hubert.py**: Extracts features using the HuBERT model.\
**Recording**: Folder containing recorded audio files.

