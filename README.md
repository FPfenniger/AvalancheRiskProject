# Dry Avalanche Risk in Switzerland:  Assessing the Feasibility of Deep Learning Models for Danger Level Prediction

This project explores the application of deep learning models for predicting avalanche danger levels in Switzerland. Using historical meteorological data from IMIS (Intercantonal Measurement and Information System) weather stations and official SLF (WSL Institute for Snow and Avalanche Research) danger level bulletins, we assess whether modern deep learning architectures can effectively predict avalanche risk. 

## Project Overview

Avalanche risk prediction is a critical safety concern in mountainous regions. This project investigates whether deep learning models—specifically Convolutional Neural Networks (CNNs) and ConvLSTM architectures—can learn spatial and temporal patterns from meteorological data to predict avalanche danger levels across Swiss alpine regions.

## Repository Structure

```
AvalancheRiskProject/
├── data/                           # Data files and geographical boundaries
│   ├── data_rf1_forecast.csv       # Main Dataset from Guillen-Pérez et al. (2022) -> https://doi.org/10.16904/envidat.330
│   ├── imis_df_cleaned.csv         # Cleaned IMIS station data
│   ├── imis_df_final_clean.csv     # Final processed IMIS data
│   ├── slf_boundaries_2020.json    # SLF regional boundaries (GeoJSON)
│   ├── stations.csv                # IMIS station metadata (for coordinates)
│   └── swissboundaries.gpkg        # Swiss boundary data (GeoPackage)
│
├── notebooks/                      # Jupyter notebooks for analysis pipeline
│   ├── 1_Preprocessing. ipynb       # Data cleaning and preparation
│   ├── 2_EDA. ipynb                 # Exploratory Data Analysis
│   ├── 3_BaselineModels.ipynb      # Traditional ML baseline models
│   ├── 4_GridDataPreparation.ipynb # Spatial grid data preparation (DEM Model)
│   ├── 5_DLModels.ipynb            # Deep Learning model training (executed on Kaggle)
│   ├── 6_Evaluation.ipynb          # Model evaluation and comparison
│   ├── Verification_NB.ipynb       # Grid-Verification notebook
│   └── models/                     # Trained model weights
│       ├── best_ConvLSTM. pth
│       ├── best_DeepCNN.pth
│       ├── best_DilatedCNN. pth
│       ├── best_DilatedCNN_Base_CE.pth
│       ├── best_DilatedCNN_Base_Focal.pth
│       ├── best_DilatedCNN_LowLR_CE.pth
│       └── best_FINAL_MODEL.pth
│
├── scripts/                        # Python scripts for data processing
│   ├── 1_static_terrain. py         # Static terrain feature extraction
│   ├── 2_daily_grids.py            # Daily meteorological grid generation
│   └── 3_rasterized_targets.py     # Target variable rasterization
│
├── stations_map.html               # Interactive map of IMIS stations
├── environment.yml                 # Conda environment specification
└── README.md
```

## Notebooks Pipeline

1. **1_Preprocessing.ipynb** - Data acquisition, cleaning, and initial preprocessing of IMIS weather station data
2. **2_EDA.ipynb** - Exploratory data analysis including temporal patterns, spatial distributions, and feature correlations
3. **3_BaselineModels.ipynb** - Implementation of traditional machine learning baselines (e.g., Random Forest) for comparison
4. **4_GridDataPreparation.ipynb** - Preparation of spatial grid data for CNN-based models
5. **5_DLModels.ipynb** - Training of deep learning models including CNN and ConvLSTM architectures
6. **6_Evaluation.ipynb** - Comprehensive evaluation 
7. **Verification_NB.ipynb** - Verification and validation of Grid Preparatoin

## Installation

### Prerequisites
- Python 3.11
- Conda (recommended for environment management)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/FPfenniger/AvalancheRiskProject.git
cd AvalancheRiskProject
```

2. Create the conda environment:
```bash
conda env create -f environment. yml
```

3. Activate the environment:
```bash
conda activate avalanche_project
```
