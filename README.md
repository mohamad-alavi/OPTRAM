# Improvement and Development of the OPtical TRApezoid Model (OPTRAM)

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

[![Libraries](https://img.shields.io/badge/libraries-numpy%20%7C%20pandas%20%7C%20rasterio%20%7C%20sklearn%20%7C%20scipy%20%7C%20matplotlib-yellow.svg)](https://github.com/MohammadAlavi81/STR-VI-Edge-Analysis/blob/main/requirements.txt)


This repository contains a Python script designed for automated detection and analysis of wet and dry edges within the Shortwave Infrared Transformed Reflectance (STR) - Vegetation Index (VI) space, commonly known as OPTRAM. The tool enables the detailed characterization of land surface dynamics, facilitating the identification of water bodies and monitoring vegetation. Analyzing the relationship between STR and VI accurately identifies wet and dry boundaries, offering critical insights into surface moisture variations and vegetation conditions. Furthermore, this framework serves as a robust foundation for assessing evapotranspiration (ET) by integrating the STR-VI relationship with developed models and auxiliary datasets. Please refer to our publication for more information on estimating evapotranspiration (ET) using the OPTRAM-ETc model.



## Overview

The core functionality of this tool includes:

1. **Data Preprocessing:** Applying multi-criteria spectral filtering (NDWI, NDVI, EVI, and STR) to mask out invalid or noisy data points, with additional thresholds for extreme STR values.
2. **Robust Temporal Smoothing:** Performing a two-step smoothing (rolling median + Gaussian filtering) to reduce spikes and noise across time series while preserving trends.
3. **Vegetation Index (VI) Range Determination:** Establishing the effective VI range by trimming configurable percentiles (default: 2nd and 98th), ensuring stable analysis across diverse landscapes.
4.  **Edge Detection Methods:** Implementing two fully automated approaches for wet and dry edge estimation, i.e, IRF and D-IRF methods.
5. **Adaptive Density Filtering (D-IRF):** Enhancing edge detection using an adaptive density core mass filter, which selects stable high-density regions of the STR-VI feature space.
6. **Visualization:** Generating density-colored scatter plots of STR vs. VI, overlaid with wet/dry edges from both IRF and D-IRF methods.
7. **Reporting:** Exporting results into a comprehensive Excel file, including all the necessary data.


## Key Features

*   **Automated Workflow:** End-to-end pipeline from filtering to visualization and reporting.
*   **Flexible Input:** Supports multiple vegetation indices (e.g., FVC, EVI, MBLL, SAVI, RENDVI, etc.) via `vi_path`.
*   **Advanced Preprocessing:** Multi-criteria filtering combined with robust temporal smoothing.
*   **Adaptive Edge Detection:** Includes both traditional IRF and enhanced D-IRF (density-adaptive) approaches.
*   **Configurable Analysis:** User-defined thresholds, VI trimming percentiles, smoothing parameters, and density filter controls.
*   **High-Quality Visualization:** Density-based scatter plots with overlaid regression edges for easy interpretation.
*   **Comprehensive Output:** Scatter plots, filtered raster outputs, and detailed Excel reports with coefficients and data samples.

_**Note**: The files within the “data/” directory are provided as examples. You should replace them with your actual raster files. The Harmonized Sentinel-2 MSI Surface Reflectance data, which serves as the basis for various spectral indices, can be downloaded directly from Google Earth Engine by using this [code](https://code.earthengine.google.com/8d60a101dff9a29531c37233e6ceb2bc).


## Setup

1.  **Clone the Repository:**

    ```bash
    git clone [repository_url]
    ```

2.  **Navigate to the Project Directory:**

    ```bash
    cd STR-VI-Edge-Analysis
    ```

3.  **Install Required Libraries:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Data Placement:** Ensure that your raster files (e.g., NDWI, NDVI, EVI, STR, and the desired VI) are placed inside the `data/` directory.

2.  **File Path Configuration:** Update the file paths within the `OPTRAM-ETc.py` script to reflect the locations of your specific raster files.

3.  **Execution:**

    ```bash
    python code/OPTRAM-ETc.py
    ```

## Inputs

The following raster files are required as inputs:
-   `ndwi_path`: Path to the NDWI raster file.
-   `ndvi_path`: Path to the NDVI raster file.
-   `evi_path`: Path to the EVI raster file.
-   `str_path`: Path to the STR raster file.
-   `vi_path`: Path to the VI raster file (e.g., FVC, EVI, MBLL, SAVI, RENDVI).
-   `output_path`: Path to save the filtered STR raster file.

## Outputs

Upon execution, the script will produce the following outputs:

*   `Filtered_STR_Data.tif`: A GeoTIFF file containing the filtered Shortwave Infrared Transformed Reflectance (STR) data.
*   `STR_VI_Final_Plot.png`: Scatter plot of STR vs. VI, including wet and dry edges (both IRF and D-IRF)
*   ![FVC_2018-2019](https://github.com/user-attachments/assets/ee4103f1-728d-4efc-a60c-5b14e0c7f471)
*   
*   `OPTRAM_Analysis_Results.xlsx`: Excel report containing all the necessary data.
*   **Console Output:** The script prints the calculated coefficients (slope, intercept, and R-squared value) for both wet and dry edges to the console.

## Advanced Configuration

* **Threshold Adjustments:** The code has predefined thresholds in the `apply_multiple_filters` function for each spectral indices (NDWI, NDVI, EVI, STR). These thresholds are:
  -   `ndwi_threshold`: Threshold for NDWI filter, by default `-0.2`.
  -   `ndvi_threshold`: Threshold for NDVI filter, by default `0`.
  -   `evi_min_threshold`: Minimum threshold for EVI filter, by default `-1`.
  -   `evi_max_threshold`: Maximum threshold for EVI filter, by default `1`.
  -   `str_max_threshold`: Maximum threshold for STR filter, by default `12`.
You can adjust these values based on the characteristics of your study area and data quality.

## Libraries Used

*   `numpy`: For numerical operations and array manipulation.
*   `pandas`: For data analysis and handling of data frames.
*   `rasterio`: For reading and writing geospatial raster data.
*   `scikit-learn`: For machine learning models, in this case, linear regression.
*   `scipy`: For advanced scientific computing, including interpolation and signal processing.
*   `matplotlib`: For creating visualizations and plots.

## License

This project is licensed under the [Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/) (CC BY-SA 4.0).
