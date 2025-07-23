# Improvement and Development of the optical trapezoid model (OPTRAM)

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

[![Libraries](https://img.shields.io/badge/libraries-numpy%20%7C%20pandas%20%7C%20rasterio%20%7C%20sklearn%20%7C%20scipy%20%7C%20matplotlib-yellow.svg)](https://github.com/MohammadAlavi81/STR-VI-Edge-Analysis/blob/main/requirements.txt)


This repository contains a Python script designed for automated detection and analysis of wet and dry edges within the Shortwave Infrared Transformed Reflectance (STR) - Vegetation Index (VI) space, commonly known as OPTRAM. The tool enables the detailed characterization of land surface dynamics, facilitating the identification of water bodies and monitoring vegetation. Analyzing the relationship between STR and VI accurately identifies wet and dry boundaries, offering critical insights into surface moisture variations and vegetation conditions. Furthermore, this framework serves as a robust foundation for assessing evapotranspiration (ET) by integrating the STR-VI relationship with developed models and auxiliary datasets. Please refer to [our publication](https://www.sciencedirect.com/journal/remote-sensing-applications-society-and-environment) for more information on estimating evapotranspiration (ET) using the OPTRAM-ETc model.



## Overview

The core functionality of this tool includes:

1.  Data Preprocessing: Applying a series of spectral index-based filters (NDWI, NDVI, EVI, and STR) to prepare the STR data by masking out undesirable areas.
2.  **Vegetation Index (VI) Range Determination:** Automatically establishing the effective range for the VI based on its 1st and 99th percentile values, ensuring a robust analysis across varying landscapes.
3.  **Point Density Filtering:** Identifying and selecting high-density point clusters within the STR-VI feature space, crucial for accurate edge detection, by using `filter_points_by_density` function.
4.  **Edge Coefficient Calculation:** Employing linear regression to compute the slopes and intercepts of both wet and dry edges. The dry edge is determined from the lower bounds and wet edge from the upper bounds of STR values within moving VI intervals, thus defining the bounds of the feature space.
5.  **Visualization:** Creating a scatter plot of STR vs. VI that includes the calculated wet and dry edge lines, enhancing visual interpretation and understanding of the STR-VI relationship using `density_scatter` function.

## Key Features

*   **Automated Processing:** Streamlined data processing and analysis pipeline.
*   **Flexibility:** Supports various vegetation indices by adjusting the `vi_path` variable.
*   **Robust Filtering:** Uses multi-criteria spectral filtering for high-quality data processing.
*   **Density-Based Edge Detection:** Utilizes point density to refine edge detection.
*   **Comprehensive Visualization:** Produces informative scatter plots for effective analysis.

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

*   `New_STR_data.tif`: A GeoTIFF file containing the filtered Shortwave Infrared Transformed Reflectance (STR) data.
*   `VI_data_plot.png`: A scatter plot visualizing the relationship between STR and VI, including the fitted wet and dry edge lines.
*   ![FVC_20m_2018-2019](https://github.com/user-attachments/assets/ee4103f1-728d-4efc-a60c-5b14e0c7f471)
*   
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
