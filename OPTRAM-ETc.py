"""
Title:  Improvement and Development of the optical trapezoid model (OPTRAM)

Description:
This code utilizes the numpy, pandas, rasterio, sklearn, scipy, and matplotlib libraries to plot wet and dry edges in the scattered space of shortwave infrared transformed reflectance (STR) and a vegetation index (VI). 
The code first applies multiple filters on raster data, then determines the VI range using percentiles, and finally computes the coefficients of wet and dry edges using linear regression and visualizes them with a scatter plot.

Author: Mohammad Alavi 
Date Created: 2024/12/11

Inputs:
    - ndwi_path: Path to the NDWI raster file
    - ndvi_path: Path to the NDVI raster file
    - evi_path: Path to the EVI raster file
    - str_path: Path to the STR raster file
    - vi_path: the path of the VI raster file, e.g., FVC, EVI, MBLL, SAVI, RENDVI, etc.    
    - output_path: Path to save the filtered STR raster file

Outputs:
    - New_STR_data.tif: Filtered STR raster file
    - ***_plot.png: Scatter plot of STR vs. VI with wet and dry edges
    - Coefficients of the wet and dry edges (slope, intercept, and R-squared)

"""
import numpy as np
import pandas as pd
import rasterio as rio
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
from scipy.ndimage import gaussian_filter1d
import matplotlib.colors as mcolors


def filter_points_by_density(df: pd.DataFrame, vi: str, str_col: str, vi_min: float, vi_max: float, density_threshold: float) -> pd.DataFrame:
    """
    Filters points in a DataFrame based on density within a specified VI range.

    Args:
        df (pd.DataFrame): Input DataFrame containing VI and STR columns.
        vi (str): Name of the VI column in the DataFrame.
        str_col (str): Name of the STR column in the DataFrame.
        vi_min (float): Minimum VI value for the range.
        vi_max (float): Maximum VI value for the range.
        density_threshold (float): Minimum density value for selecting points.

    Returns:
        pd.DataFrame: DataFrame containing the filtered points.
    """
    vi_range = (df[vi] >= vi_min) & (df[vi] <= vi_max)
    df_interval = df[vi_range]

    # Calculate point density
    x_interval = df_interval[vi]
    y_interval = df_interval[str_col]
    data, x_edges, y_edges = np.histogram2d(x_interval, y_interval, bins=100, density=True)
    density = interpn((0.5 * (x_edges[1:] + x_edges[:-1]), 0.5 * (y_edges[1:] + y_edges[:-1])), data,
                      np.vstack([x_interval, y_interval]).T, method="splinef2d", bounds_error=False)

    # Filter points based on density threshold
    selected_points = df_interval[density >= density_threshold]

    return selected_points


def apply_multiple_filters(ndwi_path: str, ndvi_path: str, evi_path: str, str_path: str, output_path: str,
                            ndwi_threshold: float = -0.2, ndvi_threshold: float = 0, evi_min_threshold: float = -1,
                            evi_max_threshold: float = 1, str_max_threshold: float = 12) -> np.ndarray:
    """
    Applies multiple filters based on NDWI, NDVI, EVI and STR thresholds to STR raster data.

    Args:
        ndwi_path (str): Path to the NDWI raster file.
        ndvi_path (str): Path to the NDVI raster file.
        evi_path (str): Path to the EVI raster file.
        str_path (str): Path to the STR raster file.
        output_path (str): Path to save the filtered STR raster file.
        ndwi_threshold (float, optional): Threshold for NDWI filter. Defaults to -0.2.
        ndvi_threshold (float, optional): Threshold for NDVI filter. Defaults to 0.
        evi_min_threshold (float, optional): Minimum threshold for EVI filter. Defaults to -1.
        evi_max_threshold (float, optional): Maximum threshold for EVI filter. Defaults to 1.
        str_max_threshold (float, optional): Maximum threshold for STR filter. Defaults to 12.

    Returns:
        np.ndarray: Flattened array of filtered STR data.
    """
    try:
        # Open NDWI raster dataset
        with rio.open(ndwi_path) as ndwi_src:
            ndwi_data = ndwi_src.read()

        # Create NDWI mask based on the threshold
        ndwi_mask = ndwi_data < ndwi_threshold

        # Open NDVI raster dataset
        with rio.open(ndvi_path) as ndvi_src:
            ndvi_data = ndvi_src.read()

        # Create NDVI mask based on the threshold
        ndvi_mask = np.logical_and(ndwi_mask, ndvi_data > ndvi_threshold)

        # Open EVI raster dataset
        with rio.open(evi_path) as evi_src:
            evi_data = evi_src.read()

        # Create EVI mask based on the thresholds
        evi_mask = np.logical_and(evi_data > evi_min_threshold, evi_data < evi_max_threshold)

        # Combine all masks
        combined_mask = np.logical_and(ndwi_mask, np.logical_and(ndvi_mask, evi_mask))

        # Open STR raster dataset
        with rio.open(str_path) as str_src:
            str_data = str_src.read()

            # Apply the combined mask to filter out values in STR for each band
            for band in range(str_data.shape[0]):
                str_data[band, ~combined_mask[band]] = np.nan

            # Save the modified STR data to a new raster file if needed
            meta = str_src.meta
            with rio.open(output_path, 'w', **meta) as output:
                output.write(str_data)

            # Additional filter to remove STR values higher than a threshold
            str_data[str_data > str_max_threshold] = np.nan

        # Return the flattened STR data
        return str_data.flatten()
    except rio.RasterioIOError as e:
        print(f"Error opening raster file: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def calculate_edge_coefficients(df: pd.DataFrame, vi: str, str_col: str, vi_min: float, vi_max: float, step: float, sub_number: int,
                                edge_type: str = 'wet') -> tuple:
    """
    Calculates the linear regression coefficients (slope and intercept) for either wet or dry edges.

    Args:
        df (pd.DataFrame): Input DataFrame containing VI and STR columns.
        vi (str): Name of the VI column in the DataFrame.
        str_col (str): Name of the STR column in the DataFrame.
        vi_min (float): Minimum VI value for the range.
        vi_max (float): Maximum VI value for the range.
        step (float): Step size for VI interval.
        sub_number (int): Subinterval for filtering.
        edge_type (str, optional): Type of edge ('wet' or 'dry'). Defaults to 'wet'.

    Returns:
        tuple: A tuple containing slope, intercept, and R-squared for the linear regression model.
    """
    median_vi_intervals = []
    str_intervals = []
    
    for current_low, current_high in zip(np.arange(vi_min, vi_max, step), np.arange(vi_min + step, vi_max + step, step)):
        current_df = df[(df[vi] < current_high) & (df[vi] >= current_low)]
        if not current_df.empty:
            if edge_type == 'wet':
               str_value = current_df[str_col].max()
            elif edge_type == 'dry':
                str_value = current_df[str_col].min()
            else:
                raise ValueError("Invalid edge_type. Choose 'wet' or 'dry'.")

            str_intervals.append(str_value)
            current_median_vi = current_df[vi].median()
            median_vi_intervals.append(current_median_vi)
    
    subinterval_data = pd.DataFrame({str_col: str_intervals, vi: median_vi_intervals})
    
    filtered_str = []
    filtered_median_vi = []

    for current_low, current_high in zip(np.arange(vi_min, vi_max, step * sub_number), np.arange(vi_min + step * sub_number, vi_max + step * sub_number, step * sub_number)):
        current_data_chunk = subinterval_data[(subinterval_data[vi] < current_high) & (subinterval_data[vi] >= current_low)]
        if not current_data_chunk.empty:
            str_threshold = current_data_chunk[str_col].median() + np.nanstd(current_data_chunk[str_col])
            filtered_data = current_data_chunk[current_data_chunk[str_col] < str_threshold]

            if not filtered_data.empty:
                filtered_str.append(filtered_data[str_col].median())
                filtered_median_vi.append(filtered_data[vi].median())

    interval_data = pd.DataFrame({str_col: filtered_str, vi: filtered_median_vi})
    interval_data = interval_data.dropna()
    interval_data.reset_index(drop=True, inplace=True)

    if not interval_data.empty:
        try:
             relation = LinearRegression().fit(interval_data[[vi]], interval_data[str_col])
             intercept = relation.intercept_
             slope = relation.coef_[0]
             r_squared = r2_score(interval_data[str_col], relation.predict(interval_data[[vi]]))
             return slope, intercept, r_squared
        except Exception as e:
            print(f"Error in linear regression: {e}")
            return np.nan, np.nan, np.nan
    else:
        print("No data available for linear regression.")
        return np.nan, np.nan, np.nan


def density_scatter(x: np.ndarray, y: np.ndarray, ax: plt.Axes = None, sort: bool = True, bins: int = 100, save_path: str = None,
                    wet_slope: float = None, wet_intercept: float = None, wet_slope_basic: float = None,
                    wet_intercept_basic: float = None, dry_slope: float = None, dry_intercept: float = None,
                    dry_slope_basic: float = None, dry_intercept_basic: float = None, **kwargs) -> None:
    """
    Creates a scatter plot with density-based coloring and fitted linear lines.

    Args:
        x (np.ndarray): X-axis data (VI).
        y (np.ndarray): Y-axis data (STR).
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. Defaults to None.
        sort (bool, optional): Sort points by density. Defaults to True.
        bins (int, optional): Number of bins for density calculation. Defaults to 100.
        save_path (str, optional): Path to save the plot. Defaults to None.
        wet_slope (float, optional): Slope for the wet edge line. Defaults to None.
        wet_intercept (float, optional): Intercept for the wet edge line. Defaults to None.
        wet_slope_basic (float, optional): Slope for the basic wet edge line. Defaults to None.
        wet_intercept_basic (float, optional): Intercept for the basic wet edge line. Defaults to None.
        dry_slope (float, optional): Slope for the dry edge line. Defaults to None.
        dry_intercept (float, optional): Intercept for the dry edge line. Defaults to None.
        dry_slope_basic (float, optional): Slope for the basic dry edge line. Defaults to None.
        dry_intercept_basic (float, optional): Intercept for the basic dry edge line. Defaults to None.
        **kwargs: Additional keyword arguments for the scatter plot.
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    # Remove NaN values from x and y
    mask = np.logical_and(~np.isnan(x), ~np.isnan(y))
    x = x[mask]
    y = y[mask]

    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
    z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data,
                np.vstack([x, y]).T, method="splinef2d", bounds_error=False)

    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    cmap = plt.get_cmap('hot_r')
    norm = mcolors.LogNorm(vmin=10 ** (-4), vmax=1)
    scatter = ax.scatter(x, y, marker=".", s=0.5, c=z, cmap=cmap, edgecolor='none', norm=norm, **kwargs)
    ax.set_xlabel('MBLL')
    ax.set_ylabel('STR')

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(10)
        item.set_fontname('Times New Roman')
        item.set_weight('bold')

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Density (Log Scale)')

    # Customize legend appearance
    legend = ax.legend()
    for text in legend.get_texts():
        text.set_fontsize(6)
        text.set_fontname('Times New Roman')
        text.set_fontweight('bold')

    ax.set_xlim(0, 1.1)  # Adjust the limits as needed
    ax.set_ylim(0, 13)  # Adjust the limits as needed

    # Plot linear lines for Wet edge
    if wet_slope is not None and wet_intercept is not None:
        plot_smooth_line(ax, wet_slope, wet_intercept, 0, 1, line_color='mediumblue', label='Wet Edge (D-IRF)', linestyle='--')
    if wet_slope_basic is not None and wet_intercept_basic is not None:
        plot_smooth_line(ax, wet_slope_basic, wet_intercept_basic, 0, 1, line_color='mediumblue', label='Wet Edge (IRF)')

    # Plot linear lines for Dry edge
    if dry_slope is not None and dry_intercept is not None:
        plot_smooth_line(ax, dry_slope, dry_intercept, 0, 1, line_color='black', label='Dry Edge (D-IRF)', linestyle='--')
    if dry_slope_basic is not None and dry_intercept_basic is not None:
        plot_smooth_line(ax, dry_slope_basic, dry_intercept_basic, 0, 1, line_color='black', label='Dry Edge (IRF)')

    # Set legend properties
    ax.legend(fontsize=6, title=None)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()


def plot_smooth_line(ax: plt.Axes, slope: float, intercept: float, start: float, end: float, line_color: str, label: str,
                     linestyle: str = '-') -> None:
    """
    Plots a smooth line on a given axes.

    Args:
        ax (matplotlib.axes.Axes): The axes to plot on.
        slope (float): Slope of the line.
        intercept (float): Intercept of the line.
        start (float): Starting x value for the line.
        end (float): Ending x value for the line.
        line_color (str): Color of the line.
        label (str): Label for the line.
        linestyle (str, optional): Line style. Defaults to '-'.
    """
    x_range = np.linspace(start, end, 1000)
    y_range = slope * x_range + intercept
    ax.plot(x_range, y_range, color=line_color, label=label, linestyle=linestyle)


if __name__ == '__main__':
    # File paths
    ndwi_path = 'NDWI_data.tif'
    ndvi_path = 'NDVI_data.tif'
    evi_path = 'EVI_data.tif'
    str_path = 'STR_data.tif'
    output_path = 'New_STR_data.tif'
    vi_path = 'MBLL_data.tif'

    # Apply multiple filters
    try:
        str_data = apply_multiple_filters(ndwi_path, ndvi_path, evi_path, str_path, output_path,
                                         ndwi_threshold=-0.2, ndvi_threshold=0, evi_min_threshold=-1,
                                         evi_max_threshold=1, str_max_threshold=12)
        if str_data is None:
            print("Error in applying multiple filters.")
            exit()
    except Exception as e:
        print(f"An error occurred: {e}")
        exit()

    # Open VI raster dataset
    try:
        with rio.open(vi_path) as vi_src:
            # Read all bands of the VI data
            vi_stack = vi_src.read()

        # Create a mask based on the condition
        vi_mask = vi_stack <= 0

        # Apply the mask to all bands
        vi_stack_edited = np.where(vi_mask, np.nan, vi_stack)

        # Flatten the edited VI stack
        vi_data = vi_stack_edited.flatten()
    except rio.RasterioIOError as e:
        print(f"Error opening VI raster file: {e}")
        exit()
    except Exception as e:
        print(f"An error occurred: {e}")
        exit()
    
    # Calculate percentiles for VI range BEFORE filtering
    vi_percentile_min = np.percentile(vi_data[~np.isnan(vi_data)], 1)
    vi_percentile_max = np.percentile(vi_data[~np.isnan(vi_data)], 99)

    print(f"VI Range: Min = {vi_percentile_min}, Max = {vi_percentile_max}")


    # Apply kernel smooth to data
    smoothed_str_data = gaussian_filter1d(str_data, sigma=1.5, mode='constant', cval=np.nan)
    smoothed_vi_data = gaussian_filter1d(vi_data, sigma=1.5, mode='constant', cval=np.nan)

    # Creating a new DataFrame from STR & VI
    multi_stack = pd.DataFrame({'STR': smoothed_str_data, 'VI': smoothed_vi_data})

    # Drop rows with NaN values
    df = multi_stack.dropna()

    # Reset the index to ensure it is sequential
    df.reset_index(drop=True, inplace=True)

    # Parameters for density filter and VI intervals
    density_threshold_interval = 0.005
    density_threshold_interval_dry = 0.001
    sub_number = 10
    step = 0.001

    # Filter DataFrame based on VI percentiles
    filtered_df = df[(df['VI'] >= vi_percentile_min) & (df['VI'] <= vi_percentile_max)]

    # Apply filter_points_by_density
    selected_points_interval = filter_points_by_density(filtered_df, 'VI', 'STR', vi_percentile_min, vi_percentile_max,
                                                        density_threshold_interval)
    selected_points_interval_dry = filter_points_by_density(filtered_df, 'VI', 'STR', vi_percentile_min, vi_percentile_max,
                                                           density_threshold_interval_dry)

    # Calculate edge coefficients
    try:
        # Wet edge (density filter)
        slope_we_density, intercept_we_density, r_squared_we_density = calculate_edge_coefficients(
            selected_points_interval, 'VI', 'STR', vi_percentile_min, vi_percentile_max, step, sub_number, edge_type='wet'
        )
        print('______________________________________________________________________')
        print('Wet_Density_threshold')
        print('Wet Edge_Slope_Density_threshold:', slope_we_density)
        print('Wet Edge_Intercept_Density_threshold:', intercept_we_density)
        #print('Wet Edge_R_squared_Density_threshold:', r_squared_we_density)

        # Wet edge (basic)
        slope_we_basic, intercept_we_basic, r_squared_we_basic = calculate_edge_coefficients(
            filtered_df, 'VI', 'STR', vi_percentile_min, vi_percentile_max, step, sub_number, edge_type='wet'
        )
        print('______________')
        print('Wet_Basic')
        print('Wet Edge_Slope_Basic:', slope_we_basic)
        print('Wet Edge_Intercept_Basic:', intercept_we_basic)
        #print('Wet Edge_R_squared_Basic:', r_squared_we_basic)

        # Dry edge (density filter)
        slope_de_density, intercept_de_density, r_squared_de_density = calculate_edge_coefficients(
            selected_points_interval_dry, 'VI', 'STR', vi_percentile_min, vi_percentile_max, step, sub_number, edge_type='dry'
        )
        print('______________________________________________________________________')
        print('Dry_Density_threshold')
        print('Dry Edge_Slope_Density_threshold:', slope_de_density)
        print('Dry Edge_Intercept_Density_threshold:', intercept_de_density)
        #print('Dry Edge_R_squared_Density_threshold:', r_squared_de_density)

        # Dry edge (basic)
        slope_de_basic, intercept_de_basic, r_squared_de_basic = calculate_edge_coefficients(
            filtered_df, 'VI', 'STR', vi_percentile_min, vi_percentile_max, step, sub_number, edge_type='dry'
        )
        print('______________')
        print('Dry Edge')
        print('Dry Edge_Slope:', slope_de_basic)
        print('Dry Edge_Intercept:', intercept_de_basic)
        #print('Dry Edge_R_squared:', r_squared_de_basic)
    except Exception as e:
        print(f"An error occurred during edge coefficient calculation: {e}")
        exit()

    # Plot
    x = vi_data
    y = str_data

    density_scatter(x, y, bins=100, save_path='MBLL_data_plot.png',
                    wet_slope=slope_we_density, wet_intercept=intercept_we_density,
                    wet_slope_basic=slope_we_basic, wet_intercept_basic=intercept_we_basic,
                    dry_slope=slope_de_density, dry_intercept=intercept_de_density,
                    dry_slope_basic=slope_de_basic, dry_intercept_basic=intercept_de_basic)
