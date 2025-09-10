"""
Title:  Improvement and Development of the Optical TRApezoid Model (OPTRAM)

Description:
This script automates the calculation of OPTRAM wet and dry edges from multi-temporal raster data.
It applies robust pre-processing, temporal smoothing, and calculates edges using two methods: 
    IRF and a D-IRF enhanced by an adaptive density filter. 
Results are visualized in a final plot and exported to a comprehensive Excel report.

Author: Mohammad Alavi 
Date Created: 2025/08/24

Inputs:
    - ndwi_path: Path to the NDWI raster file
    - ndvi_path: Path to the NDVI raster file
    - evi_path: Path to the EVI raster file
    - str_path: Path to the STR raster file
    - vi_path: the path of the VI raster file,
    e.g., FVC, EVI, MBLL, SAVI, RENDVI, etc.    
    - output_path: Path to save the filtered STR raster file

Outputs:
    - New_STR_data.tif: Filtered STR raster file
    - STR_VI_Plot.png: Scatter plot of STR vs. VI with wet and dry edges
    - Coefficients of the wet and dry edges (slope, intercept, and R-squared)
    - OPTRAM_Analysis_Results.xlsx: including all coefficients, regression points,
    and a sample of the smoothed dataset
"""

import sys
import logging
import numpy as np
import pandas as pd
import rasterio as rio
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
from scipy.ndimage import gaussian_filter1d
import matplotlib.colors as mcolors

# ---------------------------
# Logging Configuration
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("OPTRAM-ETc")
logging.getLogger('rasterio').setLevel(logging.ERROR)

# ================================================================
# CONFIGURATION
# ================================================================
CONFIG = {
    # --- Input/Output Paths ---
    'ndwi_path': 'NDWI_data.tif',
    'ndvi_path': 'NDVI_data.tif',
    'evi_path': 'EVI_data.tif',
    'str_path': 'STR_data.tif',
    'vi_path': 'MBLL_data.tif',
    'output_str_path': 'Filtered_STR_Data.tif',
    'output_plot_path': 'STR_VI_Final_Plot.png',
    'output_excel_path': 'OPTRAM_Analysis_Results.xlsx',

    # --- Data & Plot Labels ---
    'vi_name': 'MBLL',
    'str_name': 'STR',

    # --- Initial Data Filtering Thresholds ---
    'ndwi_threshold': -0.2,
    'ndvi_threshold': 0.0,
    'evi_min_threshold': -1.0,
    'evi_max_threshold': 1.0,
    'str_max_threshold': 12.0,
    
    # --- Analysis Range Control ---
    'vi_trim_percentiles': [2, 98], # Trims the specified percentiles from VI data range

    # --- Adaptive Density Filter (D-IRF) Control ---
    'density_core_mass': 99,       # Percentage of the core data mass to retain
    'hist_bins_max_cap': 500,
    'interpn_method': 'splinef2d',

    # --- Edge Logic Parameters ---
    'vi_step': 0.001,
    'sub_number': 10,

    # --- Robust Temporal Smoothing Controls ---
    'gaussian_sigma': 1.5,
    'smoothing_rolling_median_window': 3,
    
    # --- Output Settings ---
    'excel_sample_size': 50000, # Number of points to save in the 'All_Data' sheet
    'log_level': 'INFO',
}

# Apply log level from config
log.setLevel(getattr(logging, CONFIG.get('log_level', 'INFO').upper(), logging.INFO))


# ================================================================
# FUNCTIONS
# ================================================================

def apply_multiple_filters(config: dict):
    """
    Reads and applies initial filters (NDWI, NDVI, EVI, STR) to the data stacks
    """
    try:
        with rio.open(config['ndwi_path']) as ndwi_src, \
             rio.open(config['ndvi_path']) as ndvi_src, \
             rio.open(config['evi_path']) as evi_src, \
             rio.open(config['str_path']) as str_src:

            T, H, W = str_src.count, str_src.height, str_src.width
            str_meta = str_src.meta.copy()
            STR = str_src.read().astype(np.float32)

            def read_and_align(src, name):
                if (src.transform != str_src.transform or
                    src.crs != str_src.crs or
                    src.width != str_src.width or
                    src.height != str_src.height):
                    raise ValueError(f"{name} raster is not aligned with STR raster!")
                
                if src.count == T:
                    arr = src.read().astype(np.float32)
                elif src.count == 1:
                    arr = np.repeat(src.read(1)[None, :, :].astype(np.float32), T, axis=0)
                    log.warning(f"{name} has 1 band; repeated across {T} time steps.")
                else:
                    raise ValueError(f"{name} raster has {src.count} bands; cannot align with STR ({T} bands).")
                return arr

            ndwi = read_and_align(ndwi_src, "NDWI")
            ndvi = read_and_align(ndvi_src, "NDVI")
            evi  = read_and_align(evi_src, "EVI")

            # --- Apply thresholds ---
            mask = (ndwi <= config['ndwi_threshold']) & \
                   (ndvi >= config['ndvi_threshold']) & \
                   (evi >= config['evi_min_threshold']) & \
                   (evi <= config['evi_max_threshold'])

            STR[~mask] = np.nan
            STR[(STR < 0) | (STR > config['str_max_threshold'])] = np.nan

            with rio.open(config['vi_path']) as vi_src:
                VI = read_and_align(vi_src, "VI")
                VI[np.isnan(STR)] = np.nan

            out_meta = str_meta.copy()
            out_meta.update(count=T, dtype='float32', nodata=-9999)
            with rio.open(config['output_str_path'], 'w', **out_meta) as dst:
                dst.write(np.where(np.isnan(STR), -9999, STR).astype('float32'))

            return STR, VI

    except Exception as e:
        log.exception("Failed to apply initial filters.")
        return None, None


def filter_points_by_adaptive_density(df: pd.DataFrame, vi: str, str_col: str,
                                      vi_min: float, vi_max: float,
                                      config: dict) -> pd.DataFrame:
    """
    Filters points using an adaptive threshold based on the data's core mass.
    """
    df_interval = df[(df[vi] >= vi_min) & (df[vi] <= vi_max)].copy()
    if df_interval.empty:
        return df_interval

    x, y = df_interval[vi].to_numpy(), df_interval[str_col].to_numpy()
    if x.size < 100:
        log.warning("Too few points for adaptive density filtering, returning all points.")
        return df_interval

    num_bins = min(int(np.sqrt(x.size)), config['hist_bins_max_cap'])
    
    counts, x_e, y_e = np.histogram2d(x, y, bins=num_bins, density=False)
    flat_counts = counts.flatten()
    sorted_counts = np.sort(flat_counts)[::-1]
    cumulative_mass = np.cumsum(sorted_counts)
    total_mass = np.sum(sorted_counts)
    
    target_mass = total_mass * (config['density_core_mass'] / 100.0)
    mass_threshold_index = np.searchsorted(cumulative_mass, target_mass)
    
    if mass_threshold_index >= len(sorted_counts):
        mass_threshold_index = len(sorted_counts) - 1
    
    adaptive_count_threshold = sorted_counts[mass_threshold_index]
    log.debug(f"Adaptive density count threshold calculated: {adaptive_count_threshold:.0f}")
    
    x_indices = np.searchsorted(x_e, x, side='right') - 1
    y_indices = np.searchsorted(y_e, y, side='right') - 1
    
    valid_mask = (x_indices >= 0) & (x_indices < counts.shape[0]) & \
                 (y_indices >= 0) & (y_indices < counts.shape[1])
    
    keep_mask = np.zeros_like(x, dtype=bool)
    
    point_counts = counts[x_indices[valid_mask], y_indices[valid_mask]]
    
    keep_mask[valid_mask] = point_counts >= adaptive_count_threshold
    
    return df_interval[keep_mask]


def calculate_edge_coefficients(df: pd.DataFrame, vi: str, str_col: str, vi_min: float, vi_max: float, step: float, sub_number: int,
                                          edge_type: str = 'wet') -> tuple:
    """
    Implements the IRF method for edge detection.
    """
    median_vi_intervals, str_intervals = [], []
    
    for current_low, current_high in zip(np.arange(vi_min, vi_max, step), np.arange(vi_min + step, vi_max + step, step)):
        current_df = df[(df[vi] >= current_low) & (df[vi] < current_high)]
        if not current_df.empty:
            str_value = current_df[str_col].max() if edge_type == 'wet' else current_df[str_col].min()
            str_intervals.append(str_value)
            median_vi_intervals.append(current_df[vi].median())
    
    subinterval_data = pd.DataFrame({str_col: str_intervals, vi: median_vi_intervals}).dropna()
    
    filtered_str, filtered_median_vi = [], []
    for current_low, current_high in zip(np.arange(vi_min, vi_max, step * sub_number), np.arange(vi_min + step * sub_number, vi_max + step * sub_number, step * sub_number)):
        chunk = subinterval_data[(subinterval_data[vi] >= current_low) & (subinterval_data[vi] < current_high)]
        if not chunk.empty:
            median_val, std_val = np.nanmedian(chunk[str_col]), np.nanstd(chunk[str_col])
            threshold = median_val + std_val if edge_type == 'wet' else median_val - std_val
            filtered_data = chunk[chunk[str_col] < threshold] if edge_type == 'wet' else chunk[chunk[str_col] > threshold]
            if not filtered_data.empty:
                filtered_str.append(filtered_data[str_col].median())
                filtered_median_vi.append(filtered_data[vi].median())

    final_points = pd.DataFrame({'median_vi': filtered_median_vi, str_col: filtered_str}).dropna()
    if final_points.shape[0] < 2:
        log.warning(f"[{edge_type}] Not enough points for regression after filtering.")
        return np.nan, np.nan, np.nan, pd.DataFrame()

    X, y = final_points[['median_vi']].to_numpy(), final_points[str_col].to_numpy()
    try:
        model = LinearRegression().fit(X, y)
        return float(model.coef_[0]), float(model.intercept_), r2_score(y, model.predict(X)), final_points
    except Exception as e:
        log.error(f"Error in linear regression for {edge_type} edge: {e}")
        return np.nan, np.nan, np.nan, pd.DataFrame()

def vectorized_robust_temporal_smoothing(data_stack: np.ndarray, config: dict) -> np.ndarray:
    """
    Performs a robust temporal smoothing by first removing spikes with a rolling
    median filter, then applying a Gaussian filter for a smooth trend.
    """
    if data_stack is None: return None
    nan_mask = np.isnan(data_stack)
    T, H, W = data_stack.shape
    
    df = pd.DataFrame(data_stack.reshape(T, -1))
    df_interpolated = df.interpolate(method='linear', axis=0, limit_direction='both')
    
    window = config['smoothing_rolling_median_window']
    df_median_filtered = df_interpolated.rolling(window=window, center=True, min_periods=1).median()
    
    arr_filtered = df_median_filtered.to_numpy().reshape(T, H, W)
    smoothed = gaussian_filter1d(arr_filtered, sigma=config['gaussian_sigma'], axis=0, mode='nearest')
    smoothed[nan_mask] = np.nan
    return smoothed


def density_scatter(x: np.ndarray, y: np.ndarray, config: dict, **kwargs) -> None:
    """Generates a density-colored scatter plot and overlays the fitted edge lines."""
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = (~np.isnan(x)) & (~np.isnan(y))
    x, y = x[mask], y[mask]
    if x.size == 0:
        log.warning("No valid data to plot."); return

    num_bins = min(int(np.sqrt(x.size)), config['hist_bins_max_cap'])
    
    data, x_e, y_e = np.histogram2d(x, y, bins=num_bins, density=True)
    
    x_c, y_c = 0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])
    try:
        z = interpn((x_c, y_c), data, np.vstack([x, y]).T, method=config['interpn_method'], bounds_error=False, fill_value=0.0)
    except (ValueError, IndexError):
        log.warning(f"Plot interpolation failed with '{config['interpn_method']}'. Falling back to 'linear'.")
        z = interpn((x_c, y_c), data, np.vstack([x, y]).T, method='linear', bounds_error=False, fill_value=0.0)

    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    
    cmap = plt.get_cmap('hot_r')
    
    sc = ax.scatter(x, y, marker=".", s=0.5, c=z, cmap=cmap, edgecolor='none', norm=mcolors.LogNorm(vmin=max(z.min(), 1e-4), vmax=1.0))

    cbar = fig.colorbar(sc, ax=ax); cbar.set_label('Density (Log Scale)')
    ax.set_xlabel(f"{config['vi_name']}")
    ax.set_ylabel(f"{config['str_name']}")
    ax.set_title(f"OPTRAM")
    
    def _plot_line(ax, m, c, x0, x1, color, label, style):
        if not np.isnan(m) and not np.isnan(c):
            xx = np.linspace(x0, x1, 200)
            ax.plot(xx, m * xx + c, color=color, label=label, linestyle=style, linewidth=2)
            
    _plot_line(ax, kwargs.get('wet_slope'), kwargs.get('wet_intercept'), x.min(), x.max(), 'mediumblue', 'Wet Edge (D-IRF)', '--')
    _plot_line(ax, kwargs.get('dry_slope'), kwargs.get('dry_intercept'), x.min(), x.max(), 'black', 'Dry Edge (D-IRF)', '--')
    _plot_line(ax, kwargs.get('wet_slope_basic'), kwargs.get('wet_intercept_basic'), x.min(), x.max(), 'deepskyblue', 'Wet Edge (IRF)', '-')
    _plot_line(ax, kwargs.get('dry_slope_basic'), kwargs.get('dry_intercept_basic'), x.min(), x.max(), 'dimgray', 'Dry Edge (IRF)', '-')
    
    ax.legend()
    plt.savefig(config['output_plot_path'], bbox_inches='tight', dpi=300)
    plt.show()


# ================================================================
# MAIN WORKFLOW
# ================================================================
if __name__ == '__main__':
    log.info("Starting OPTRAM parameterization workflow...")

    log.info("Step 1: Applying initial filters to raw data...")
    STR, VI = apply_multiple_filters(CONFIG)
    if STR is None: log.error("Failed to load/filter rasters. Terminating."); sys.exit(1)

    log.info("Step 2: Applying robust temporal smoothing for calculations...")
    STR_s = vectorized_robust_temporal_smoothing(STR, CONFIG)
    VI_s  = vectorized_robust_temporal_smoothing(VI, CONFIG)

    log.info("Step 3: Building DataFrame from SMOOTHED data for analysis...")
    df = pd.DataFrame({CONFIG['str_name']: STR_s.flatten(), CONFIG['vi_name']: VI_s.flatten()}).dropna()
    if df.empty: log.error("No valid data points after smoothing. Terminating."); sys.exit(1)

    vi_min, vi_max = np.percentile(df[CONFIG['vi_name']], CONFIG['vi_trim_percentiles'])
    log.info(f"Analysis VI Range ({CONFIG['vi_trim_percentiles']}% trim): min={vi_min:.4f}, max={vi_max:.4f}")
    filtered_df = df[(df[CONFIG['vi_name']] >= vi_min) & (df[CONFIG['vi_name']] <= vi_max)]

    log.info("Step 4: Computing edge coefficients...")
    
    log.info(" -> Calculating D-IRF edges")
    sel_pts = filter_points_by_adaptive_density(filtered_df, CONFIG['vi_name'], CONFIG['str_name'], vi_min, vi_max, CONFIG)
    
    we_m, we_b, we_r2, we_pts = calculate_edge_coefficients(df=sel_pts, vi=CONFIG['vi_name'], str_col=CONFIG['str_name'], vi_min=vi_min, vi_max=vi_max, step=CONFIG['vi_step'], sub_number=CONFIG['sub_number'], edge_type='wet')
    de_m, de_b, de_r2, de_pts = calculate_edge_coefficients(df=sel_pts, vi=CONFIG['vi_name'], str_col=CONFIG['str_name'], vi_min=vi_min, vi_max=vi_max, step=CONFIG['vi_step'], sub_number=CONFIG['sub_number'], edge_type='dry')

    log.info(" -> Calculating IRF edges")
    we_m_b, we_b_b, we_r2_b, we_pts_b = calculate_edge_coefficients(df=filtered_df, vi=CONFIG['vi_name'], str_col=CONFIG['str_name'], vi_min=vi_min, vi_max=vi_max, step=CONFIG['vi_step'], sub_number=CONFIG['sub_number'], edge_type='wet')
    de_m_b, de_b_b, de_r2_b, de_pts_b = calculate_edge_coefficients(df=filtered_df, vi=CONFIG['vi_name'], str_col=CONFIG['str_name'], vi_min=vi_min, vi_max=vi_max, step=CONFIG['vi_step'], sub_number=CONFIG['sub_number'], edge_type='dry')
    
    log.info("Step 5: Displaying calculated coefficients...")
    print("\n" + "="*60)
    print("      EDGE COEFFICIENTS")
    print("="*60)
    print(f"\n--- D-IRF (Adaptive Density Filter) ---")
    print(f"  Wet Edge: slope = {we_m if not np.isnan(we_m) else 'N/A':<12.6f} intercept = {we_b if not np.isnan(we_b) else 'N/A':<12.6f} R² = {we_r2 if not np.isnan(we_r2) else 'N/A':.4f}")
    print(f"  Dry Edge: slope = {de_m if not np.isnan(de_m) else 'N/A':<12.6f} intercept = {de_b if not np.isnan(de_b) else 'N/A':<12.6f} R² = {de_r2 if not np.isnan(de_r2) else 'N/A':.4f}")
    print(f"\n--- IRF (Basic - All Smoothed Points) ---")
    print(f"  Wet Edge: slope = {we_m_b if not np.isnan(we_m_b) else 'N/A':<12.6f} intercept = {we_b_b if not np.isnan(we_b_b) else 'N/A':<12.6f} R² = {we_r2_b if not np.isnan(we_r2_b) else 'N/A':.4f}")
    print(f"  Dry Edge: slope = {de_m_b if not np.isnan(de_m_b) else 'N/A':<12.6f} intercept = {de_b_b if not np.isnan(de_b_b) else 'N/A':<12.6f} R² = {de_r2_b if not np.isnan(de_r2_b) else 'N/A':.4f}")
    print("="*60 + "\n")

    log.info(f"Step 6: Saving detailed results to Excel -> {CONFIG['output_excel_path']}")
    try:
        with pd.ExcelWriter(CONFIG['output_excel_path'], engine="openpyxl") as writer:
            pd.DataFrame({'Method': ['D-IRF','D-IRF','IRF','IRF'],'Edge': ['Wet','Dry','Wet','Dry'],'Slope': [we_m,de_m,we_m_b,de_m_b],'Intercept': [we_b,de_b,we_b_b,de_b_b],'R_Squared': [we_r2,de_r2,we_r2_b,de_r2_b]}).to_excel(writer, sheet_name='Coefficients', index=False)
            if not we_pts.empty: we_pts.to_excel(writer, sheet_name='EdgePoints_Wet_D-IRF', index=False)
            if not de_pts.empty: de_pts.to_excel(writer, sheet_name='EdgePoints_Dry_D-IRF', index=False)
            if not we_pts_b.empty: we_pts_b.to_excel(writer, sheet_name='EdgePoints_Wet_IRF', index=False)
            if not de_pts_b.empty: de_pts_b.to_excel(writer, sheet_name='EdgePoints_Dry_IRF', index=False)
            if not filtered_df.empty:
                filtered_df.sample(n=min(CONFIG['excel_sample_size'], len(filtered_df))).to_excel(writer, sheet_name='Smoothed_Data_Sample', index=False)
            log.info("Excel file saved successfully.")
    except Exception as e:
        log.error(f"Could not write to Excel file. Error: {e}")

    log.info("Step 7: Creating plot on data to show model fit...")
    density_scatter(
        VI.flatten(), STR.flatten(), CONFIG,
        wet_slope=we_m, wet_intercept=we_b,
        dry_slope=de_m, dry_intercept=de_b,
        wet_slope_basic=we_m_b, wet_intercept_basic=we_b_b,
        dry_slope_basic=de_m_b, dry_intercept_basic=de_b_b
    )

    log.info("Workflow complete.")
