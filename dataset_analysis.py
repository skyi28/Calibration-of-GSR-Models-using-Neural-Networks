#!/usr/bin/env python3
"""
Standalone Python script for Descriptive Data Analysis.

This script performs a comprehensive descriptive analysis on the datasets used
to train a Hull-White model predicting neural network. It is designed to be
run independently of the main project's code but uses the same data sources
and file structures.

The analysis covers:
1.  Swaption Volatility Surfaces
2.  Bootstrapped Zero-Coupon Yield Curves
3.  External Market Indices (VIX, MOVE) and FX Rates (EUR/USD)

For each dataset, it computes and saves summary statistics and generates
a series of visualizations to illustrate the data's characteristics and
evolution over time. All outputs (CSVs and plots) are saved to a dedicated
'results/descriptive_analysis' directory. All plots now use the 'coolwarm'
color palette and specified custom colors for consistency.
"""

import datetime
import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple
from scipy.stats import norm # Added for normal distribution overlay

# --- Configuration: Set paths to your data folders ---

# Define the base directory for data and results
BASE_DATA_DIR = Path('data')
BASE_RESULTS_DIR = Path('results')

# Input data folders
FOLDER_ZERO_CURVES: Path = BASE_DATA_DIR / 'EUR ZERO CURVE'
FOLDER_VOLATILITY_CUBES: Path = BASE_DATA_DIR / 'EUR BVOL CUBE'
FOLDER_EXTERNAL_DATA: Path = BASE_DATA_DIR / 'EXTERNAL'

# Output directory for analysis results
OUTPUT_DIR = BASE_RESULTS_DIR / 'descriptive_analysis'


# --- Helper Functions (adapted from the user's script) ---

def load_volatility_cube(file_path: str) -> pd.DataFrame:
    """
    Reads and preprocesses an Excel file containing a swaption volatility cube.
    This function is copied from the user's script for compatibility.
    """
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        df.rename(columns={df.columns[1]: 'Type'}, inplace=True)
        # Drop any columns that are fully unnamed, which can occur during Excel import
        df.drop([col for col in df.columns if 'Unnamed' in str(col)], axis=1, inplace=True)
        # Forward-fill the 'Expiry' column to handle merged cells
        df['Expiry'] = df['Expiry'].ffill()
        return df
    except FileNotFoundError:
        print(f"Warning: Volatility cube file not found at {file_path}. Skipping.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading volatility cube from {file_path}: {e}")
        return pd.DataFrame()


def parse_tenor_to_years(tenor_str: str) -> float:
    """
    Parses a tenor string (e.g., '1Yr', '6Mo') into a float representing years.
    """
    if not isinstance(tenor_str, str):
        return np.nan
    tenor_str = tenor_str.strip().upper()
    try:
        if 'YR' in tenor_str:
            return float(tenor_str.replace('YR', ''))
        if 'MO' in tenor_str:
            return float(tenor_str.replace('MO', '')) / 12.0
    except ValueError:
        return np.nan
    return np.nan


# --- Data Loading and Preparation ---

def find_available_data_files() -> List[Tuple[datetime.date, Path, Path]]:
    """
    Scans data directories to find matching pairs of zero curve and volatility files.
    Returns a chronologically sorted list of tuples with the date and file paths.
    """
    print("--- Discovering available data files... ---")
    vol_cube_xlsx_folder = FOLDER_VOLATILITY_CUBES / 'xlsx'
    if not FOLDER_ZERO_CURVES.is_dir() or not vol_cube_xlsx_folder.is_dir():
        raise FileNotFoundError(
            f"Required data folders not found. Searched for:\n"
            f"- {FOLDER_ZERO_CURVES}\n- {vol_cube_xlsx_folder}"
        )

    available_files = []
    for zero_curve_path in FOLDER_ZERO_CURVES.glob('*.csv'):
        date_str = zero_curve_path.stem
        try:
            eval_date = datetime.datetime.strptime(date_str, '%d.%m.%Y').date()
            vol_path = vol_cube_xlsx_folder / f"{date_str}.xlsx"
            if vol_path.exists():
                available_files.append((eval_date, zero_curve_path, vol_path))
        except ValueError:
            print(f"Warning: Could not parse date from filename: {zero_curve_path.name}. Skipping.")

    available_files.sort(key=lambda x: x[0])
    print(f"Found {len(available_files)} matching data sets from {available_files[0][0]} to {available_files[-1][0]}.")
    return available_files


def download_and_load_external_data(start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    """
    Loads external market data (VIX, MOVE, EUR/USD) from a CSV file.
    If the file doesn't exist, it downloads the data using yfinance.
    """
    print("\n--- Loading or Downloading External Market Data... ---")
    csv_path = FOLDER_EXTERNAL_DATA / 'external_market_data.csv'
    FOLDER_EXTERNAL_DATA.mkdir(exist_ok=True)

    if not csv_path.exists():
        print(f"External data not found. Downloading for range {start_date} to {end_date}...")
        try:
            tickers = ['^VIX', '^MOVE', 'EURUSD=X']
            # Add one day to end_date to ensure the last day's data is included
            raw_data = yf.download(tickers, start=start_date, end=end_date + datetime.timedelta(days=1))
            if raw_data.empty:
                raise ValueError("No data downloaded from yfinance. Check tickers and date range.")

            df_ext = raw_data['Open'].copy()
            df_ext.rename(columns={'^VIX': 'VIX_Open', '^MOVE': 'MOVE_Open', 'EURUSD=X': 'EURUSD_Open'}, inplace=True)
            df_ext.to_csv(csv_path)
            print(f"External data downloaded and saved to {csv_path}")
        except Exception as e:
            print(f"ERROR: Failed to download external market data: {e}")
            return pd.DataFrame()

    df_ext = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
    # Forward-fill to ensure data for non-trading days is available
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    df_ext = df_ext.reindex(date_range).ffill().bfill()
    print("External market data loaded and prepared.")
    return df_ext


# --- Analysis and Visualization Functions ---

def analyze_external_data(df: pd.DataFrame, output_dir: Path):
    """
    Performs descriptive analysis on external market data (indices and FX).
    """
    print("\n--- Analyzing External Market Data (VIX, MOVE, EUR/USD)... ---")
    if df.empty:
        print("Skipping external data analysis: DataFrame is empty.")
        return

    # 1. Summary Statistics
    stats = df.describe().transpose()
    stats['skew'] = df.skew()
    stats['kurtosis'] = df.kurt()
    stats_path = output_dir / "external_data_summary_statistics.csv"
    stats.to_csv(stats_path)
    print(f"Summary statistics saved to: {stats_path}")

    # Use the user-specified custom colors
    colors = ['#00004c', '#0000ff', '#ff0000', '#800000']

    # 2. Time Series Visualization
    fig_ts, axes_ts = plt.subplots(3, 1, figsize=(14, 15), sharex=True)
    fig_ts.suptitle('External Market Data Over Time', fontsize=18)

    df['VIX_Open'].plot(ax=axes_ts[0], title='VIX Index (Equity Market Volatility)', color=colors[0])
    axes_ts[0].set_ylabel('Index Value')

    df['MOVE_Open'].plot(ax=axes_ts[1], title='MOVE Index (Treasury Market Volatility)', color=colors[1])
    axes_ts[1].set_ylabel('Index Value')

    df['EURUSD_Open'].plot(ax=axes_ts[2], title='EUR/USD Exchange Rate', color=colors[2])
    axes_ts[2].set_ylabel('FX Rate')
    axes_ts[2].set_xlabel('Date')

    for ax in axes_ts:
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plot_path_ts = output_dir / "external_data_time_series.png"
    plt.savefig(plot_path_ts)
    print(f"Time series plot saved to: {plot_path_ts}")
    plt.close(fig_ts)
    
    # 3. Box Plot Visualization
    fig_box, axes_box = plt.subplots(1, 3, figsize=(15, 7))
    fig_box.suptitle('Distribution of External Market Data', fontsize=16)

    sns.boxplot(y=df['VIX_Open'], ax=axes_box[0], color=colors[0])
    axes_box[0].set_title('VIX Index')
    axes_box[0].set_ylabel('Index Value')
    axes_box[0].set_xlabel('')

    sns.boxplot(y=df['MOVE_Open'], ax=axes_box[1], color=colors[1])
    axes_box[1].set_title('MOVE Index')
    axes_box[1].set_ylabel('')
    axes_box[1].set_xlabel('')

    sns.boxplot(y=df['EURUSD_Open'], ax=axes_box[2], color=colors[2])
    axes_box[2].set_title('EUR/USD Exchange Rate')
    axes_box[2].set_ylabel('')
    axes_box[2].set_xlabel('')
    
    for ax in axes_box:
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path_box = output_dir / "external_data_box_plots.png"
    plt.savefig(plot_path_box)
    print(f"Box plot chart saved to: {plot_path_box}")
    plt.close(fig_box)


def analyze_yield_curves(files: List[Tuple[datetime.date, Path, Path]], output_dir: Path):
    """
    Loads all zero curves, calculates summary statistics, and creates visualizations.
    """
    print("\n--- Analyzing Zero-Coupon Yield Curves... ---")
    all_curves = []
    for eval_date, zero_path, _ in files:
        try:
            df = pd.read_csv(zero_path)
            df['Date'] = eval_date
            all_curves.append(df)
        except Exception as e:
            print(f"Could not process yield curve {zero_path.name}: {e}")

    if not all_curves:
        print("No yield curves to analyze.")
        return

    # Combine into a single DataFrame and create a pivot table for analysis
    full_df = pd.concat(all_curves, ignore_index=True)
    full_df['Tenor'] = full_df['Tenor'].round(2)
    pivot_df = full_df.pivot_table(index='Date', columns='Tenor', values='ZeroRate')

    # 1. Summary Statistics
    stats = pivot_df.describe().transpose()
    stats['skew'] = pivot_df.skew()
    stats['kurtosis'] = pivot_df.kurt()
    stats_path = output_dir / "yield_curve_summary_statistics.csv"
    stats.to_csv(stats_path)
    print(f"Summary statistics for yield curve tenors saved to: {stats_path}")

    # 2. Visualizations
    # a) Line plot of selected tenors over time
    plt.figure(figsize=(14, 7))
    tenors_to_plot = [col for col in [1.0, 2.0, 5.0, 10.0, 30.0] if col in pivot_df.columns]
    pivot_df[tenors_to_plot].plot(title='Selected Zero-Coupon Rates Over Time', ax=plt.gca(), colormap='coolwarm')
    plt.ylabel('Zero Rate'); plt.xlabel('Date'); plt.legend(title='Tenor (Years)'); plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(output_dir / "yield_curve_selected_tenors.png"); plt.close()
    print(f"Plot of selected tenors saved.")

    # b) Heatmap of yield curve evolution
    plt.figure(figsize=(14, 10))
    sns.heatmap(pivot_df.transpose(), cmap='coolwarm', cbar_kws={'label': 'Zero Rate'})
    plt.title('Yield Curve Evolution Heatmap'); plt.xlabel('Date'); plt.ylabel('Tenor (Years)')
    plt.savefig(output_dir / "yield_curve_heatmap.png"); plt.close()
    print(f"Yield curve heatmap saved.")

    # c) 2D line plots for key dates
    key_dates = [pivot_df.index[0], pivot_df.index[len(pivot_df)//2], pivot_df.index[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    fig.suptitle('Yield Curve Shape at Different Points in Time', fontsize=16)
    line_color = '#00004c'
    for i, date in enumerate(key_dates):
        ax = axes[i]; curve = pivot_df.loc[date].dropna()
        ax.plot(curve.index, curve.values, color=line_color, marker='o', linestyle='-')
        ax.set_title(f'Yield Curve on {date.strftime("%Y-%m-%d")}'); ax.set_xlabel('Tenor (Years)'); ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    axes[0].set_ylabel('Zero Rate')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(output_dir / "yield_curve_2d_snapshots.png"); plt.close()
    print(f"2D yield curve snapshots saved.")

    # d) Bar chart of standard deviation by tenor
    plt.figure(figsize=(14, 7))
    stats['std'].sort_index().plot(kind='bar', title='Volatility of Yield Curve Tenors (Standard Deviation)', color='#00004c', edgecolor='black')
    plt.xlabel('Tenor (Years)'); plt.ylabel('Standard Deviation of Zero Rate'); plt.xticks(rotation=45); plt.grid(True, axis='y', linestyle='--', linewidth=0.5); plt.tight_layout()
    plt.savefig(output_dir / "yield_curve_std_by_tenor.png"); plt.close()
    print(f"Standard deviation bar chart for yield curve tenors saved.")

    # e) Histograms of daily changes
    daily_changes = pivot_df.diff().dropna(how='all')
    desired_tenors = [1.0, 5.0, 10.01, 30.02]
    hist_tenors = [t for t in desired_tenors if t in daily_changes.columns]
    
    if not hist_tenors:
        print("No tenors found for daily change histogram of yield curves. Skipping plot.")
    else:
        num_plots = len(hist_tenors)
        ncols = 2 if num_plots > 1 else 1
        nrows = (num_plots + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 6 * nrows), squeeze=False)
        fig.suptitle('Distribution of Daily Yield Curve Changes vs. Normal Distribution', fontsize=16)
        axes = axes.flatten()

        for i, tenor in enumerate(hist_tenors):
            ax = axes[i]
            data = daily_changes[tenor].dropna()

            if data.empty:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                ax.set_title(f'{tenor}-Year Tenor');
                continue

            sns.histplot(data, bins=50, kde=False, stat='density', ax=ax, label='Actual Distribution', color='#0000ff')
            
            # Fit normal distribution and plot
            mu, std = norm.fit(data)
            xmin, xmax = ax.get_xlim(); x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            ax.plot(x, p, 'r', linewidth=2, label='Normal Distribution')
            
            # Add Mean, Median, and Skewness
            mean_val = data.mean()
            median_val = data.median()
            skew_val = data.skew()
            ax.axvline(mean_val, color='#800000', linestyle='--', label=f'Mean: {mean_val:.4f}')
            ax.axvline(median_val, color='#800000', linestyle=':', label=f'Median: {median_val:.4f}')
            ax.text(0.95, 0.95, f'Skew: {skew_val:.2f}', transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

            ax.set_title(f'{tenor}-Year Tenor'); ax.set_xlabel('Daily Change'); ax.legend()

        for i in range(num_plots, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(output_dir / "yield_curve_daily_changes_histogram.png"); plt.close()
        print(f"Histograms of daily yield curve changes saved.")


def analyze_volatility_surfaces(files: List[Tuple[datetime.date, Path, Path]], output_dir: Path):
    """
    Loads all volatility surfaces, calculates summary statistics, and creates visualizations.
    """
    print("\n--- Analyzing Swaption Volatility Surfaces... ---")
    all_vols = []
    for eval_date, _, vol_path in files:
        df = load_volatility_cube(str(vol_path))
        if df.empty: continue
        vols = df[df['Type'] == 'Vol'].drop('Type', axis=1)
        vols = vols.melt(id_vars=['Expiry'], var_name='Tenor', value_name='Volatility_bps')
        vols['Date'] = eval_date
        all_vols.append(vols)

    if not all_vols:
        print("No volatility surfaces to analyze."); return

    full_df = pd.concat(all_vols, ignore_index=True).dropna()
    full_df['Expiry_yrs'] = full_df['Expiry'].apply(parse_tenor_to_years)
    full_df['Tenor_yrs'] = full_df['Tenor'].apply(parse_tenor_to_years)

    # 1. Summary Statistics
    stats_path = output_dir / "vol_surface_summary_statistics.csv"
    stats_df = pd.DataFrame()
    try:
        stats_df = full_df.groupby(['Expiry_yrs', 'Tenor_yrs'])['Volatility_bps'].agg(['mean', 'median', 'std', pd.Series.skew, pd.Series.kurt]).reset_index()
        stats_df.to_csv(stats_path, index=False); print(f"Summary statistics for volatility points saved to: {stats_path}")
    except Exception as e:
        print(f"Could not compute summary statistics for volatility: {e}")

    # 2. Visualizations
    # a) Line plot of key co-terminal swaption vols
    coterminal_df = full_df[np.isclose(full_df['Expiry_yrs'], full_df['Tenor_yrs'])]
    coterminal_pivot = coterminal_df.pivot_table(index='Date', columns='Tenor_yrs', values='Volatility_bps')
    plt.figure(figsize=(14, 7))
    tenors_to_plot = [col for col in [2.0, 5.0, 10.0] if col in coterminal_pivot.columns]
    coterminal_pivot[tenors_to_plot].plot(title='Selected Co-Terminal Swaption Volatilities Over Time', ax=plt.gca(), colormap='coolwarm')
    plt.ylabel('Volatility (bps)'); plt.xlabel('Date'); plt.legend(title='Expiry/Tenor (Years)'); plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(output_dir / "vol_surface_coterminal_series.png"); plt.close()
    print(f"Plot of selected co-terminal volatilities saved.")

    # b) 3D Surface plots for key dates
    key_dates = [full_df['Date'].min(), full_df['Date'].iloc[len(full_df['Date'].unique())//2], full_df['Date'].max()]
    fig = plt.figure(figsize=(24, 8)); fig.suptitle('Volatility Surface Shape at Different Points in Time', fontsize=16)
    for i, date in enumerate(key_dates):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d'); day_df = full_df[full_df['Date'] == date]
        if day_df.shape[0] < 3: print(f"Warning: Not enough data points to plot 3D surface for {date}. Skipping."); continue
        ax.plot_trisurf(day_df['Expiry_yrs'], day_df['Tenor_yrs'], day_df['Volatility_bps'], cmap=cm.coolwarm)
        ax.set_title(f'Volatility Surface on {date.strftime("%Y-%m-%d")}'); ax.set_xlabel('Expiry (Years)'); ax.set_ylabel('Tenor (Years)'); ax.set_zlabel('Volatility (bps)'); ax.view_init(elev=30, azim=-120)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(output_dir / "vol_surface_3d_snapshots.png"); plt.close()
    print(f"3D volatility surface snapshots saved.")

    # c) Heatmap of standard deviation by Expiry and Tenor
    if not stats_df.empty:
        plt.figure(figsize=(14, 10))
        std_pivot = stats_df.pivot_table(index='Expiry_yrs', columns='Tenor_yrs', values='std')
        sns.heatmap(std_pivot, cmap='coolwarm', cbar_kws={'label': 'Standard Deviation (bps)'})
        plt.title('Standard Deviation of Swaption Volatility Points'); plt.xlabel('Tenor (Years)'); plt.ylabel('Expiry (Years)'); plt.tight_layout()
        plt.savefig(output_dir / "vol_surface_std_heatmap.png"); plt.close()
        print(f"Standard deviation heatmap for volatility surface saved.")

    # d) Histograms of daily changes for co-terminal vols
    daily_changes_vol = coterminal_pivot.diff().dropna(how='all')
    desired_tenors_vol = [1.0, 5.0, 10.0, 30.0]
    hist_tenors_vol = [t for t in desired_tenors_vol if t in daily_changes_vol.columns]

    if not hist_tenors_vol:
        print("No tenors found for daily change histogram of volatilities. Skipping plot.")
    else:
        num_plots = len(hist_tenors_vol)
        ncols = 2 if num_plots > 1 else 1
        nrows = (num_plots + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 6 * nrows), squeeze=False)
        fig.suptitle('Distribution of Daily Co-Terminal Volatility Changes vs. Normal Distribution', fontsize=16)
        axes = axes.flatten()

        for i, tenor in enumerate(hist_tenors_vol):
            ax = axes[i]
            data = daily_changes_vol[tenor].dropna()

            if data.empty:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                ax.set_title(f'{int(tenor)}Y x {int(tenor)}Y Swaption Vol');
                continue

            sns.histplot(data, bins=50, kde=False, stat='density', ax=ax, label='Actual Distribution', color='#0000ff')
            
            # Fit normal distribution and plot
            mu, std = norm.fit(data)
            xmin, xmax = ax.get_xlim(); x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            ax.plot(x, p, 'r', linewidth=2, label='Normal Distribution')

            # Add Mean, Median, and Skewness
            mean_val = data.mean()
            median_val = data.median()
            skew_val = data.skew()
            ax.axvline(mean_val, color='#800000', linestyle='--', label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='#800000', linestyle=':', label=f'Median: {median_val:.2f}')
            ax.text(0.95, 0.95, f'Skew: {skew_val:.2f}', transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            
            ax.set_title(f'{int(tenor)}Y x {int(tenor)}Y Swaption Vol'); ax.set_xlabel('Daily Change (bps)'); ax.legend()
        
        for i in range(num_plots, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(output_dir / "vol_surface_daily_changes_histogram.png"); plt.close()
        print(f"Histograms of daily swaption volatility changes saved.")


# --- Main Execution Block ---

def main():
    """
    Main function to orchestrate the descriptive data analysis.
    """
    print("=" * 60)
    print(" Starting Descriptive Data Analysis Script ".center(60, "="))
    print("=" * 60)

    # --- Setup ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    print(f"Output will be saved to: {OUTPUT_DIR.resolve()}")

    try:
        # --- Data Discovery and Loading ---
        available_files = find_available_data_files()
        if not available_files:
            print("No data files found. Exiting."); return

        start_date, end_date = available_files[0][0], available_files[-1][0]

        # --- Analysis Execution ---
        # 1. External Market Data
        external_df = download_and_load_external_data(start_date, end_date)
        analyze_external_data(external_df, OUTPUT_DIR)

        # 2. Yield Curves
        analyze_yield_curves(available_files, OUTPUT_DIR)

        # 3. Volatility Surfaces
        analyze_volatility_surfaces(available_files, OUTPUT_DIR)

        print("\n" + "=" * 60)
        print(" Analysis Complete ".center(60, "="))
        print("=" * 60)

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()