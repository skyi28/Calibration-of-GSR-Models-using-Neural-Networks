"""
This script is designed to calibrate Hull-White GSR models using a neural network.

This version is a fully operational implementation for performing a "real run" of
both hyperparameter tuning (using a custom Keras HyperModel and the Hyperband algorithm)
and final model training.

It now includes three distinct workflows controlled by settings:
1.  Full hyperparameter tuning and final model training.
2.  Final model training using pre-existing best hyperparameters.
3.  Evaluation-only mode, which loads a pre-trained model artifact and runs it
    on the test set.

Crucially, it now saves a comprehensive CSV file with all evaluation results,
making it easy to generate plots and tables for academic papers or reports.

WARNING: This script is computationally intensive and is intended for long-running
execution on powerful hardware.
"""
import datetime
import glob
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Filter out tensorflow information messages
import sys
import time
import pandas as pd
from pandas._typing import ArrayLike
import numpy as np
import QuantLib as ql
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from numpy.typing import NDArray
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from matplotlib import cm
from concurrent.futures import ThreadPoolExecutor # Tried ProcessPoolExecutor but it was slower for I/O-bound tasks
import bisect
import traceback
import json
import keras_tuner as kt
import joblib
import yfinance as yf
import random
import shap

#--------------------CONFIG--------------------
# Set to False to skip the initial data preparation steps if they have already been run
PREPROCESS_CURVES: bool = False
BOOTSTRAP_CURVES: bool = False

FOLDER_SWAP_CURVES: str = r'data/EUR SWAP CURVE'
FOLDER_ZERO_CURVES: str = r'data/EUR ZERO CURVE'
FOLDER_VOLATILITY_CUBES: str = r'data/EUR BVOL CUBE'
FOLDER_EXTERNAL_DATA: str = r'data/EXTERNAL'
FOLDER_MODELS: str = r'results/neural_network/models'
FOLDER_HYPERPARAMETERS: str = r'results/neural_network/hyperparameters'


#--------------------PREPROCESS CURVES--------------------
if PREPROCESS_CURVES:
    print("--- Starting: Preprocessing Raw Swap Curves ---")
    processed_folder: str = os.path.join(FOLDER_SWAP_CURVES, 'processed')
    raw_folder: str = os.path.join(FOLDER_SWAP_CURVES, 'raw')
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)
    if not os.path.exists(raw_folder):
        print(f"Warning: Raw data folder not found at {raw_folder}")

    for entry_name in os.listdir(raw_folder):
        if entry_name.endswith('.xlsx'):
            path_swap_curve: str = os.path.join(raw_folder, entry_name)
            swap_curve: pd.DataFrame = pd.read_excel(path_swap_curve, engine='openpyxl')
            
            processed_swap_curve: pd.DataFrame = swap_curve[['Term', 'Unit']].copy()

            swap_curve_date_str: str = entry_name.split('.xlsx')[0]
            swap_curve_date: datetime.datetime = datetime.datetime.strptime(swap_curve_date_str, '%d.%m.%Y')
            
            processed_swap_curve_dates: list[datetime.datetime] = []
            for i in range(len(swap_curve)):
                new_date: datetime.datetime = swap_curve_date + datetime.timedelta(days=int(30.4375 * swap_curve['Term'].iloc[i]) if swap_curve['Unit'].iloc[i] == 'MO'
                                                                                   else int(365.25 * swap_curve['Term'].iloc[i]))
                processed_swap_curve_dates.append(new_date)
            
            processed_swap_curve['Date'] = processed_swap_curve_dates
            processed_swap_curve['Rate'] = swap_curve[['Bid', 'Ask']].mean(axis=1) / 100
            
            processed_swap_curve = processed_swap_curve[['Date', 'Rate', 'Term', 'Unit']]
            
            processed_swap_curve.to_csv(os.path.join(processed_folder, f"{swap_curve_date_str}.csv"), index=False)
    print("--- Finished: Preprocessing ---")


#--------------------BOOTSTRAP ZERO CURVES--------------------
def bootstrap_zero_curve_with_quantlib(
    processed_swap_curve: pd.DataFrame,
    valuation_date: datetime.datetime
) -> pd.DataFrame:
    """
    Bootstraps a zero-coupon curve from a given swap curve DataFrame using QuantLib.
    
    Args:
        processed_swap_curve (pd.DataFrame): The DataFrame containing the swap curve data.
        valuation_date (datetime.datetime): The valuation date of the yield curve.
    
    Returns:
        pd.DataFrame: The bootstrapped zero curve.
    """
    ql_valuation_date = ql.Date(valuation_date.day, valuation_date.month, valuation_date.year)
    ql.Settings.instance().evaluationDate = ql_valuation_date

    calendar: ql.Calendar = ql.TARGET()
    day_count: ql.Actual365Fixed = ql.Actual365Fixed()

    df: pd.DataFrame = processed_swap_curve.sort_values(by='Date').reset_index(drop=True)
    df = df[df['Date'] > valuation_date]

    rate_helpers: list[ql.SwapRateHelper] = []
    swap_index: ql.Euribor6M = ql.Euribor6M(ql.YieldTermStructureHandle())

    for _, row in df.iterrows():
        rate: float = float(row['Rate'])
        unit: str = row['Unit'].strip().upper()
        term_length: int = int(row['Term'])
        
        if unit == 'MO':
            tenor_period = ql.Period(term_length, ql.Months)
        elif unit == 'YR':
            tenor_period = ql.Period(term_length, ql.Years)
        else:
            raise ValueError(f"Unsupported tenor unit: {row['Unit']}")

        helper: ql.SwapRateHelper = ql.SwapRateHelper(
            ql.QuoteHandle(ql.SimpleQuote(rate)),
            tenor_period,
            calendar,
            ql.Annual,  # Fixed leg frequency for EUR swaps is Annual
            ql.Unadjusted,
            day_count,
            swap_index
        )
        rate_helpers.append(helper)

    yield_curve: ql.PiecewiseLogLinearDiscount = ql.PiecewiseLogLinearDiscount(
        ql_valuation_date, rate_helpers, day_count
    )
    yield_curve.enableExtrapolation()

    results: list[dict] = []
    for row in df.itertuples():
        maturity_date = row.Date
        ql_date = ql.Date(maturity_date.day, maturity_date.month, maturity_date.year)
        tenor_frac = day_count.yearFraction(ql_valuation_date, ql_date)
        zero_rate = yield_curve.zeroRate(ql_date, day_count, ql.Continuous).rate()
        discount_factor = yield_curve.discount(ql_date)
        
        results.append({
            'Date': maturity_date,
            'Tenor': tenor_frac,
            'DiscountFactor': discount_factor,
            'ZeroRate': zero_rate
        })

    return pd.DataFrame(results)

if BOOTSTRAP_CURVES:
    print("--- Starting: Bootstrapping Zero Curves ---")
    preprocessed_folder = os.path.join(FOLDER_SWAP_CURVES, 'processed')
    if not os.path.exists(FOLDER_ZERO_CURVES):
        os.makedirs(FOLDER_ZERO_CURVES)

    for entry_name in os.listdir(preprocessed_folder):
        if entry_name.endswith('.csv'):
            swap_curve_date_str: str = entry_name.split('.csv')[0]
            swap_curve_date: datetime.datetime = datetime.datetime.strptime(swap_curve_date_str, '%d.%m.%Y')
            processed_swap_curve: pd.DataFrame = pd.read_csv(os.path.join(preprocessed_folder, entry_name), parse_dates=['Date'])
            zero_curve: pd.DataFrame = bootstrap_zero_curve_with_quantlib(processed_swap_curve, swap_curve_date)
            zero_curve.to_csv(os.path.join(FOLDER_ZERO_CURVES, f"{swap_curve_date_str}.csv"), index=False)
    print("--- Finished: Bootstrapping ---")
    sys.exit(0)


#--------------------DATA DISCOVERY AND SPLITTING--------------------
def load_and_split_data_chronologically(
    zero_curve_folder: str,
    vol_cube_folder: str,
    train_split_percentage: float = 50.0,
    validation_split_percentage: float = 20.0,
    load_all: bool = True
) -> Tuple[List[Tuple[datetime.date, str, str]], List[Tuple[datetime.date, str, str]], List[Tuple[datetime.date, str, str]]]:
    """
    Discovers data files and splits them chronologically. If load_all is False,
    only loads test files.
    """
    print("\n--- Discovering and splitting data files chronologically ---")
    vol_cube_xlsx_folder: str = os.path.join(vol_cube_folder, 'xlsx')
    if not os.path.exists(zero_curve_folder) or not os.path.exists(vol_cube_xlsx_folder):
        raise FileNotFoundError(f"Data folders not found. Searched for:\n- {zero_curve_folder}\n- {vol_cube_xlsx_folder}")

    available_files: List[Tuple[datetime.date, str, str]] = []
    for entry_name in os.listdir(zero_curve_folder):
        if entry_name.endswith('.csv'):
            date_str: str = entry_name.replace('.csv', '')
            eval_date: datetime.date = datetime.datetime.strptime(date_str, '%d.%m.%Y').date()
            zero_path: str = os.path.join(zero_curve_folder, entry_name)
            vol_path: str = os.path.join(vol_cube_xlsx_folder, f"{date_str}.xlsx")
            if os.path.exists(vol_path):
                available_files.append((eval_date, zero_path, vol_path))

    if not available_files:
        raise ValueError("No matching pairs of zero curve and volatility data found.")

    available_files.sort(key=lambda x: x[0])
    total_files: int = len(available_files)
    print(f"Found {total_files} complete data sets from {available_files[0][0]} to {available_files[-1][0]}.")

    train_end_index = int(total_files * train_split_percentage / 100.0)
    validation_end_index = train_end_index + int(total_files * validation_split_percentage / 100.0)

    test_files: List[Tuple[datetime.date, str, str]] = available_files[validation_end_index:]

    if not load_all:
        print(f"Loading test data only: {len(test_files)} files.")
        return [], [], test_files
    
    train_files: List[Tuple[datetime.date, str, str]] = available_files[:train_end_index]
    validation_files: List[Tuple[datetime.date, str, str]] = available_files[train_end_index:validation_end_index]

    if not train_files or not validation_files or not test_files:
        raise ValueError("Data splitting resulted in one or more empty sets. Adjust percentages or add more data.")

    print(f"Splitting data: {len(train_files)} for training, {len(validation_files)} for validation, and {len(test_files)} for testing.")
    return train_files, validation_files, test_files


#--------------------HELPER FOR LOADING VOL CUBE--------------------
def load_volatility_cube(file_path: str) -> pd.DataFrame:
    """
    Reads an Excel file containing a volatility cube from the given file path.

    The Excel file is expected to have the following format:

    - The first column is the Expiry column, which is expected to be contiguous.
    - The second column is the Type column, which indicates whether the row is a Volatility or a Strike.
    - The remaining columns are the tenors, which are used to index the Volatility and Strike values.

    The function returns a Pandas DataFrame with the following columns:
    - Expiry: The contiguous expiry dates.
    - Type: The type of the row (Vol or Strike).
    - Tenors: The tenors as columns, with the Volatility or Strike values as the cell values.

    The function does the following:
    - Reads the Excel file into a Pandas DataFrame.
    - Renames the second column to 'Type'.
    - Drops any columns with 'Unnamed' in their name.
    - Forward-fills the Expiry column to make the expiry dates contiguous.

    :param file_path: The file path of the Excel file containing the volatility cube.
    :return: A Pandas DataFrame containing the volatility cube.
    """
    df: pd.DataFrame = pd.read_excel(file_path, engine='openpyxl')
    df.rename(columns={df.columns[1]: 'Type'}, inplace=True)
    for col in df.columns:
        if 'Unnamed' in str(col): df.drop(col, axis=1, inplace=True)
    df['Expiry'] = df['Expiry'].ffill()
    return df

#--------------------HELPER AND PLOTTING FUNCTIONS--------------------
def parse_tenor(tenor_str: str) -> ql.Period:
    """
    Parses a tenor string (e.g., '1Yr', '6Mo') into a QuantLib Period object.

    :param tenor_str: The tenor string to parse.
    :return: A QuantLib Period object representing the tenor.
    """
    tenor_str = tenor_str.strip().upper()
    if 'YR' in tenor_str: return ql.Period(int(tenor_str.replace('YR', '')), ql.Years)
    if 'MO' in tenor_str: return ql.Period(int(tenor_str.replace('MO', '')), ql.Months)
    raise ValueError(f"Could not parse tenor string: {tenor_str}")

def parse_tenor_to_years(tenor_str: str) -> float:
    """
    Parses a tenor string (e.g., '1Yr', '6Mo') into a float representing years.

    :param tenor_str: The tenor string to parse.
    :return: A float representing the tenor in years.
    """
    tenor_str = tenor_str.strip().upper()
    if 'YR' in tenor_str: return float(int(tenor_str.replace('YR', '')))
    if 'MO' in tenor_str: return int(tenor_str.replace('MO', '')) / 12.0
    raise ValueError(f"Could not parse tenor string to years: {tenor_str}")

def create_ql_yield_curve(
    zero_curve_df: pd.DataFrame, eval_date: datetime.date
) -> ql.RelinkableYieldTermStructureHandle:
    """
    Creates a QuantLib YieldTermStructure from a bootstrapped zero curve DataFrame.
    
    Args:
        zero_curve_df (pd.DataFrame): The DataFrame containing the zero curve data.
        eval_date (datetime.date): The evaluation date of the yield curve.
    
    Returns:
        ql.RelinkableYieldTermStructureHandle: The created yield curve.
    """
    ql_eval_date: ql.Date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
    dates: list[ql.Date] = [ql_eval_date] + [ql.Date(d.day, d.month, d.year) for d in pd.to_datetime(zero_curve_df['Date'])]
    rates: list[float] = [zero_curve_df['ZeroRate'].iloc[0]] + zero_curve_df['ZeroRate'].tolist()
    term_structure: ql.ZeroCurve = ql.ZeroCurve(dates, rates, ql.Actual365Fixed(), ql.TARGET(), ql.Linear(), ql.Continuous, ql.Annual)
    term_structure.enableExtrapolation()
    handle: ql.RelinkableYieldTermStructureHandle = ql.RelinkableYieldTermStructureHandle()
    handle.linkTo(term_structure)
    return handle

def prepare_calibration_helpers(
    vol_cube_df: pd.DataFrame,
    term_structure_handle: ql.RelinkableYieldTermStructureHandle,
    settings: Dict
) -> List[Tuple[ql.SwaptionHelper, str, str]]:
    """
    Parses the swaption volatility cube and creates a list of QuantLib SwaptionHelper objects,
    optionally filtering by minimum expiry, tenor, or for co-terminal swaptions only.

    Args:
        vol_cube_df (pd.DataFrame): The DataFrame containing the swaption volatility cube.
        term_structure_handle (ql.RelinkableYieldTermStructureHandle): The yield curve to use for pricing.
        settings (Dict): A dictionary of settings which may contain 'min_expiry_years', 
            'min_tenor_years', and 'use_coterminal_only'.

    Returns:
        List[Tuple[ql.SwaptionHelper, str, str]]: A list of tuples, where each tuple contains a SwaptionHelper,
            the expiry string and the tenor string.
    """
    min_expiry_years: float = settings.get("min_expiry_years", 0.0)
    min_tenor_years: float = settings.get("min_tenor_years", 0.0)
    use_coterminal_only: bool = settings.get("use_coterminal_only", False)
    
    helpers_with_info: List[Tuple[ql.SwaptionHelper, str, str]] = []
    vols_df: pd.DataFrame = vol_cube_df[vol_cube_df['Type'] == 'Vol'].set_index('Expiry')
    strikes_df: pd.DataFrame = vol_cube_df[vol_cube_df['Type'] == 'Strike'].set_index('Expiry')
    swap_index: ql.Euribor6M = ql.Euribor6M(term_structure_handle)
    
    for expiry_str in vols_df.index:
        for tenor_str in vols_df.columns:
            if tenor_str == 'Type': continue

            if use_coterminal_only:
                if parse_tenor_to_years(expiry_str) != parse_tenor_to_years(tenor_str):
                    continue

            vol, strike = vols_df.loc[expiry_str, tenor_str], strikes_df.loc[expiry_str, tenor_str]
            if pd.isna(vol) or pd.isna(strike): continue
            
            if parse_tenor_to_years(expiry_str) < min_expiry_years or parse_tenor_to_years(tenor_str) < min_tenor_years:
                continue

            vol_handle: ql.QuoteHandle = ql.QuoteHandle(ql.SimpleQuote(float(vol) / 10000.0))
            helper: ql.SwaptionHelper = ql.SwaptionHelper(
                parse_tenor(expiry_str), parse_tenor(tenor_str), vol_handle, swap_index,
                ql.Period(6, ql.Months), swap_index.dayCounter(), swap_index.dayCounter(),
                term_structure_handle, ql.Normal
            )
            helpers_with_info.append((helper, expiry_str, tenor_str))
            
    return helpers_with_info

def plot_calibration_results(results_df: pd.DataFrame, eval_date: datetime.date, model_save_dir: str):
    """
    Shows diagrams as plots in the same figure after the calibration.

    If the data is from a full surface calibration, it plots 3D surfaces of the 
    observed market volatilities, model implied volatilities, and the difference.
    If the data is from a co-terminal calibration (i.e., Expiry equals Tenor),
    it plots 2D line and bar charts which are more appropriate for the data.

    Args:
        results_df (pd.DataFrame): The DataFrame with the calibration results.
        eval_date (datetime.date): The evaluation date of the yield curve.
        model_save_dir (str): The directory where the plot image will be saved.
    """
    plot_data: pd.DataFrame = results_df.dropna(subset=['MarketVol', 'ModelVol', 'Difference_bps']).copy()
    if plot_data.empty:
        print(f"\nCould not generate plots for {eval_date}: No valid data points available.")
        return

    # Determine if the data is one-dimensional (co-terminal)
    is_coterminal = np.allclose(plot_data['Expiry'].values, plot_data['Tenor'].values)

    if is_coterminal:
        # --- 2D PLOTTING FOR CO-TERMINAL SWAPTIONS ---
        print(f"Co-terminal data detected for {eval_date}. Generating 2D plots.")
        
        # Sort data by tenor for a clean line plot
        plot_data = plot_data.sort_values(by='Tenor').reset_index()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle(f'Hull-White Co-terminal Calibration for {eval_date}', fontsize=16)

        # Subplot 1: Market vs Model Volatility
        ax1.plot(plot_data['Tenor'], plot_data['MarketVol'], 'o-', label='Market Volatility')
        ax1.plot(plot_data['Tenor'], plot_data['ModelVol'], 'x--', label='Model Volatility')
        ax1.set_title('Market vs. Model Implied Volatilities')
        ax1.set_xlabel('Tenor (Years)')
        ax1.set_ylabel('Volatility (bps)')
        ax1.legend()
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Subplot 2: Difference in BPS
        colors = ['red' if x > 0 else 'green' for x in plot_data['Difference_bps']]
        ax2.bar(plot_data['Tenor'], plot_data['Difference_bps'], width=0.5, color=colors, alpha=0.8)
        ax2.axhline(0, color='black', linestyle='--', linewidth=1)
        ax2.set_title('Difference (Model - Market)')
        ax2.set_xlabel('Tenor (Years)')
        ax2.set_ylabel('Difference (bps)')
        ax2.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    else:
        # --- 3D PLOTTING FOR FULL SURFACE ---
        X = plot_data['Expiry'].values
        Y = plot_data['Tenor'].values
        Z_market: ArrayLike = plot_data['MarketVol'].values
        Z_model = plot_data['ModelVol'].values
        Z_diff: ArrayLike = plot_data['Difference_bps'].values
        
        fig = plt.figure(figsize=(24, 8))
        fig.suptitle(f'Hull-White Calibration Volatility Surfaces for {eval_date}', fontsize=16)
        
        # Plot 1: Market Volatilities
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        ax1.set_title('Observed Market Volatilities (bps)')
        surf1 = ax1.plot_trisurf(X, Y, Z_market, cmap=cm.viridis, antialiased=True, linewidth=0.1)
        ax1.set_xlabel('Expiry (Years)')
        ax1.set_ylabel('Tenor (Years)')
        ax1.set_zlabel('Volatility (bps)')
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, pad=0.1)
        ax1.view_init(elev=30, azim=-120)
        
        # Plot 2: Model Volatilities
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        ax2.set_title('Model Implied Volatilities (bps)')
        surf2 = ax2.plot_trisurf(X, Y, Z_model, cmap=cm.viridis, antialiased=True, linewidth=0.1)
        ax2.set_xlabel('Expiry (Years)')
        ax2.set_ylabel('Tenor (Years)')
        ax2.set_zlabel('Volatility (bps)')
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, pad=0.1)
        market_min, market_max = np.nanmin(Z_market), np.nanmax(Z_market)
        ax2.set_zlim(market_min * 0.9, market_max * 1.1)
        ax2.view_init(elev=30, azim=-120)
        
        # Plot 3: Difference
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        ax3.set_title('Difference (Model - Market) (bps)')
        surf3 = ax3.plot_trisurf(X, Y, Z_diff, cmap=cm.coolwarm, antialiased=True, linewidth=0.1)
        ax3.set_xlabel('Expiry (Years)')
        ax3.set_ylabel('Tenor (Years)')
        ax3.set_zlabel('Difference (bps)')
        fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10, pad=0.1)
        max_reasonable_diff: float = np.nanmax(np.abs(Z_diff))
        ax3.set_zlim(-max_reasonable_diff, max_reasonable_diff)
        ax3.view_init(elev=30, azim=-120)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save and show the plot for both cases
    plt.savefig(os.path.join(model_save_dir, f'CalibrationPlot_{eval_date}.png'))
    plt.show()

#--------------------TENSORFLOW NEURAL NETWORK CALIBRATION HELPERS--------------------
def _format_time(seconds: float) -> str:
    """
    Convert seconds to a string in the format "HH:MM:SS"

    Args:
        seconds (float): The time in seconds

    Returns:
        str: Formatted string
    """
    s: int = int(round(seconds)); h, r = divmod(s, 3600); m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def _get_step_dates_from_expiries(ql_eval_date: ql.Date, included_expiries_yrs: List[float], num_segments: int) -> List[ql.Date]:
    """
    Calculates the step dates for piecewise parameters based on available expiry dates.
    
    Args:
        ql_eval_date (ql.Date): The evaluation date of the yield curve.
        included_expiries_yrs (List[float]): The sorted list of expiry years.
        num_segments (int): The number of segments to split the expiry years into.
    
    Returns:
        List[ql.Date]: The step dates of the piecewise parameters.
    """
    if num_segments <= 1: return []
    unique_expiries: list[float] = sorted(list(set(included_expiries_yrs)))
    if len(unique_expiries) < num_segments: num_segments = len(unique_expiries)
    if num_segments <= 1: return []
    indices: NDArray = np.linspace(0, len(unique_expiries) - 1, num_segments + 1).astype(int)[1:-1]
    time_points_in_years: list[float] = [unique_expiries[i] for i in indices]
    return [ql_eval_date + ql.Period(int(y * 365.25), ql.Days) for y in time_points_in_years]

def _expand_params_to_unified_timeline(initial_params_quotes: List[ql.SimpleQuote], param_step_dates: List[ql.Date], unified_step_dates: List[ql.Date]) -> List[ql.QuoteHandle]:
    """
    Expands a list of initial parameter quotes to align with a unified timeline.

    Args:
        initial_params_quotes (List[ql.SimpleQuote]): The initial parameter quotes as SimpleQuote objects.
        param_step_dates (List[ql.Date]): The original step dates associated with the initial parameters.
        unified_step_dates (List[ql.Date]): The new, unified step dates to align the parameter quotes with.

    Returns:
        List[ql.QuoteHandle]: A list of QuoteHandles that represent the initial parameters expanded to the unified timeline.
    """

    initial_params_handles: list[ql.QuoteHandle] = [ql.QuoteHandle(q) for q in initial_params_quotes]
    if not unified_step_dates: return initial_params_handles
    if not param_step_dates: return [initial_params_handles[0]] * (len(unified_step_dates) + 1)
    expanded_handles: list[ql.QuoteHandle] = []; time_intervals: list[float] = [float('-inf')] + [d.serialNumber() for d in unified_step_dates] + [float('inf')]
    for i in range(len(time_intervals) - 1):
        mid_point_serial: float = (time_intervals[i] + time_intervals[i+1]) / 2
        original_date_serials: list[int] = [d.serialNumber() for d in param_step_dates]
        idx: int = bisect.bisect_right(original_date_serials, mid_point_serial)
        expanded_handles.append(initial_params_handles[idx])
    return expanded_handles

def extract_raw_features(
    term_structure_handle: ql.RelinkableYieldTermStructureHandle,
    eval_date: ql.Date,
    external_data: pd.DataFrame,
    feature_tenors: List[float] = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
) -> List[float]:
    """
    Extracts a raw (unscaled) feature vector from a yield curve and external market data.
    The vector includes rates, key slopes, curvature, MOVE, VIX, and their ratio.
    """
    day_counter: ql.Actual365Fixed = ql.Actual365Fixed()
    
    # Get the base rates as before
    rates: list[float] = [term_structure_handle.zeroRate(eval_date + ql.Period(int(ty * 365.25), ql.Days), day_counter, ql.Continuous).rate() for ty in feature_tenors]
    
    # Helper to get rates for specific tenors
    def get_rate(tenor_in_years: float) -> float:
        if tenor_in_years < 1.0:
            num_months = int(tenor_in_years * 12)
            return term_structure_handle.zeroRate(eval_date + ql.Period(num_months, ql.Months), day_counter, ql.Continuous).rate()
        
        days = int(tenor_in_years * 365.25)
        return term_structure_handle.zeroRate(eval_date + ql.Period(days, ql.Days), day_counter, ql.Continuous).rate()

    # Get the specific rates needed for slopes and curvature
    rate_3m = get_rate(0.25)
    rate_2y = get_rate(2.0)
    rate_5y = get_rate(5.0)
    rate_10y = get_rate(10.0)
    rate_30y = get_rate(30.0)
    
    # --- CALCULATE NEW FEATURES ---
    # 1. Slopes
    slope_3m10s = rate_10y - rate_3m
    slope_2s10s = rate_10y - rate_2y
    slope_5s30s = rate_30y - rate_5y
    
    # 2. Curvature (Butterfly)
    curvature_2s5s10s = (2 * rate_5y) - rate_2y - rate_10y

    # 3. External Data (MOVE, VIX, Ratio)
    py_date = datetime.date(eval_date.year(), eval_date.month(), eval_date.dayOfMonth())
    pd_timestamp = pd.to_datetime(py_date)
    
    move_value = external_data.loc[pd_timestamp, 'MOVE_Open']
    vix_value = external_data.loc[pd_timestamp, 'VIX_Open']
    
    if vix_value > 1e-6: # Avoid division by zero or near-zero
        move_vix_ratio = move_value / vix_value
    else:
        move_vix_ratio = 1.0  # Assign a neutral default value

    # Combine all features into a single list
    all_features = rates + [slope_3m10s, slope_2s10s, slope_5s30s, curvature_2s5s10s, move_value, vix_value, move_vix_ratio]
    
    return all_features

def prepare_nn_features(
    term_structure_handle: ql.RelinkableYieldTermStructureHandle,
    eval_date: ql.Date,
    scaler: StandardScaler,
    external_data: pd.DataFrame,
    feature_tenors: List[float] = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
) -> np.ndarray:
    """
    Prepares a feature vector for a neural network, extracting raw features and then scaling them.
    This now includes yield curve rates, slopes, curvature, MOVE, VIX, and their ratio.

    Args:
        term_structure_handle (ql.RelinkableYieldTermStructureHandle): The yield curve term structure.
        eval_date (ql.Date): The evaluation date of the yield curve.
        scaler (StandardScaler): A pre-fitted StandardScaler object from scikit-learn.
        external_data (pd.DataFrame): DataFrame with MOVE and VIX index data.
        feature_tenors (List[float], optional): The tenors to use for the feature vector. Defaults to [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0].

    Returns:
        np.ndarray: The prepared and scaled feature vector as a numpy array.
    """
    # Step 1: Extract all raw features
    raw_features = extract_raw_features(term_structure_handle, eval_date, external_data, feature_tenors)
    
    # Step 2: Scale the raw features using the pre-fitted scaler
    return scaler.transform(np.array(raw_features).reshape(1, -1))


class ResidualParameterModel(tf.keras.Model):
    def __init__(self, total_params_to_predict: int, upper_bound: float, layers: list, activation: str, use_dropout: bool, dropout_rate: float, **kwargs):
        super().__init__(**kwargs)
        self.total_params_to_predict = total_params_to_predict
        self.upper_bound_value = upper_bound
        self.upper_bound = tf.constant(self.upper_bound_value, dtype=tf.float64)
        
        self._config = {
            'total_params_to_predict': total_params_to_predict, 'upper_bound': upper_bound,
            'layers': layers, 'activation': activation,
            'use_dropout': use_dropout, 'dropout_rate': dropout_rate
        }

        self.hidden_layers = []
        for num_neurons in layers:
            self.hidden_layers.append(tf.keras.layers.Dense(num_neurons, activation=activation, dtype=tf.float64))
            if use_dropout:
                self.hidden_layers.append(tf.keras.layers.Dropout(dropout_rate, dtype=tf.float64))
        
        self.output_layer = tf.keras.layers.Dense(
            self.total_params_to_predict, dtype=tf.float64,
            kernel_initializer='zeros', bias_initializer='zeros'
        )

    def call(self, inputs, training=False) -> tf.Tensor:
        feature_vector, initial_logits = inputs
        x = feature_vector
        for layer in self.hidden_layers:
            if isinstance(layer, tf.keras.layers.Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        
        delta_logits = self.output_layer(x)
        final_logits = initial_logits + delta_logits
        return self.upper_bound * tf.keras.activations.sigmoid(final_logits)

    def get_config(self):
        """
        Gets the configuration of the model as a dictionary.

        This method is needed to support model serialization and deserialization.

        Returns:
            dict: The configuration of the model.
        """
        config = super().get_config()
        config.update(self._config)
        return config

    @classmethod
    def from_config(cls, config):
        """
        Instantiates a ResidualParameterModel from its configuration (serializedName -> constructor arguments).

        Args:
            config (dict): The configuration of the model.

        Returns:
            ResidualParameterModel: The instantiated model.
        """

        return cls(**config)

def evaluate_model_on_day(
    eval_date: datetime.date,
    zero_curve_df: pd.DataFrame,
    vol_cube_df: pd.DataFrame,
    calibrated_params: NDArray[np.float64],
    settings: dict
) -> Tuple[float, pd.DataFrame]:
    """
    Evaluates the performance of the Hull-White model on a given day. The evaluation consists of calculating the root
    mean squared error (RMSE) between the model's predicted swaption volatilities and the market volatilities.

    Args:
        eval_date (datetime.date): The date on which the evaluation is being performed.
        zero_curve_df (pd.DataFrame): The bootstrapped zero curve for the evaluation date.
        vol_cube_df (pd.DataFrame): The volatility cube for the evaluation date.
        calibrated_params (List[float]): The calibrated parameters of the Hull-White model.
        settings (dict): The settings for the evaluation.

    Returns:
        Tuple[float, pd.DataFrame]: A tuple containing the final RMSE in bps and a DataFrame with the results of the
            evaluation for each swaption. The DataFrame has columns 'ExpiryStr', 'TenorStr', 'MarketVol', 'ModelVol', and
            'Difference', where 'Difference' is the difference between the model and market volatilities. If the
            evaluation was not successful, the returned DataFrame is empty.
    """
    ql_eval_date: ql.Date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
    ql.Settings.instance().evaluationDate = ql_eval_date
    term_structure_handle: ql.RelinkableYieldTermStructureHandle = create_ql_yield_curve(zero_curve_df, eval_date)
    helpers_with_info: List[Tuple[ql.SwaptionHelper, str, str]] = prepare_calibration_helpers(vol_cube_df, term_structure_handle, settings)
    if not helpers_with_info: return float('nan'), pd.DataFrame()
    num_a_params: int = settings['num_a_segments'] if settings['optimize_a'] else 0
    calibrated_as: list[float] = calibrated_params[:num_a_params]
    calibrated_sigmas: list[float] = calibrated_params[num_a_params:]
    if not settings['optimize_a']: calibrated_as: list[float] = settings['initial_guess'][:num_a_params] if num_a_params > 0 else []
    included_expiries_yrs = sorted(list(set([parse_tenor_to_years(expiry) for _, expiry, _ in helpers_with_info])))
    a_step_dates: list[ql.Date] = _get_step_dates_from_expiries(ql_eval_date, included_expiries_yrs, settings['num_a_segments'])
    sigma_step_dates: list[ql.Date] = _get_step_dates_from_expiries(ql_eval_date, included_expiries_yrs, settings['num_sigma_segments'])
    unified_step_dates: list[ql.Date] = sorted(list(set(a_step_dates + sigma_step_dates)))
    reversion_quotes: list[ql.SimpleQuote] = [ql.SimpleQuote(p) for p in calibrated_as]
    sigma_quotes: list[ql.SimpleQuote] = [ql.SimpleQuote(p) for p in calibrated_sigmas]
    if not settings['optimize_a']: reversion_quotes = [ql.SimpleQuote(p) for p in settings['initial_guess'][:num_a_params]] if num_a_params > 0 else [ql.SimpleQuote(0.01)]
    expanded_reversion_handles: list[ql.QuoteHandle] = _expand_params_to_unified_timeline(reversion_quotes, a_step_dates, unified_step_dates)
    expanded_sigma_handles: list[ql.QuoteHandle] = _expand_params_to_unified_timeline(sigma_quotes, sigma_step_dates, unified_step_dates)
    final_model: ql.Gsr = ql.Gsr(term_structure_handle, unified_step_dates, expanded_sigma_handles, expanded_reversion_handles, 61.0)
    final_engine: ql.Gaussian1dSwaptionEngine = ql.Gaussian1dSwaptionEngine(final_model, settings['pricing_engine_integration_points'], 7.0, True, False, term_structure_handle)
    results_data: list[dict] = []; squared_errors: list[float] = []
    for helper, expiry_str, tenor_str in helpers_with_info:
        helper.setPricingEngine(final_engine)
        market_vol_bps: float = helper.volatility().value() * 10000
        try:
            model_npv: float = helper.modelValue()
            model_vol: float = helper.impliedVolatility(model_npv, 1e-4, 500, 0.0001, 1.0)
            model_vol_bps: float = model_vol * 10000; error_bps = model_vol_bps - market_vol_bps
            squared_errors.append((model_vol - helper.volatility().value())**2)
        except (RuntimeError, ValueError):
            model_vol_bps, error_bps = float('nan'), float('nan')
        results_data.append({'ExpiryStr': expiry_str, 'TenorStr': tenor_str, 'MarketVol': market_vol_bps, 'ModelVol': model_vol_bps, 'Difference_bps': error_bps})
    results_df: pd.DataFrame = pd.DataFrame(results_data)
    if not results_df.empty:
        results_df['Expiry'] = results_df['ExpiryStr'].apply(parse_tenor_to_years); results_df['Tenor'] = results_df['TenorStr'].apply(parse_tenor_to_years)
    final_rmse_bps: float = np.sqrt(np.mean(squared_errors)) * 10000 if squared_errors else float('nan')
    return final_rmse_bps, results_df


# ------------------- SHAP ANALYSIS FUNCTION -------------------
def perform_and_save_shap_analysis(
    model: tf.keras.Model,
    scaler: StandardScaler,
    initial_logits_tensor: tf.Tensor,
    test_files_list: List[Tuple[datetime.date, str, str]],
    external_market_data: pd.DataFrame,
    settings: Dict,
    output_dir: str
):
    """
    Performs SHAP analysis on the trained model to explain its predictions on the
    test set and saves the resulting plots to the model's output directory.

    Args:
        model (tf.keras.Model): The trained Keras model to be explained.
        scaler (StandardScaler): The fitted scaler used for feature transformation.
        initial_logits_tensor (tf.Tensor): The fixed initial logits tensor used by the model.
        test_files_list (List[Tuple[datetime.date, str, str]]): List of test data file paths.
        external_market_data (pd.DataFrame): DataFrame with external market data (MOVE, VIX).
        settings (Dict): The configuration dictionary for the model run.
        output_dir (str): The directory where the SHAP plots will be saved.
    """
    print("\n--- Starting SHAP Analysis for Model Interpretability ---")
    
    # 1. Define Feature and Output Names for clear plot labels
    feature_tenors = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
    feature_names = [f'{int(t)}y_rate' for t in feature_tenors] + [
        'slope_3m10y', 'slope_2y10y', 'slope_5y30y', 'curvature_2y5y10y',
        'MOVE_Open', 'VIX_Open', 'MOVE_VIX_Ratio'
    ]
    num_a = settings['num_a_segments'] if settings.get('optimize_a', False) else 0
    num_sigma = settings['num_sigma_segments']
    output_names = [f'a_{i+1}' for i in range(num_a)] + [f'sigma_{i+1}' for i in range(num_sigma)]

    # 2. Prepare the test data features for SHAP analysis
    test_features_list = []
    print("Preparing test data for SHAP explanation...")
    for eval_date, zero_path, _ in test_files_list:
        zero_curve_df = pd.read_csv(zero_path, parse_dates=['Date'])
        term_structure = create_ql_yield_curve(zero_curve_df, eval_date)
        ql_eval_date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
        raw_features = extract_raw_features(term_structure, ql_eval_date, external_market_data, feature_tenors)
        test_features_list.append(raw_features)
    
    if not test_features_list:
        print("Warning: No test data found to perform SHAP analysis.")
        return
        
    scaled_test_features = scaler.transform(np.array(test_features_list))
    test_features_df = pd.DataFrame(scaled_test_features, columns=feature_names)

# 3. Create a wrapper for the model using the Keras Functional API so SHAP can inspect it
    features_input = tf.keras.Input(shape=(scaled_test_features.shape[1],), dtype=tf.float64, name="features")
    
    # The initial_logits tensor needs to be tiled to match the batch size of the features_input
    # We use a Lambda layer to perform this dynamic tiling within the model graph
    def tile_logits(features):
        num_samples = tf.shape(features)[0]
        return tf.tile(initial_logits_tensor, [num_samples, 1])

    logits_input = tf.keras.layers.Lambda(tile_logits)(features_input)
    
    # Call the original model with the two inputs
    outputs = model((features_input, logits_input))
    
    # Create the new, inspectable wrapper model
    shap_wrapped_model = tf.keras.Model(inputs=features_input, outputs=outputs)
    
    # 4. Create SHAP DeepExplainer
    # A subset of the test data is used as the background distribution for the explainer.
    # This provides a baseline for comparing feature contributions.
    print("Creating SHAP explainer...")
    # background_data = shap.sample(scaled_test_features, 100)
    background_data = scaled_test_features
    explainer = shap.DeepExplainer(shap_wrapped_model, background_data)
    
    # 5. Calculate SHAP values for all instances in the test set
    print("Calculating SHAP values... (This may take a while)")
    shap_values = explainer.shap_values(scaled_test_features)
    
    # 6. Generate and save plots for each of the model's output parameters
    print(f"Generating and saving {len(output_names) * 2} SHAP plots...")
    for i, param_name in enumerate(output_names):
        print(f"  -> Plotting for parameter: {param_name}")
        
        # Summary Plot (dot plot): Shows the impact of each feature on the model output.
        shap.summary_plot(shap_values[:, :, i], test_features_df, show=False)
        fig = plt.gcf()
        fig.suptitle(f'SHAP Value Summary for Parameter "{param_name}"', fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(output_dir, f'SHAP_summary_{param_name}.png'), bbox_inches='tight')
        plt.close(fig)

        # Feature Importance Plot (bar plot): Ranks features by their mean absolute SHAP value.
        shap.summary_plot(shap_values[:, :, i], test_features_df, plot_type='bar', show=False)
        fig = plt.gcf()
        fig.suptitle(f'SHAP Feature Importance for Parameter "{param_name}"', fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(output_dir, f'SHAP_importance_{param_name}.png'), bbox_inches='tight')
        plt.close(fig)
        
    print("--- SHAP Analysis Completed ---")


# ------------------- HYPERPARAMETER TUNING AND TRAINING COMPONENTS -------------------
def _calculate_loss_for_day(params: np.ndarray, ql_eval_date, term_structure_handle, helpers_with_info, settings) -> float:
    """
    The core, computationally-heavy loss calculation using QuantLib.
    Now includes an asymmetric penalty for underestimation.
    """
    try:
        num_a = settings['num_a_segments'] if settings['optimize_a'] else 0
        a_params, sigma_params = params[:num_a], params[num_a:]
        reversion_quotes = [ql.SimpleQuote(p) for p in a_params]
        sigma_quotes = [ql.SimpleQuote(p) for p in sigma_params]
        if not settings['optimize_a']:
            reversion_quotes = [ql.SimpleQuote(p) for p in settings['initial_guess'][:num_a]] if num_a > 0 else [ql.SimpleQuote(0.01)]
        
        included_expiries = sorted(list(set([parse_tenor_to_years(e) for _, e, _ in helpers_with_info])))
        a_steps = _get_step_dates_from_expiries(ql_eval_date, included_expiries, settings['num_a_segments'])
        s_steps = _get_step_dates_from_expiries(ql_eval_date, included_expiries, settings['num_sigma_segments'])
        unified_steps = sorted(list(set(a_steps + s_steps)))

        exp_rev_h = _expand_params_to_unified_timeline(reversion_quotes, a_steps, unified_steps)
        exp_sig_h = _expand_params_to_unified_timeline(sigma_quotes, s_steps, unified_steps)

        thread_model = ql.Gsr(term_structure_handle, unified_steps, exp_sig_h, exp_rev_h, 61.0)
        thread_engine = ql.Gaussian1dSwaptionEngine(thread_model, settings['pricing_engine_integration_points'], 7.0, True, False, term_structure_handle)
        
        instrument_batch_size: int = int(len(helpers_with_info) * settings.get("instrument_batch_size_percentage", 100) / 100)
        if settings.get("instrument_batch_size_percentage", 100) / 100 < 1:
            helpers_to_price = random.sample(helpers_with_info, instrument_batch_size)
        else:
            helpers_to_price = helpers_with_info
        
        weighted_squared_errors = []
        underestimation_penalty = settings.get("underestimation_penalty", 1.0)

        for helper, _, _ in helpers_to_price:
            helper.setPricingEngine(thread_engine)
            market_vol = helper.volatility().value()
            model_val = helper.modelValue()
            implied_vol = helper.impliedVolatility(model_val, 1e-3, 500, market_vol * 0.01, market_vol * 4)
            
            error = implied_vol - market_vol
            
            # --- NEW: Asymmetric Loss Calculation ---
            if error < 0: # This is an underestimation (implied_vol < market_vol)
                weighted_squared_errors.append((error**2) * underestimation_penalty)
            else: # This is an overestimation
                weighted_squared_errors.append(error**2)
        
        return float(np.mean(weighted_squared_errors)) if weighted_squared_errors else 1e6
    except (RuntimeError, ValueError):
        print(traceback.format_exc())
        return 1e6

def _perform_training_step(model, optimizer, feature_vector, initial_logits, ql_eval_date, term_structure_handle, helpers_with_info, settings):
    """
    Executes one forward and backward pass for a single day of data.
    """
    @tf.custom_gradient
    def ql_loss_on_params(params_tensor):
        # Forward pass: calculate the loss for the unperturbed parameters.
        # This loss_val will be returned and can be reused for the forward difference method.
        base_params = params_tensor.numpy()[0]
        loss_val = _calculate_loss_for_day(base_params, ql_eval_date, term_structure_handle, helpers_with_info, settings)
        
        def grad_fn(dy):
            # Backward pass: calculate gradient using the method specified in settings.
            h = settings['h_relative']
            gradient_method = settings.get('gradient_method', 'forward').lower()

            # --- Select Gradient Calculation Method ---
            if gradient_method == 'central':
                # Central Difference: More accurate, 2N calls to loss function.
                def _calc_single_grad(i):
                    p_plus = base_params.copy(); p_plus[i] += h
                    p_minus = base_params.copy(); p_minus[i] -= h
                    loss_plus = _calculate_loss_for_day(p_plus, ql_eval_date, term_structure_handle, helpers_with_info, settings)
                    loss_minus = _calculate_loss_for_day(p_minus, ql_eval_date, term_structure_handle, helpers_with_info, settings)
                    return (loss_plus - loss_minus) / (2 * h)
            else:
                # Forward Difference (default): Less accurate, N calls to loss function (reuses loss_val).
                def _calc_single_grad(i):
                    p_plus = base_params.copy()
                    p_plus[i] += h
                    loss_plus = _calculate_loss_for_day(p_plus, ql_eval_date, term_structure_handle, helpers_with_info, settings)
                    # Reuse loss_val from the forward pass
                    return (loss_plus - loss_val) / h
            
            with ThreadPoolExecutor(max_workers=settings.get("num_threads", os.cpu_count() or 1)) as executor:
                grad = np.array(list(executor.map(_calc_single_grad, range(len(base_params)))))
            
            return tf.constant([dy.numpy() * grad], dtype=tf.float64)
        
        return tf.constant(loss_val, dtype=tf.float64), grad_fn

    with tf.GradientTape() as tape:
        predicted_params = model((feature_vector, initial_logits), training=True)
        loss = ql_loss_on_params(predicted_params)
    
    grads = tape.gradient(loss, model.trainable_variables)
    if any(g is not None for g in grads):
        clipped_grads = [(tf.clip_by_norm(g, settings['gradient_clip_norm']) if g is not None else None) for g in grads]
        optimizer.apply_gradients(zip(clipped_grads, model.trainable_variables))
    
    return loss

class HullWhiteHyperModel(kt.HyperModel):
    """
    A Keras Tuner HyperModel that defines the search space and the custom training loop.
    """
    def build(self, hp: kt.HyperParameters) -> ResidualParameterModel:
        """Builds the Keras model with hyperparameters."""
        num_layers = hp.Int('num_layers', 1, 5)
        activation = hp.Choice('activation', ['relu', 'tanh'])
        use_dropout = hp.Boolean('use_dropout')
        dropout_rate = hp.Float('dropout_rate', 0.1, 0.5, step=0.1) if use_dropout else 0.0

        layers = [hp.Int(f'neurons_{i}', 16, 128, step=16) for i in range(num_layers)]

        # These parameters are fixed during the search but needed for model instantiation
        total_params = TF_NN_CALIBRATION_SETTINGS['num_a_segments'] + TF_NN_CALIBRATION_SETTINGS['num_sigma_segments']
        upper_bound = TF_NN_CALIBRATION_SETTINGS['upper_bound']

        model = ResidualParameterModel(
            total_params_to_predict=total_params, upper_bound=upper_bound,
            layers=layers, activation=activation, use_dropout=use_dropout,
            dropout_rate=dropout_rate, dtype=tf.float64
        )
        return model

    def fit(self, hp, model, loaded_train_data, loaded_val_data, settings, feature_scaler, initial_logits, external_data, **kwargs):
        """Overrides the default fit method to implement the custom training loop."""
        learning_rate = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # --- NEW: Define underestimation_penalty as a hyperparameter ---
        penalty = hp.Float("underestimation_penalty", 1.0, 4.0, step=0.5)
        # Create a trial-specific settings copy to inject the penalty
        trial_settings = settings.copy()
        trial_settings['underestimation_penalty'] = penalty

        epochs = kwargs.get('epochs', 1)
        trial = None
        for callback in kwargs.get('callbacks', []):
            if hasattr(callback, 'trial'):
                trial = callback.trial
                break
        trial_id = trial.trial_id if trial else "unknown"
        
        for epoch in range(epochs):
            print(f"\nTrial {trial_id} | Starting Epoch {epoch + 1}/{epochs} | Penalty: {penalty:.2f}")
            epoch_losses = []
            total_days: int = len(loaded_train_data)
            for day_idx, (eval_date, zero_df, vol_df) in enumerate(loaded_train_data):
                print(f"\r  Training day {day_idx + 1}/{total_days}...", end='', flush=True)
                ql_eval_date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
                ql.Settings.instance().evaluationDate = ql_eval_date
                term_structure = create_ql_yield_curve(zero_df, eval_date)
                helpers = prepare_calibration_helpers(vol_df, term_structure, trial_settings)
                if not helpers: continue
                
                feature_vec = prepare_nn_features(term_structure, ql_eval_date, feature_scaler, external_data)
                # Pass the trial-specific settings to the training step
                loss = _perform_training_step(model, optimizer, tf.constant(feature_vec, dtype=tf.float64), initial_logits, ql_eval_date, term_structure, helpers, trial_settings)
                if not np.isnan(loss):
                    epoch_losses.append(np.sqrt(loss) * 10000)
            
            avg_epoch_rmse = np.mean(epoch_losses) if epoch_losses else float('inf')
            print(f"\n  Trial {trial_id} | Epoch {epoch+1}/{epochs} | Avg Train RMSE (Weighted): {avg_epoch_rmse:.2f} bps")

            val_losses = []
            for eval_date, zero_df, vol_df in loaded_val_data:
                term_structure = create_ql_yield_curve(zero_df, eval_date)
                ql_eval_date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
                feature_vec = prepare_nn_features(term_structure, ql_eval_date, feature_scaler, external_data)
                predicted_params = model((tf.constant(feature_vec, dtype=tf.float64), initial_logits), training=False).numpy()[0]
                # Pass trial_settings here as well for consistency, although it doesn't use the penalty
                rmse, _ = evaluate_model_on_day(eval_date, zero_df, vol_df, predicted_params, trial_settings)
                if not np.isnan(rmse): val_losses.append(rmse)
            
            avg_val_rmse = np.mean(val_losses) if val_losses else float('inf')
            print(f"  Trial {trial_id} | Epoch {epoch+1}/{epochs} | Avg Validation RMSE (Unweighted): {avg_val_rmse:.2f} bps")
            
        return { 'val_rmse': avg_val_rmse }

#--------------------MAIN EXECUTION LOGIC--------------------
if __name__ == '__main__':
    try:
        print("="*80)
        print("WARNING: This script is computationally intensive and will take a very")
        print("         long time to run. A full hyperparameter search may take days.")
        print("="*80)
        
        os.makedirs(FOLDER_MODELS, exist_ok=True)
        os.makedirs(FOLDER_HYPERPARAMETERS, exist_ok=True)

        TF_NN_CALIBRATION_SETTINGS = {
            # --- WORKFLOW CONTROLS ---
            # Mode 1: Tune and Train
            #   evaluate_only = False, perform_hyperparameter_tuning = True
            # Mode 2: Train from best HPs
            #   evaluate_only = False, perform_hyperparameter_tuning = False
            # Mode 3: Evaluate a pre-trained model
            #   evaluate_only = True
            "evaluate_only": False,
            "perform_hyperparameter_tuning": False,
            "model_evaluation_dir": r"results\neural_network\models\model_20251008_110404",
            "hyperband_settings": {
                "max_epochs": 1, "factor": 3,
                "directory": "results/neural_network/hyperband_tuner",
                "project_name": "hull_white_calibration"
            },
            # --- MODEL AND TRAINING SETTINGS ---
            "num_a_segments": 1, "num_sigma_segments": 7, "optimize_a": True, # Increased sigma segments
            "instrument_batch_size_percentage": 100,
            "upper_bound": 0.1, "pricing_engine_integration_points": 32,
            "num_epochs": 2, "h_relative": 1e-7, # Increased epochs
            "initial_guess": [0.02, 0.0002, 0.0002, 0.00017, 0.00017, 0.00017, 0.00017, 0.00017], # Adjusted for more segments
            "gradient_method": "forward", # either "forward" or "central"
            "gradient_clip_norm": 2.0, "num_threads": os.cpu_count() or 1,
            "min_expiry_years": 2.0, "min_tenor_years": 2.0,
            "use_coterminal_only": True, # If True, only uses swaptions where expiry equals tenor.
            "SAVE_MODEL_DIR_NAME": f"model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        final_model = None
        feature_scaler = None
        initial_logits = None
        model_save_dir = None
        
        if TF_NN_CALIBRATION_SETTINGS['evaluate_only']:
            print("\n--- WORKFLOW: EVALUATION-ONLY MODE ---")
            model_save_dir = TF_NN_CALIBRATION_SETTINGS['model_evaluation_dir']
            if not os.path.isdir(model_save_dir):
                raise FileNotFoundError(f"Model evaluation directory not found: {model_save_dir}")

            print(f"Loading model artifact from: {model_save_dir}")
            final_model = tf.keras.models.load_model(
                os.path.join(model_save_dir, 'model.keras'), 
                custom_objects={'ResidualParameterModel': ResidualParameterModel}
            )
            feature_scaler = joblib.load(os.path.join(model_save_dir, 'feature_scaler.joblib'))
            initial_logits = tf.constant(np.load(os.path.join(model_save_dir, 'initial_logits.npy')), dtype=tf.float64)
            print("Model artifact loaded successfully.")
            
            _, _, test_files = load_and_split_data_chronologically(
                FOLDER_ZERO_CURVES, FOLDER_VOLATILITY_CUBES, load_all=False
            )

        else:
            print("\n--- WORKFLOW: TRAINING MODE ---")
            train_files, val_files, test_files = load_and_split_data_chronologically(
                FOLDER_ZERO_CURVES, FOLDER_VOLATILITY_CUBES, train_split_percentage=50, validation_split_percentage=20
            )

        # --- DOWNLOAD AND PREPARE EXTERNAL MARKET DATA (MOVE, VIX) ---
        external_data_csv_path = os.path.join(FOLDER_EXTERNAL_DATA, 'external_market_data.csv')
        all_files = train_files + val_files + test_files
        start_date = all_files[0][0]
        end_date = all_files[-1][0]

        if not os.path.exists(external_data_csv_path):
            print(f"\n--- External market data not found. Downloading for range {start_date} to {end_date} ---")
            os.makedirs(FOLDER_EXTERNAL_DATA, exist_ok=True)
            try:
                tickers_to_download = ['^MOVE', '^VIX']
                # Add one day to end_date to make yfinance download inclusive
                raw_data = yf.download(tickers_to_download, start=start_date, end=end_date + datetime.timedelta(days=1))
                if raw_data.empty:
                    raise ValueError("No data downloaded from yfinance. Check tickers and date range.")
                
                # We only need the 'Open' prices
                external_data_df = raw_data['Open'].copy()
                external_data_df.rename(columns={'^MOVE': 'MOVE_Open', '^VIX': 'VIX_Open'}, inplace=True)
                
                external_data_df.to_csv(external_data_csv_path)
                print(f"External market data (MOVE, VIX) saved to {external_data_csv_path}")
            except Exception as e:
                print(f"ERROR: Could not download external market data. {e}")
                sys.exit(1)
        
        print("\n--- Loading and preparing external market data ---")
        external_data = pd.read_csv(external_data_csv_path, parse_dates=['Date'], index_col='Date')
        
        # Fill missing values (weekends, holidays) using forward then backward fill
        full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        external_data = external_data.reindex(full_date_range).ffill().bfill()
        external_data.index.name = 'Date'
        print("External market data prepared.")
        
        if not TF_NN_CALIBRATION_SETTINGS['evaluate_only']:
            print("\n--- Preparing Feature Scaler using Training Data ---")
            all_training_features = []
            for eval_date, zero_path, _ in train_files:
                zero_curve_df = pd.read_csv(zero_path, parse_dates=['Date'])
                term_structure = create_ql_yield_curve(zero_curve_df, eval_date)
                ql_eval_date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
                
                # Extract the new, wider raw feature set
                raw_features = extract_raw_features(term_structure, ql_eval_date, external_data)
                all_training_features.append(raw_features)
            
            # Fit the scaler on the new, richer feature set
            feature_scaler = StandardScaler()
            feature_scaler.fit(np.array(all_training_features))
            print("Feature scaler has been fitted to the training data (including all new features).")

            print("\n--- Pre-loading all data into memory ---")
            loaded_train_data = [(d, pd.read_csv(zp, parse_dates=['Date']), load_volatility_cube(vp)) for d, zp, vp in train_files]
            loaded_val_data = [(d, pd.read_csv(zp, parse_dates=['Date']), load_volatility_cube(vp)) for d, zp, vp in val_files]
            print("All training and validation data has been loaded.")
            
            # Make sure initial guess matches number of parameters
            num_params_to_predict = (TF_NN_CALIBRATION_SETTINGS['num_a_segments'] if TF_NN_CALIBRATION_SETTINGS['optimize_a'] else 0) + TF_NN_CALIBRATION_SETTINGS['num_sigma_segments']
            if len(TF_NN_CALIBRATION_SETTINGS['initial_guess']) != num_params_to_predict:
                 raise ValueError(f"Length of 'initial_guess' ({len(TF_NN_CALIBRATION_SETTINGS['initial_guess'])}) must match the total number of parameters to predict ({num_params_to_predict}).")

            p_scaled = np.clip(np.array(TF_NN_CALIBRATION_SETTINGS['initial_guess'], dtype=np.float64) / TF_NN_CALIBRATION_SETTINGS['upper_bound'], 1e-9, 1 - 1e-9)
            initial_logits = tf.constant([np.log(p_scaled / (1 - p_scaled))], dtype=tf.float64)
            
            best_hyperparameters = None

            if TF_NN_CALIBRATION_SETTINGS['perform_hyperparameter_tuning']:
                print("\n--- Starting Hyperparameter Tuning with Hyperband ---")
                hypermodel = HullWhiteHyperModel()
                tuner = kt.Hyperband(
                    hypermodel=hypermodel,
                    objective=kt.Objective("val_rmse", direction="min"),
                    max_epochs=TF_NN_CALIBRATION_SETTINGS['hyperband_settings']['max_epochs'],
                    factor=TF_NN_CALIBRATION_SETTINGS['hyperband_settings']['factor'],
                    directory=TF_NN_CALIBRATION_SETTINGS['hyperband_settings']['directory'],
                    project_name=TF_NN_CALIBRATION_SETTINGS['hyperband_settings']['project_name']
                )
                
                tuning_start_time = time.monotonic()
                print(f"Hyperband Tuner initialized. Starting search at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                tuner.search(
                    loaded_train_data=loaded_train_data, loaded_val_data=loaded_val_data,
                    settings=TF_NN_CALIBRATION_SETTINGS, feature_scaler=feature_scaler,
                    initial_logits=initial_logits, external_data=external_data
                )
                
                tuning_elapsed = time.monotonic() - tuning_start_time
                print("\n--- Hyperparameter Tuning Finished ---")
                print(f"\nHyperparameter tuning completed in {_format_time(tuning_elapsed)}.")
                best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
                print("Best Hyperparameters Found:")
                for hp, value in best_hyperparameters.values.items(): print(f"  - {hp}: {value}")
                    
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"best_hyperparameters_{timestamp}.json"
                filepath = os.path.join(FOLDER_HYPERPARAMETERS, filename)
                with open(filepath, 'w') as f: json.dump(best_hyperparameters.values, f, indent=4)
                print(f"Best hyperparameters saved to {filepath}")
                timing_log = {'hyperparameter_tuning_time_seconds': tuning_elapsed}
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                timing_log_filepath = os.path.join(FOLDER_HYPERPARAMETERS, f"tuning_time_{timestamp}.json")
                with open(timing_log_filepath, 'w') as f: json.dump(timing_log, f, indent=4)
                print(f"Tuning time log saved to {timing_log_filepath}")
            else:
                print("\n--- Loading Best Hyperparameters from File ---")
                list_of_files = glob.glob(os.path.join(FOLDER_HYPERPARAMETERS, '*.json'))
                if not list_of_files:
                    raise FileNotFoundError("perform_hyperparameter_tuning is False, but no hyperparameter files found.")
                
                latest_file = max(list_of_files, key=os.path.getctime)
                print(f"Loading hyperparameters from: {latest_file}")
                with open(latest_file, 'r') as f: loaded_hps = json.load(f)
                best_hyperparameters = kt.HyperParameters()
                for key, value in loaded_hps.items(): best_hyperparameters.Fixed(key, value)
            
            print("\n--- Training Final Model using Best Hyperparameters ---")
            
            # --- NEW: Use the best penalty found by the tuner ---
            best_penalty = best_hyperparameters.get('underestimation_penalty')
            TF_NN_CALIBRATION_SETTINGS['underestimation_penalty'] = best_penalty
            print(f"Using best underestimation_penalty for final training: {best_penalty:.2f}")

            final_model = HullWhiteHyperModel().build(best_hyperparameters)
            learning_rate = best_hyperparameters.get('learning_rate') or 0.001 # Fallback
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            
            print(f"Training final model on {len(loaded_train_data)} days of data for up to {TF_NN_CALIBRATION_SETTINGS['num_epochs']} epochs, with validation on {len(loaded_val_data)} days.")
            
            train_loss_history = []
            val_loss_history = []
            best_val_rmse = float('inf')
            best_model_weights = None
            best_epoch = -1

            start_time = time.monotonic()
            for epoch in range(TF_NN_CALIBRATION_SETTINGS['num_epochs']):
                epoch_train_losses = []
                for day_idx, (eval_date, zero_df, vol_df) in enumerate(loaded_train_data):
                    ql_eval_date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
                    ql.Settings.instance().evaluationDate = ql_eval_date
                    term_structure = create_ql_yield_curve(zero_df, eval_date)
                    helpers = prepare_calibration_helpers(vol_df, term_structure, TF_NN_CALIBRATION_SETTINGS)
                    if not helpers: continue
                    
                    feature_vec = prepare_nn_features(term_structure, ql_eval_date, feature_scaler, external_data)
                    loss = _perform_training_step(final_model, optimizer, tf.constant(feature_vec, dtype=tf.float64), initial_logits, ql_eval_date, term_structure, helpers, TF_NN_CALIBRATION_SETTINGS)
                    
                    current_rmse = np.sqrt(loss) * 10000
                    epoch_train_losses.append(current_rmse)
                    progress = (day_idx + 1) / len(loaded_train_data)
                    bar = ('=' * int(progress * 20)).ljust(20)
                    sys.stdout.write(f"\rEpoch {epoch+1:2d}/{TF_NN_CALIBRATION_SETTINGS['num_epochs']} [{bar}] Training Day {day_idx+1:3d}/{len(loaded_train_data)} - Weighted RMSE: {current_rmse:7.2f} bps")
                    sys.stdout.flush()
                
                avg_epoch_train_rmse = np.mean(epoch_train_losses) if epoch_train_losses else float('nan')
                train_loss_history.append(avg_epoch_train_rmse)

                epoch_val_losses = []
                for eval_date, zero_df, vol_df in loaded_val_data:
                    term_structure = create_ql_yield_curve(zero_df, eval_date)
                    ql_eval_date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
                    feature_vec = prepare_nn_features(term_structure, ql_eval_date, feature_scaler, external_data)
                    predicted_params = final_model((tf.constant(feature_vec, dtype=tf.float64), initial_logits), training=False).numpy()[0]
                    rmse, _ = evaluate_model_on_day(eval_date, zero_df, vol_df, predicted_params, TF_NN_CALIBRATION_SETTINGS)
                    if not np.isnan(rmse): epoch_val_losses.append(rmse)
                
                avg_epoch_val_rmse = np.mean(epoch_val_losses) if epoch_val_losses else float('inf')
                val_loss_history.append(avg_epoch_val_rmse)

                elapsed = time.monotonic() - start_time
                summary_msg = (
                    f"\nEpoch {epoch+1} Summary | "
                    f"Train RMSE (Weighted): {avg_epoch_train_rmse:7.2f} bps | "
                    f"Validation RMSE: {avg_epoch_val_rmse:7.2f} bps | "
                    f"Time: {_format_time(elapsed)}"
                )
                print(summary_msg)

                if avg_epoch_val_rmse < best_val_rmse:
                    best_val_rmse = avg_epoch_val_rmse
                    best_model_weights = final_model.get_weights()
                    best_epoch = epoch + 1
                    print(f"  -> New best model found! Validation RMSE: {best_val_rmse:.2f} bps.")

            print("\n--- Final Training Finished ---")
            
            model_save_dir = os.path.join(FOLDER_MODELS, TF_NN_CALIBRATION_SETTINGS["SAVE_MODEL_DIR_NAME"])
            os.makedirs(model_save_dir, exist_ok=True)

            plt.figure(figsize=(12, 6))
            plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, 'o-', label='Training RMSE (Weighted)')
            plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, 'o-', label='Validation RMSE')
            plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Model (Epoch {best_epoch})')
            plt.title('Training and Validation Loss History')
            plt.xlabel('Epoch')
            plt.ylabel('RMSE (bps)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            history_plot_path = os.path.join(model_save_dir, 'training_history.png')
            plt.savefig(history_plot_path)
            print(f"\n--- Training history plot saved to {history_plot_path} ---")
            plt.show()

            if best_model_weights:
                print(f"\nRestoring model weights from Epoch {best_epoch} with best validation RMSE: {best_val_rmse:.2f} bps.")
                final_model.set_weights(best_model_weights)
            else:
                print("\nWarning: No best model weights were saved. Saving the model from the final epoch.")
            
            print(f"\n--- Saving best model artifact to {model_save_dir} ---")
            final_model.save(os.path.join(model_save_dir, 'model.keras'))
            joblib.dump(feature_scaler, os.path.join(model_save_dir, 'feature_scaler.joblib'))
            np.save(os.path.join(model_save_dir, 'initial_logits.npy'), initial_logits.numpy())
            
            print("--- Model artifact saved successfully ---")
            
            timing_log = {'training_time_seconds': time.monotonic() - start_time}
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            timing_log_filepath = os.path.join(model_save_dir, f"training_time_{timestamp}.json")
            with open(timing_log_filepath, 'w') as f: json.dump(timing_log, f, indent=4)
            print(f"Training time log saved to {timing_log_filepath}")

        if test_files and final_model is not None:
            print("\n--- Evaluating Final Model on Out-of-Sample Test Data ---")
            all_results_dfs = []
            for eval_date, zero_path, vol_path in test_files:
                prediction_start_time = time.monotonic()
                print(f"\n--- Processing test day: {eval_date.strftime('%d-%m-%Y')} ---")
                zero_df = pd.read_csv(zero_path, parse_dates=['Date'])
                vol_df = load_volatility_cube(vol_path)
                term_structure = create_ql_yield_curve(zero_df, eval_date)
                
                ql_eval_date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
                feature_vector = prepare_nn_features(term_structure, ql_eval_date, feature_scaler, external_data)
                predicted_params = final_model((tf.constant(feature_vector, dtype=tf.float64), initial_logits), training=False).numpy()[0]
                
                rmse_bps, results_df = evaluate_model_on_day(eval_date, zero_df, vol_df, predicted_params, TF_NN_CALIBRATION_SETTINGS)
                print(f"Test RMSE for {eval_date}: {rmse_bps:.4f} bps")
                prediction_elapsed = time.monotonic() - prediction_start_time
                print(f"Time taken for evaluation: {_format_time(prediction_elapsed)}")

                if not results_df.empty:
                    day_results_df = results_df.copy()
                    day_results_df['EvaluationDate'] = eval_date
                    day_results_df['DailyRMSE_bps'] = rmse_bps
                    day_results_df['PredictionTimeSeconds'] = prediction_elapsed
                    
                    num_a = TF_NN_CALIBRATION_SETTINGS['num_a_segments'] if TF_NN_CALIBRATION_SETTINGS['optimize_a'] else 0
                    for i in range(num_a):
                        day_results_df[f'a_{i+1}'] = predicted_params[i]
                    for i in range(TF_NN_CALIBRATION_SETTINGS['num_sigma_segments']):
                        day_results_df[f'sigma_{i+1}'] = predicted_params[num_a + i]
                        
                    all_results_dfs.append(day_results_df)

            if all_results_dfs:
                master_results_df = pd.concat(all_results_dfs, ignore_index=True)
                results_save_path = os.path.join(model_save_dir, 'evaluation_results.csv')
                master_results_df.to_csv(results_save_path, index=False)
                print(f"\n--- Comprehensive evaluation results saved to: {results_save_path} ---")

                last_test_date = test_files[-1][0]
                print(f"\n--- Plotting calibration results for last test day: {last_test_date} ---")
                last_day_df = master_results_df[master_results_df['EvaluationDate'] == last_test_date]
                if not last_day_df.empty:
                    plot_calibration_results(last_day_df, last_test_date, model_save_dir)
            
            # --- SHAP ANALYSIS EXECUTION ---
            perform_and_save_shap_analysis(
                model=final_model,
                scaler=feature_scaler,
                initial_logits_tensor=initial_logits,
                test_files_list=test_files,
                external_market_data=external_data,
                settings=TF_NN_CALIBRATION_SETTINGS,
                output_dir=model_save_dir
            )
        else:
            print("\n--- No test files or model available for final evaluation. ---")

    except (FileNotFoundError, ValueError) as e: print(f"\nERROR: {e}")
    except Exception as e: print(f"\nAn unexpected error occurred: {e}"); traceback.print_exc()