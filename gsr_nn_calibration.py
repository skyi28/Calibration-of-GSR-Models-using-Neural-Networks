"""
This script is designed to calibrate Hull-White GSR models using a neural network.

This version is a fully operational implementation for performing a "real run" of
both hyperparameter tuning (using a custom Keras HyperModel and the Hyperband algorithm)
and final model training. All placeholder logic has been removed.

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
from concurrent.futures import ThreadPoolExecutor
import bisect
import traceback
import json
import keras_tuner as kt
import joblib

#--------------------CONFIG--------------------
# Set to False to skip the initial data preparation steps if they have already been run
PREPROCESS_CURVES: bool = False
BOOTSTRAP_CURVES: bool = False

FOLDER_SWAP_CURVES: str = r'data\EUR SWAP CURVE'
FOLDER_ZERO_CURVES: str = r'data\EUR ZERO CURVE'
FOLDER_VOLATILITY_CUBES: str = r'data\EUR BVOL CUBE'
FOLDER_MODELS: str = r'results\neural_network\models'
FOLDER_HYPERPARAMETERS: str = r'results\neural_network\hyperparameters'


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
    min_expiry_years: float = 0.0,
    min_tenor_years: float = 0.0
) -> List[Tuple[ql.SwaptionHelper, str, str]]:
    """
    Parses the swaption volatility cube and creates a list of QuantLib SwaptionHelper objects,
    optionally filtering by minimum expiry and tenor.

    Args:
        vol_cube_df (pd.DataFrame): The DataFrame containing the swaption volatility cube.
        term_structure_handle (ql.RelinkableYieldTermStructureHandle): The yield curve to use for pricing.
        min_expiry_years (float, optional): The minimum expiry time in years. Defaults to 0.0.
        min_tenor_years (float, optional): The minimum tenor time in years. Defaults to 0.0.

    Returns:
        List[Tuple[ql.SwaptionHelper, str, str]]: A list of tuples, where each tuple contains a SwaptionHelper,
            the expiry string and the tenor string.
    """
    helpers_with_info: List[Tuple[ql.SwaptionHelper, str, str]] = []
    vols_df: pd.DataFrame = vol_cube_df[vol_cube_df['Type'] == 'Vol'].set_index('Expiry')
    strikes_df: pd.DataFrame = vol_cube_df[vol_cube_df['Type'] == 'Strike'].set_index('Expiry')
    swap_index: ql.Euribor6M = ql.Euribor6M(term_structure_handle)
    for expiry_str in vols_df.index:
        for tenor_str in vols_df.columns:
            if tenor_str == 'Type': continue
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
    Shows 3 diagrams as 3D plots in the same figure after the calibration.

    Plots the observed market volatilities, model implied volatilities, and the difference between them.

    Args:
        results_df (pd.DataFrame): The DataFrame with the calibration results.
        eval_date (datetime.date): The evaluation date of the yield curve.
    """
    plot_data: pd.DataFrame = results_df.dropna(subset=['MarketVol', 'ModelVol', 'Difference_bps']).copy()
    if plot_data.empty:
        print(f"\nCould not generate plots for {eval_date}: No valid data points available.")
        return
    X = plot_data['Expiry'].values; Y = plot_data['Tenor'].values
    Z_market: ArrayLike = plot_data['MarketVol'].values; Z_model = plot_data['ModelVol'].values
    Z_diff: ArrayLike = plot_data['Difference_bps'].values
    fig = plt.figure(figsize=(24, 8)); fig.suptitle(f'Hull-White Calibration Volatility Surfaces for {eval_date}', fontsize=16)
    ax1 = fig.add_subplot(1, 3, 1, projection='3d'); ax1.set_title('Observed Market Volatilities (bps)')
    surf1 = ax1.plot_trisurf(X, Y, Z_market, cmap=cm.viridis, antialiased=True, linewidth=0.1)
    ax1.set_xlabel('Expiry (Years)'); ax1.set_ylabel('Tenor (Years)'); ax1.set_zlabel('Volatility (bps)')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, pad=0.1); ax1.view_init(elev=30, azim=-120)
    ax2 = fig.add_subplot(1, 3, 2, projection='3d'); ax2.set_title('Model Implied Volatilities (bps)')
    surf2 = ax2.plot_trisurf(X, Y, Z_model, cmap=cm.viridis, antialiased=True, linewidth=0.1)
    ax2.set_xlabel('Expiry (Years)'); ax2.set_ylabel('Tenor (Years)'); ax2.set_zlabel('Volatility (bps)')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, pad=0.1)
    market_min, market_max = np.nanmin(Z_market), np.nanmax(Z_market)
    ax2.set_zlim(market_min * 0.9, market_max * 1.1); ax2.view_init(elev=30, azim=-120)
    ax3 = fig.add_subplot(1, 3, 3, projection='3d'); ax3.set_title('Difference (Market - Model) (bps)')
    surf3 = ax3.plot_trisurf(X, Y, Z_diff, cmap=cm.coolwarm_r, antialiased=True, linewidth=0.1)
    ax3.set_xlabel('Expiry (Years)'); ax3.set_ylabel('Tenor (Years)'); ax3.set_zlabel('Difference (bps)')
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10, pad=0.1)
    max_reasonable_diff: float = np.nanmax(np.abs(Z_diff)); ax3.set_zlim(-max_reasonable_diff, max_reasonable_diff)
    ax3.view_init(elev=30, azim=-120); plt.tight_layout(rect=[0, 0.03, 1, 0.95]); 
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
    feature_tenors: List[float] = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
) -> List[float]:
    """
    Extracts a raw (unscaled) feature vector, including rates and key slopes,
    from a yield curve term structure.
    """
    day_counter: ql.Actual365Fixed = ql.Actual365Fixed()
    
    # Get the base rates as before
    rates: list[float] = [term_structure_handle.zeroRate(eval_date + ql.Period(int(ty * 365.25), ql.Days), day_counter, ql.Continuous).rate() for ty in feature_tenors]
    
    # Helper to get rates for specific tenors
    def get_rate(tenor_in_years: float) -> float:
        if tenor_in_years < 1.0:
            # Use ql.Months for tenors less than a year for precision
            num_months = int(tenor_in_years * 12)
            return term_structure_handle.zeroRate(eval_date + ql.Period(num_months, ql.Months), day_counter, ql.Continuous).rate()
        
        # Use ql.Years for integer years. Using days for non-integer years can be more precise.
        days = int(tenor_in_years * 365.25)
        return term_structure_handle.zeroRate(eval_date + ql.Period(days, ql.Days), day_counter, ql.Continuous).rate()

    # Get the specific rates needed for all slopes
    rate_3m = get_rate(0.25)
    rate_2y = get_rate(2.0)
    rate_5y = get_rate(5.0)
    rate_10y = get_rate(10.0)
    rate_30y = get_rate(30.0)
    
    # Calculate the slopes
    slope_3m10s = rate_10y - rate_3m
    slope_2s10s = rate_10y - rate_2y
    slope_5s30s = rate_30y - rate_5y
    
    # Combine the original rates with the new slope features
    all_features = rates + [slope_3m10s, slope_2s10s, slope_5s30s]
    
    return all_features

def prepare_nn_features(
    term_structure_handle: ql.RelinkableYieldTermStructureHandle,
    eval_date: ql.Date,
    scaler: StandardScaler,
    feature_tenors: List[float] = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
) -> np.ndarray:
    """
    Prepares a feature vector from a yield curve term structure for a neural network calibration.
    This function now extracts raw features (including slopes) and then scales them.

    Args:
        term_structure_handle (ql.RelinkableYieldTermStructureHandle): The yield curve term structure.
        eval_date (ql.Date): The evaluation date of the yield curve.
        scaler (StandardScaler): A pre-fitted StandardScaler object from scikit-learn.
        feature_tenors (List[float], optional): The tenors to use for the feature vector. Defaults to [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0].

    Returns:
        np.ndarray: The prepared and scaled feature vector as a numpy array.
    """
    # Step 1: Extract all raw features (rates and slopes)
    raw_features = extract_raw_features(term_structure_handle, eval_date, feature_tenors)
    
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
    helpers_with_info: List[Tuple[ql.SwaptionHelper, str, str]] = prepare_calibration_helpers(vol_cube_df, term_structure_handle, settings.get("min_expiry_years", 0.0), settings.get("min_tenor_years", 0.0))
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


# ------------------- HYPERPARAMETER TUNING AND TRAINING COMPONENTS -------------------
def _calculate_loss_for_day(params: np.ndarray, ql_eval_date, term_structure_handle, helpers_with_info, settings) -> float:
    """
    The core, computationally-heavy loss calculation using QuantLib.
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
        
        squared_errors = []
        for helper, _, _ in helpers_with_info:
            helper.setPricingEngine(thread_engine)
            model_val = helper.modelValue()
            implied_vol = helper.impliedVolatility(model_val, 1e-4, 500, 0.0001, 1.0)
            squared_errors.append((implied_vol - helper.volatility().value())**2)
        
        return float(np.mean(squared_errors)) if squared_errors else 1e6
    except (RuntimeError, ValueError):
        return 1e6

def _perform_training_step(model, optimizer, feature_vector, initial_logits, ql_eval_date, term_structure_handle, helpers_with_info, settings):
    """
    Executes one forward and backward pass for a single day of data.
    """
    @tf.custom_gradient
    def ql_loss_on_params(params_tensor):
        loss_val = _calculate_loss_for_day(params_tensor.numpy()[0], ql_eval_date, term_structure_handle, helpers_with_info, settings)
        
        def grad_fn(dy):
            base_params = params_tensor.numpy()[0]; h = settings['h_relative']
            def _calc_single_grad(i):
                p_plus = base_params.copy(); p_plus[i] += h
                p_minus = base_params.copy(); p_minus[i] -= h
                loss_plus = _calculate_loss_for_day(p_plus, ql_eval_date, term_structure_handle, helpers_with_info, settings)
                loss_minus = _calculate_loss_for_day(p_minus, ql_eval_date, term_structure_handle, helpers_with_info, settings)
                return (loss_plus - loss_minus) / (2 * h)
            
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

        layers = [hp.Int(f'neurons_{i}', 32, 256, step=32) for i in range(num_layers)]

        # These parameters are fixed during the search but needed for model instantiation
        total_params = TF_NN_CALIBRATION_SETTINGS['num_a_segments'] + TF_NN_CALIBRATION_SETTINGS['num_sigma_segments']
        upper_bound = TF_NN_CALIBRATION_SETTINGS['upper_bound']

        model = ResidualParameterModel(
            total_params_to_predict=total_params, upper_bound=upper_bound,
            layers=layers, activation=activation, use_dropout=use_dropout,
            dropout_rate=dropout_rate, dtype=tf.float64
        )
        return model

    def fit(self, hp, model, loaded_train_data, loaded_val_data, settings, feature_scaler, initial_logits, **kwargs):
        """Overrides the default fit method to implement the custom training loop."""
        learning_rate = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        epochs = kwargs.get('epochs', 1)
        trial = None
        for callback in kwargs.get('callbacks', []):
            if hasattr(callback, 'trial'):
                trial = callback.trial
                break
        trial_id = trial.trial_id if trial else "unknown"
        
        for epoch in range(epochs):
            print(f"\nTrial {trial_id} | Starting Epoch {epoch + 1}/{epochs}")
            epoch_losses = []
            total_days: int = len(loaded_train_data)
            for day_idx, (eval_date, zero_df, vol_df) in enumerate(loaded_train_data):
                print(f"\r  Training day {day_idx + 1}/{total_days}...", end='', flush=True)
                ql_eval_date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
                ql.Settings.instance().evaluationDate = ql_eval_date
                term_structure = create_ql_yield_curve(zero_df, eval_date)
                helpers = prepare_calibration_helpers(vol_df, term_structure, settings['min_expiry_years'], settings['min_tenor_years'])
                if not helpers: continue
                
                feature_vec = prepare_nn_features(term_structure, ql_eval_date, feature_scaler)
                loss = _perform_training_step(model, optimizer, tf.constant(feature_vec, dtype=tf.float64), initial_logits, ql_eval_date, term_structure, helpers, settings)
                if not np.isnan(loss):
                    epoch_losses.append(np.sqrt(loss) * 10000)
            
            avg_epoch_rmse = np.mean(epoch_losses) if epoch_losses else float('inf')
            print(f"\n  Trial {trial_id} | Epoch {epoch+1}/{epochs} | Avg Train RMSE: {avg_epoch_rmse:.2f} bps")

            val_losses = []
            for eval_date, zero_df, vol_df in loaded_val_data:
                term_structure = create_ql_yield_curve(zero_df, eval_date)
                feature_vec = prepare_nn_features(term_structure, ql.Date(eval_date.day, eval_date.month, eval_date.year), feature_scaler)
                predicted_params = model((tf.constant(feature_vec, dtype=tf.float64), initial_logits), training=False).numpy()[0]
                rmse, _ = evaluate_model_on_day(eval_date, zero_df, vol_df, predicted_params, settings)
                if not np.isnan(rmse): val_losses.append(rmse)
            
            avg_val_rmse = np.mean(val_losses) if val_losses else float('inf')
            print(f"  Trial {trial_id} | Epoch {epoch+1}/{epochs} | Avg Validation RMSE: {avg_val_rmse:.2f} bps")
            
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
            "num_a_segments": 1, "num_sigma_segments": 3, "optimize_a": True,
            "upper_bound": 0.1, "pricing_engine_integration_points": 32,
            "num_epochs": 1, "h_relative": 1e-7,
            "initial_guess": [0.02, 0.0002, 0.0002, 0.00017],
            "gradient_clip_norm": 2.0, "num_threads": os.cpu_count() or 4,
            "min_expiry_years": 2.0, "min_tenor_years": 2.0,
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

            print("\n--- Preparing Feature Scaler using Training Data ---")
            all_training_features = []
            for eval_date, zero_path, _ in train_files:
                zero_curve_df = pd.read_csv(zero_path, parse_dates=['Date'])
                term_structure = create_ql_yield_curve(zero_curve_df, eval_date)
                ql_eval_date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
                
                # Extract the new, wider raw feature set (rates + slopes)
                raw_features = extract_raw_features(term_structure, ql_eval_date)
                all_training_features.append(raw_features)
            
            # Fit the scaler on the new, richer feature set
            feature_scaler = StandardScaler()
            feature_scaler.fit(np.array(all_training_features))
            print("Feature scaler has been fitted to the training data (including slopes).")

            print("\n--- Pre-loading all data into memory ---")
            loaded_train_data = [(d, pd.read_csv(zp, parse_dates=['Date']), load_volatility_cube(vp)) for d, zp, vp in train_files]
            loaded_val_data = [(d, pd.read_csv(zp, parse_dates=['Date']), load_volatility_cube(vp)) for d, zp, vp in val_files]
            print("All training and validation data has been loaded.")

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
                    initial_logits=initial_logits
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
            final_model = HullWhiteHyperModel().build(best_hyperparameters)
            learning_rate = best_hyperparameters.get('learning_rate') or 0.001 # Fallback
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            combined_train_data = loaded_train_data + loaded_val_data
            print(f"Training final model on {len(combined_train_data)} days of data for {TF_NN_CALIBRATION_SETTINGS['num_epochs']} epochs.")
            
            start_time = time.monotonic()
            for epoch in range(TF_NN_CALIBRATION_SETTINGS['num_epochs']):
                epoch_losses = []
                for day_idx, (eval_date, zero_df, vol_df) in enumerate(combined_train_data):
                    ql_eval_date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
                    ql.Settings.instance().evaluationDate = ql_eval_date
                    term_structure = create_ql_yield_curve(zero_df, eval_date)
                    helpers = prepare_calibration_helpers(vol_df, term_structure, TF_NN_CALIBRATION_SETTINGS['min_expiry_years'], TF_NN_CALIBRATION_SETTINGS['min_tenor_years'])
                    if not helpers: continue
                    
                    feature_vec = prepare_nn_features(term_structure, ql_eval_date, feature_scaler)
                    loss = _perform_training_step(final_model, optimizer, tf.constant(feature_vec, dtype=tf.float64), initial_logits, ql_eval_date, term_structure, helpers, TF_NN_CALIBRATION_SETTINGS)
                    
                    current_rmse = np.sqrt(loss) * 10000
                    epoch_losses.append(current_rmse)
                    progress = (day_idx + 1) / len(combined_train_data)
                    bar = ('=' * int(progress * 20)).ljust(20)
                    sys.stdout.write(f"\rEpoch {epoch+1:2d}/{TF_NN_CALIBRATION_SETTINGS['num_epochs']} [{bar}] Day {day_idx+1:3d}/{len(combined_train_data)} - RMSE: {current_rmse:7.2f} bps")
                    sys.stdout.flush()
                
                avg_epoch_rmse = np.mean(epoch_losses) if epoch_losses else float('nan')
                elapsed = time.monotonic() - start_time
                print(f"\nEpoch {epoch+1} Summary | Avg. Train RMSE: {avg_epoch_rmse:7.2f} bps | Time Elapsed: {_format_time(elapsed)}")
            
            print("\n--- Final Training Finished ---")
            
            model_save_dir = os.path.join(FOLDER_MODELS, TF_NN_CALIBRATION_SETTINGS["SAVE_MODEL_DIR_NAME"])
            os.makedirs(model_save_dir, exist_ok=True)
            print(f"\n--- Saving model artifact to {model_save_dir} ---")
            
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
            all_results_dfs = [] # NEW: Initialize list to hold daily results
            for eval_date, zero_path, vol_path in test_files:
                prediction_start_time = time.monotonic()
                print(f"\n--- Processing test day: {eval_date.strftime('%d-%m-%Y')} ---")
                zero_df = pd.read_csv(zero_path, parse_dates=['Date'])
                vol_df = load_volatility_cube(vol_path)
                term_structure = create_ql_yield_curve(zero_df, eval_date)
                
                feature_vector = prepare_nn_features(term_structure, ql.Date(eval_date.day, eval_date.month, eval_date.year), feature_scaler)
                predicted_params = final_model((tf.constant(feature_vector, dtype=tf.float64), initial_logits), training=False).numpy()[0]
                
                rmse_bps, results_df = evaluate_model_on_day(eval_date, zero_df, vol_df, predicted_params, TF_NN_CALIBRATION_SETTINGS)
                print(f"Test RMSE for {eval_date}: {rmse_bps:.4f} bps")
                prediction_elapsed = time.monotonic() - prediction_start_time
                print(f"Time taken for evaluation: {_format_time(prediction_elapsed)}")

                if not results_df.empty:
                    # NEW: Enrich the daily DataFrame with all necessary info
                    day_results_df = results_df.copy()
                    day_results_df['EvaluationDate'] = eval_date
                    day_results_df['DailyRMSE_bps'] = rmse_bps
                    day_results_df['PredictionTimeSeconds'] = prediction_elapsed
                    
                    # Add parameter columns dynamically
                    num_a = TF_NN_CALIBRATION_SETTINGS['num_a_segments'] if TF_NN_CALIBRATION_SETTINGS['optimize_a'] else 0
                    for i in range(num_a):
                        day_results_df[f'a_{i+1}'] = predicted_params[i]
                    for i in range(TF_NN_CALIBRATION_SETTINGS['num_sigma_segments']):
                        day_results_df[f'sigma_{i+1}'] = predicted_params[num_a + i]
                        
                    all_results_dfs.append(day_results_df)

            if all_results_dfs:
                # NEW: Concatenate all daily results and save to a single CSV
                master_results_df = pd.concat(all_results_dfs, ignore_index=True)
                results_save_path = os.path.join(model_save_dir, 'evaluation_results.csv')
                master_results_df.to_csv(results_save_path, index=False)
                print(f"\n--- Comprehensive evaluation results saved to: {results_save_path} ---")

                # Plot results for the last test day as a final check
                last_test_date = test_files[-1][0]
                print(f"\n--- Plotting calibration results for last test day: {last_test_date} ---")
                last_day_df = master_results_df[master_results_df['EvaluationDate'] == last_test_date]
                if not last_day_df.empty:
                    plot_calibration_results(last_day_df, last_test_date, model_save_dir)
        else:
            print("\n--- No test files or model available for final evaluation. ---")

    except (FileNotFoundError, ValueError) as e: print(f"\nERROR: {e}")
    except Exception as e: print(f"\nAn unexpected error occurred: {e}"); traceback.print_exc()