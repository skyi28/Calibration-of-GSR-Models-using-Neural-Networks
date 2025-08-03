"""
This script is designed to calibrate Hull-White GSR models using a neural network.

It now includes functionality to save the trained model and to load a pre-trained
model to continue the training process.
"""
import datetime
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

#--------------------CONFIG--------------------
PREPROCESS_CURVES: bool = False
BOOTSTRAP_CURVES: bool = False

FOLDER_SWAP_CURVES: str = r'data\EUR SWAP CURVE'
FOLDER_ZERO_CURVES: str = r'data\EUR ZERO CURVE'
FOLDER_VOLATILITY_CUBES: str = r'data\EUR BVOL CUBE'
FOLDER_RESULTS_NN: str = r'results\neural_network\predictions'
FOLDER_RESULTS_PARAMS_NN: str = r'results\neural_network\parameters'
FOLDER_MODELS: str = r'results\neural_network\models'


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
    This version correctly uses 'Term' and 'Unit' columns for tenor creation.
    """
    # 1. Set Evaluation Date
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


#--------------------DATA DISCOVERY AND SPLITTING--------------------
def load_and_split_data_files(
    zero_curve_folder: str,
    vol_cube_folder: str,
    train_split_percentage: float
) -> Tuple[List[Tuple[datetime.date, str, str]], List[Tuple[datetime.date, str, str]]]:
    """
    Discovers and loads available zero curve and volatility cube data files and splits them into
    training and validation sets according to the given percentage.

    Args:
        zero_curve_folder (str): The folder containing the zero curve data files.
        vol_cube_folder (str): The folder containing the volatility cube data files.
        train_split_percentage (float): The percentage of available data to use for training.

    Returns:
        Tuple[List[Tuple[datetime.date, str, str]], List[Tuple[datetime.date, str, str]]]:
            A tuple of two lists. The first list contains the training data, the second list contains the
            validation data. Each list element is a tuple of (evaluation_date, zero_curve_path, vol_cube_path).
    """
    print("\n--- Discovering and loading data files ---")
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
            else:
                print(f"Warning: Zero curve found for {date_str}, but no corresponding volatility cube. Skipping.")
    if not available_files:
        raise ValueError("No matching pairs of zero curve and volatility data found.")
    available_files.sort(key=lambda x: x[0])
    print(f"Found {len(available_files)} complete data sets from {available_files[0][0]} to {available_files[-1][0]}.")
    split_index: int = int(len(available_files) * (train_split_percentage / 100.0))
    train_files: List[Tuple[datetime.date, str, str]] = available_files[:split_index]
    test_files: List[Tuple[datetime.date, str, str]] = available_files[split_index:]
    if not train_files:
        raise ValueError("Training split resulted in 0 files. Adjust train_split_percentage or add more data.")
    print(f"Splitting data: {len(train_files)} days for training, {len(test_files)} days for validation.")
    return train_files, test_files

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
    total_count, filtered_count = 0, 0
    for expiry_str in vols_df.index:
        for tenor_str in vols_df.columns:
            if tenor_str == 'Type': continue
            vol, strike = vols_df.loc[expiry_str, tenor_str], strikes_df.loc[expiry_str, tenor_str]
            if pd.isna(vol) or pd.isna(strike): continue
            total_count += 1
            if parse_tenor_to_years(expiry_str) < min_expiry_years or parse_tenor_to_years(tenor_str) < min_tenor_years:
                filtered_count += 1
                continue
            vol_handle: ql.QuoteHandle = ql.QuoteHandle(ql.SimpleQuote(float(vol) / 10000.0))
            helper: ql.SwaptionHelper = ql.SwaptionHelper(parse_tenor(expiry_str), parse_tenor(tenor_str), vol_handle, swap_index, ql.Period(6, ql.Months), swap_index.dayCounter(), swap_index.dayCounter(), term_structure_handle, ql.Normal)
            helpers_with_info.append((helper, expiry_str, tenor_str))
    return helpers_with_info

def plot_calibration_results(results_df: pd.DataFrame, eval_date: datetime.date):
    """
    Shows 3 diagrams as 3D plots in the same figure after the calibration.

    Plots the observed market volatilities, model implied volatilities, and the difference between them.

    Args:
        results_df (pd.DataFrame): The DataFrame with the calibration results.
        eval_date (datetime.date): The evaluation date of the yield curve.
    """
    
    plot_data: pd.DataFrame = results_df.dropna(subset=['MarketVol', 'ModelVol', 'Difference']).copy()
    if plot_data.empty:
        print(f"\nCould not generate plots for {eval_date}: No valid data points available.")
        return
    X = plot_data['Expiry'].values; Y = plot_data['Tenor'].values
    Z_market: ArrayLike = plot_data['MarketVol'].values; Z_model = plot_data['ModelVol'].values
    Z_diff: ArrayLike = plot_data['Difference'].values
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
    ax3.view_init(elev=30, azim=-120); plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()


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

def prepare_nn_features(
    term_structure_handle: ql.RelinkableYieldTermStructureHandle,
    eval_date: ql.Date,
    scaler: StandardScaler,
    feature_tenors: List[float] = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
) -> np.ndarray:
    """
    Prepares a feature vector from a yield curve term structure for a neural network calibration.

    Args:
        term_structure_handle (ql.RelinkableYieldTermStructureHandle): The yield curve term structure.
        eval_date (ql.Date): The evaluation date of the yield curve.
        scaler (StandardScaler): A StandardScaler object from scikit-learn.
        feature_tenors (List[float], optional): The tenors to use for the feature vector. Defaults to [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0].

    Returns:
        np.ndarray: The prepared feature vector as a numpy array.
    """
    day_counter: ql.Actual365Fixed = ql.Actual365Fixed()
    rates: list[float] = [term_structure_handle.zeroRate(eval_date + ql.Period(int(ty * 365.25), ql.Days), day_counter, ql.Continuous).rate() for ty in feature_tenors]
    return scaler.transform(np.array(rates).reshape(1, -1))

class ResidualParameterModel(tf.keras.Model):
    def __init__(self, total_params_to_predict, upper_bound, hidden_layer_size=64, batch_momentum=0.99, **kwargs):
        """
        Initializes the ResidualParameterModel with specified parameters.

        Args:
            total_params_to_predict (int): The number of parameters the model is designed to predict.
            upper_bound (float): The upper bound for the predicted values.
            hidden_layer_size (int, optional): The size of the hidden layers. Defaults to 64.
            batch_momentum (float, optional): The momentum for batch normalization. Defaults to 0.99.
            **kwargs: Additional keyword arguments passed to the parent tf.keras.Model constructor.
        """
        # Pass standard Keras arguments (like name, trainable, dtype) to the parent
        super().__init__(**kwargs)
        
        # Store config parameters
        self.total_params_to_predict = total_params_to_predict
        self.upper_bound_value = upper_bound
        self.hidden_layer_size = hidden_layer_size
        self.batch_momentum = batch_momentum
        
        # Define layers
        self.upper_bound = tf.constant(self.upper_bound_value, dtype=tf.float64)
        self.hidden1 = tf.keras.layers.Dense(self.hidden_layer_size, dtype=tf.float64, activation='relu')
        self.hidden2 = tf.keras.layers.Dense(self.hidden_layer_size, dtype=tf.float64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(self.total_params_to_predict, dtype=tf.float64, kernel_initializer='zeros', bias_initializer='zeros')

    def call(self, inputs) -> tf.Tensor:
        """
        Forward pass through the ResidualParameterModel.
        """
        feature_vector, initial_logits = inputs
        x = self.hidden1(feature_vector)
        x = self.hidden2(x)
        delta_logits = self.output_layer(x)
        final_logits = initial_logits + delta_logits
        return self.upper_bound * tf.keras.activations.sigmoid(final_logits)

    def get_config(self):
        """Returns the configuration of the model."""
        # Get the config from the parent class (which includes name, dtype, etc.)
        config = super().get_config()
        # Update it with our custom constructor arguments
        config.update({
            'total_params_to_predict': self.total_params_to_predict,
            'upper_bound': self.upper_bound_value,
            'hidden_layer_size': self.hidden_layer_size,
            'batch_momentum': self.batch_momentum,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Creates a model from its config."""
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
        results_data.append({'ExpiryStr': expiry_str, 'TenorStr': tenor_str, 'MarketVol': market_vol_bps, 'ModelVol': model_vol_bps, 'Difference': error_bps})
    results_df: pd.DataFrame = pd.DataFrame(results_data)
    if not results_df.empty:
        results_df['Expiry'] = results_df['ExpiryStr'].apply(parse_tenor_to_years); results_df['Tenor'] = results_df['TenorStr'].apply(parse_tenor_to_years)
    final_rmse_bps: float = np.sqrt(np.mean(squared_errors)) * 10000 if squared_errors else float('nan')
    return final_rmse_bps, results_df


#--------------------MAIN EXECUTION LOGIC--------------------
if __name__ == '__main__':
    try:
        os.makedirs(FOLDER_RESULTS_NN, exist_ok=True)
        os.makedirs(FOLDER_RESULTS_PARAMS_NN, exist_ok=True)
        os.makedirs(FOLDER_MODELS, exist_ok=True)

        TRAIN_SPLIT_PERCENTAGE: float = 80.0
        
        TF_NN_CALIBRATION_SETTINGS = {
            # Number of piecewise constant segments for the mean-reversion parameter 'a'. 1 means 'a' is constant.
            "num_a_segments": 1,
            # Number of piecewise constant segments for the volatility parameter 'sigma'. More segments allow for a more flexible term structure of volatility.
            "num_sigma_segments": 3,
            # If True, the neural network will predict the mean-reversion 'a'. If False, 'a' is fixed to its value in 'initial_guess'.
            "optimize_a": True,
            # The upper limit for the predicted parameters ('a' and 'sigma'). The NN output is scaled by this value, effectively constraining the search space.
            "upper_bound": 0.1,
            # Number of integration points for the Gaussian1dSwaptionEngine. Higher values increase pricing accuracy but slow down calibration.
            "pricing_engine_integration_points": 32,
            # Configuration for the learning rate decay schedule. Helps in finding a good solution by reducing the learning rate during training.
            "learning_rate_config": {
                "initial_learning_rate": 1e-2,  # The starting learning rate for the Adam optimizer.
                "decay_steps": 100,             # The number of training steps after which the learning rate is reduced.
                "decay_rate": 0.95              # The factor by which the learning rate is multiplied.
            },
            # The total number of times the model will iterate over the entire training dataset.
            "num_epochs": 1,
            # The relative step size used for the finite difference method to approximate gradients of the QuantLib loss function.
            "h_relative": 1e-7,
            # The number of neurons in each of the two hidden layers of the neural network.
            "hidden_layer_size": 32,
            # Initial guess for the parameters to be calibrated ('a' and 'sigma' values). The NN learns to predict a 'residual' or adjustment to these.
            "initial_guess": [0.020566979477481182, 0.0002299520131905028, 0.00021618386141957264, 0.00017446618702943582],
            # The maximum norm for gradients. Used for gradient clipping to prevent the exploding gradients problem during training.
            "gradient_clip_norm": 2.0,
            # The number of parallel threads to use for calculating the finite-difference gradients, speeding up training steps.
            "num_threads": 8,
            # The momentum parameter for the moving average in Batch Normalization layers. Note: BN layers are currently commented out.
            "batch_momentum": 0.99,
            # Minimum swaption expiry (in years) to be included in the calibration. Swaptions with shorter expiries will be filtered out.
            "min_expiry_years": 1.0,
            # Minimum swaption tenor (in years) to be included in the calibration. Swaptions with shorter tenors will be filtered out.
            "min_tenor_years": 1.0,
            # Set to a model name (e.g., "model_20250802_215000" - no file ending) to load a pre-trained model. If None, a new model is created.
            "LOAD_MODEL_NAME": None,
            # The name used to save the model after training. A timestamp is added to ensure uniqueness.
            "SAVE_MODEL_NAME": f"model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        # 1. Load and split all available data
        train_files, test_files = load_and_split_data_files(FOLDER_ZERO_CURVES, FOLDER_VOLATILITY_CUBES, TRAIN_SPLIT_PERCENTAGE)
        if not test_files: print("Warning: No files left for testing/validation after split.")

        # 2. Prepare the feature scaler using only training data
        print("\n--- Preparing Feature Scaler using Training Data ---")
        all_training_features: list[list[float]] = []
        feature_tenors_for_scaling = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
        for eval_date, zero_path, _ in train_files:
            zero_curve_df: pd.DataFrame = pd.read_csv(zero_path)
            ql_eval_date: ql.Date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
            term_structure_handle: ql.RelinkableYieldTermStructureHandle = create_ql_yield_curve(zero_curve_df, eval_date)
            day_counter: ql.Actual365Fixed = ql.Actual365Fixed()
            rates: list[float] = [term_structure_handle.zeroRate(ql_eval_date + ql.Period(int(ty * 365.25), ql.Days), day_counter, ql.Continuous).rate() for ty in feature_tenors_for_scaling]
            all_training_features.append(rates)
        
        feature_scaler: StandardScaler = StandardScaler()
        if not all_training_features:
            raise ValueError("Could not extract any features from the training data to fit the scaler.")
        feature_scaler.fit(np.array(all_training_features))
        print("Feature scaler has been fitted to the training data.")

        # 3. Pre-load all training data into memory
        print("\n--- Pre-loading all training data into memory ---")
        loaded_train_data: list[tuple[datetime.date, pd.DataFrame, pd.DataFrame]] = []
        for i, (eval_date, zero_path, vol_path) in enumerate(train_files):
            sys.stdout.write(f"\rLoading file {i+1}/{len(train_files)}: {os.path.basename(zero_path)}")
            sys.stdout.flush()
            zero_curve_df: pd.DataFrame = pd.read_csv(zero_path)
            vol_cube_df: pd.DataFrame = load_volatility_cube(vol_path)
            loaded_train_data.append((eval_date, zero_curve_df, vol_cube_df))
        print("\nAll training data has been loaded.")

        # 4. Initialize or load the shared model and optimizer
        num_a_params_to_predict: int = TF_NN_CALIBRATION_SETTINGS['num_a_segments'] if TF_NN_CALIBRATION_SETTINGS['optimize_a'] else 0
        total_params_to_predict: int = num_a_params_to_predict + TF_NN_CALIBRATION_SETTINGS['num_sigma_segments']
        assert len(TF_NN_CALIBRATION_SETTINGS['initial_guess']) == total_params_to_predict, \
            f"Initial guess length must match total predictable parameters ({total_params_to_predict})"

        nn_model: ResidualParameterModel
        load_model_name: str | None = TF_NN_CALIBRATION_SETTINGS.get("LOAD_MODEL_NAME")

        if load_model_name:
            model_path: str = os.path.join(FOLDER_MODELS, load_model_name + '.keras')
            if os.path.exists(model_path):
                print(f"\n--- Loading existing model: {load_model_name} ---")
                nn_model = tf.keras.models.load_model(model_path, custom_objects={'ResidualParameterModel': ResidualParameterModel})
                print("Model loaded successfully.")
                TF_NN_CALIBRATION_SETTINGS["SAVE_MODEL_NAME"] = f"{load_model_name}_cont_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            else:
                print(f"Warning: Model '{load_model_name}' not found at '{model_path}'. A new model will be initialized.")
                nn_model = ResidualParameterModel(total_params_to_predict, TF_NN_CALIBRATION_SETTINGS['upper_bound'],
                                                  TF_NN_CALIBRATION_SETTINGS['hidden_layer_size'], TF_NN_CALIBRATION_SETTINGS['batch_momentum'], dtype=tf.float64)
        else:
            print("\n--- Initializing a new model ---")
            nn_model = ResidualParameterModel(total_params_to_predict, TF_NN_CALIBRATION_SETTINGS['upper_bound'],
                                              TF_NN_CALIBRATION_SETTINGS['hidden_layer_size'], TF_NN_CALIBRATION_SETTINGS['batch_momentum'], dtype=tf.float64)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(**TF_NN_CALIBRATION_SETTINGS['learning_rate_config'])
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        p_scaled = np.clip(np.array(TF_NN_CALIBRATION_SETTINGS['initial_guess'], dtype=np.float64) / TF_NN_CALIBRATION_SETTINGS['upper_bound'], 1e-9, 1 - 1e-9)
        initial_logits = tf.constant([np.log(p_scaled / (1 - p_scaled))], dtype=tf.float64)

        # 5. --- TRAINING LOOP ---
        print("\n--- Starting Model Training ---")
        start_time: float = time.monotonic()
        for epoch in range(TF_NN_CALIBRATION_SETTINGS['num_epochs']):
            epoch_losses: list[float] = []
            for day_idx, (eval_date, zero_curve_df, vol_cube_df) in enumerate(loaded_train_data):
                ql_eval_date: ql.Date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
                ql.Settings.instance().evaluationDate = ql_eval_date
                term_structure_handle: ql.RelinkableYieldTermStructureHandle = create_ql_yield_curve(zero_curve_df, eval_date)
                helpers_with_info: list[Tuple[ql.SwaptionHelper, str, str]] = prepare_calibration_helpers(vol_cube_df, term_structure_handle,
                                                              TF_NN_CALIBRATION_SETTINGS['min_expiry_years'], TF_NN_CALIBRATION_SETTINGS['min_tenor_years'])
                if not helpers_with_info:
                    print(f"\nSkipping {eval_date} due to lack of valid swaption helpers.")
                    continue
                
                feature_vector = prepare_nn_features(term_structure_handle, ql_eval_date, feature_scaler)
                feature_vector = tf.constant(feature_vector, dtype=tf.float64)

                def _calculate_loss_for_day(params: np.ndarray) -> float:
                    """
                    Calculates the loss for a given day using swaption pricing and implied volatility.

                    Args:
                        params (np.ndarray): Array containing the model parameters to be calibrated, 
                                            split into mean reversion ('a') and volatility ('sigma') segments.

                    Returns:
                        float: Mean squared error between model-implied volatilities and market volatilities.
                            Returns a large constant (1e6) in case of exceptions during calculation.

                    This function sets up the GSR model using the provided parameters, expands these
                    parameters to a unified timeline, and uses a Gaussian1dSwaptionEngine to price swaptions.
                    The loss is computed as the mean squared difference between model-implied volatilities 
                    and observed market volatilities for a set of calibration helpers.
                    """
                    try:
                        num_a = num_a_params_to_predict; a_params, sigma_params = params[:num_a], params[num_a:]
                        reversion_quotes: list[ql.SimpleQuote] = [ql.SimpleQuote(p) for p in a_params]
                        sigma_quotes: list[ql.SimpleQuote] = [ql.SimpleQuote(p) for p in sigma_params]
                        if not TF_NN_CALIBRATION_SETTINGS['optimize_a']: reversion_quotes = [ql.SimpleQuote(p) for p in TF_NN_CALIBRATION_SETTINGS['initial_guess'][:num_a]] if num_a > 0 else [ql.SimpleQuote(0.01)]
                        included_expiries: list[float] = sorted(list(set([parse_tenor_to_years(e) for _, e, _ in helpers_with_info])))
                        a_steps: list[ql.Date] = _get_step_dates_from_expiries(ql_eval_date, included_expiries, TF_NN_CALIBRATION_SETTINGS['num_a_segments'])
                        s_steps: list[ql.Date] = _get_step_dates_from_expiries(ql_eval_date, included_expiries, TF_NN_CALIBRATION_SETTINGS['num_sigma_segments'])
                        unified_steps: list[ql.Date] = sorted(list(set(a_steps + s_steps)))
                        exp_rev_h: list[ql.QuoteHandle] = _expand_params_to_unified_timeline(reversion_quotes, a_steps, unified_steps)
                        exp_sig_h: list[ql.QuoteHandle] = _expand_params_to_unified_timeline(sigma_quotes, s_steps, unified_steps)
                        thread_model: ql.Gsr = ql.Gsr(term_structure_handle, unified_steps, exp_sig_h, exp_rev_h, 61.0)
                        thread_engine: ql.Gaussian1dSwaptionEngine = ql.Gaussian1dSwaptionEngine(thread_model, TF_NN_CALIBRATION_SETTINGS['pricing_engine_integration_points'], 7.0, True, False, term_structure_handle)
                        squared_errors: list[float] = []
                        for original_helper, _, _ in helpers_with_info:
                            original_helper.setPricingEngine(thread_engine); model_val = original_helper.modelValue()
                            implied_vol: float = original_helper.impliedVolatility(model_val, 1e-4, 500, 0.0001, 1.0)
                            squared_errors.append((implied_vol - original_helper.volatility().value())**2)
                        return float(np.mean(squared_errors)) if squared_errors else 1e6
                    except (RuntimeError, ValueError): return 1e6

                @tf.custom_gradient
                def ql_loss_on_params(params_tensor):
                    """
                    Computes the loss for a set of model parameters given a set of GSR helpers.

                    The loss is computed by pricing each of the calibration helpers using a Gaussian1dSwaptionEngine
                    and calculating the mean squared difference between the model-implied volatilities and the
                    observed market volatilities.

                    The gradient is computed using a finite difference approach.

                    Args:
                        params_tensor (tf.Tensor): A 1D tensor containing the model parameters to be calibrated.

                    Returns:
                        A tuple containing the loss value and a gradient function.
                    """
                    loss_val: float = _calculate_loss_for_day(params_tensor.numpy()[0])
                    def grad_fn(dy):
                        """
                        Computes the gradient of the loss function using a finite difference approach.

                        Args:
                            dy (tf.Tensor): The upstream derivative, i.e. the loss value times the gradient of the loss function with respect to the model parameters.

                        Returns:
                            A tf.Tensor containing the gradient of the loss function with respect to the model parameters.
                        """
                        base_params = params_tensor.numpy()[0]; h: float = TF_NN_CALIBRATION_SETTINGS['h_relative']
                        def _calc_single_grad(i):
                            """
                            Computes the gradient of the loss function with respect to a single model parameter.

                            Args:
                                i (int): Index of the model parameter to compute the gradient for.

                            Returns:
                                float: The gradient of the loss function with respect to the model parameter.

                            This function computes the gradient of the loss function using a finite difference approach.
                            The `base_params` are modified by adding and subtracting a small value (`h`) to the i-th parameter.
                            The gradient is then computed as the difference between the loss values evaluated at the modified
                            parameters, divided by twice the step size (`h`).
                            """
                            p_plus = base_params.copy(); p_plus[i] += h
                            p_minus = base_params.copy(); p_minus[i] -= h
                            return (_calculate_loss_for_day(p_plus) - _calculate_loss_for_day(p_minus)) / (2 * h)
                        with ThreadPoolExecutor(max_workers=TF_NN_CALIBRATION_SETTINGS.get("num_threads", os.cpu_count() or 1)) as executor:
                            grad = np.array(list(executor.map(_calc_single_grad, range(len(base_params)))))
                        return tf.constant([dy.numpy() * grad], dtype=tf.float64)
                    return tf.constant(loss_val, dtype=tf.float64), grad_fn

                with tf.GradientTape() as tape:
                    predicted_params = nn_model((feature_vector, initial_logits), training=True)
                    loss = ql_loss_on_params(predicted_params)
                
                grads = tape.gradient(loss, nn_model.trainable_variables)
                if grads:
                    clipped_grads = [(tf.clip_by_norm(g, TF_NN_CALIBRATION_SETTINGS['gradient_clip_norm']) if g is not None else None) for g in grads]
                    optimizer.apply_gradients(zip(clipped_grads, nn_model.trainable_variables))
                
                current_rmse: float = tf.sqrt(loss).numpy() * 10000; epoch_losses.append(current_rmse)
                progress: float = (day_idx + 1) / len(loaded_train_data); bar: str = ('=' * int(progress * 20)).ljust(20)
                sys.stdout.write(f"\rEpoch {epoch+1:2d}/{TF_NN_CALIBRATION_SETTINGS['num_epochs']} [{bar}] Day {day_idx+1:2d}/{len(loaded_train_data)} - RMSE: {current_rmse:7.2f} bps")
                sys.stdout.flush()

            avg_epoch_rmse: float = float(np.mean(epoch_losses)) if epoch_losses else float('nan'); elapsed: float = time.monotonic() - start_time
            print(f"\nEpoch {epoch+1} Summary | Avg. Train RMSE: {avg_epoch_rmse:7.2f} bps | Time Elapsed: {_format_time(elapsed)}")
        
        print("\n--- Training Finished ---")
        
        # --- Save the final trained model ---
        save_model_name: str | None = TF_NN_CALIBRATION_SETTINGS.get("SAVE_MODEL_NAME")
        if save_model_name:
            model_save_path = os.path.join(FOLDER_MODELS, save_model_name + '.keras')
            print(f"\n--- Saving model to {model_save_path} ---")
            nn_model.save(model_save_path)
            print("--- Model saved successfully ---")
        else:
            print("\n--- Model not saved (no SAVE_MODEL_NAME specified) ---")


        # 6. --- VALIDATION/TESTING AND SAVING LOOP ---
        if test_files:
            print("\n--- Evaluating Model on Test Data and Saving Results ---")
            output_folder_name = save_model_name if TF_NN_CALIBRATION_SETTINGS.get('num_epochs', 0) > 0 else load_model_name
            if not output_folder_name:
                print("Warning: Could not determine output folder name for results. Saving to base directory.")
                output_folder_name = "unknown_model"

            # Create the specific subdirectories for the results of this model
            params_output_dir = os.path.join(FOLDER_RESULTS_PARAMS_NN, output_folder_name)
            predictions_output_dir = os.path.join(FOLDER_RESULTS_NN, output_folder_name)
            os.makedirs(params_output_dir, exist_ok=True)
            os.makedirs(predictions_output_dir, exist_ok=True)
            
            test_results_for_plot: dict = {}
            for eval_date, zero_path, vol_path in test_files:
                print(f"\n--- Processing test day: {eval_date.strftime('%d-%m-%Y')} ---")
                zero_curve_df: pd.DataFrame = pd.read_csv(zero_path)
                vol_cube_df: pd.DataFrame = load_volatility_cube(vol_path)
                ql_eval_date: ql.Date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
                ql.Settings.instance().evaluationDate = ql_eval_date
                term_structure_handle: ql.RelinkableYieldTermStructureHandle = create_ql_yield_curve(zero_curve_df, eval_date)
                
                feature_vector: NDArray = prepare_nn_features(term_structure_handle, ql_eval_date, feature_scaler)
                predicted_params: NDArray = nn_model((tf.constant(feature_vector, dtype=tf.float64), initial_logits), training=False).numpy()[0]
                
                num_a: int = num_a_params_to_predict
                calibrated_as: list[float] = predicted_params[:num_a].tolist()
                calibrated_sigmas: list[float] = predicted_params[num_a:].tolist()
                print("Predicted Parameters:")
                print(f"  Mean Reversion (a): {calibrated_as}")
                print(f"  Volatility (sigma): {calibrated_sigmas}")
                
                helpers_with_info: List[Tuple[ql.SwaptionHelper, str, str]] = prepare_calibration_helpers(vol_cube_df, term_structure_handle, TF_NN_CALIBRATION_SETTINGS['min_expiry_years'], TF_NN_CALIBRATION_SETTINGS['min_tenor_years'])
                if not helpers_with_info:
                    print(f"Skipping parameter/result saving for {eval_date} due to lack of valid swaption helpers.")
                    continue
                
                included_expiries_yrs: list[float] = sorted(list(set([parse_tenor_to_years(expiry) for _, expiry, _ in helpers_with_info])))
                a_step_dates: list[ql.Date] = _get_step_dates_from_expiries(ql_eval_date, included_expiries_yrs, TF_NN_CALIBRATION_SETTINGS['num_a_segments'])
                sigma_step_dates: list[ql.Date] = _get_step_dates_from_expiries(ql_eval_date, included_expiries_yrs, TF_NN_CALIBRATION_SETTINGS['num_sigma_segments'])

                param_data: dict = {
                    'evaluationDate': eval_date.strftime('%Y-%m-%d'),
                    'meanReversion': {
                        'values': calibrated_as,
                        'stepDates': [datetime.date(d.year(), d.month(), d.dayOfMonth()).strftime('%Y-%m-%d') for d in a_step_dates]
                    },
                    'volatility': {
                        'values': calibrated_sigmas,
                        'stepDates': [datetime.date(d.year(), d.month(), d.dayOfMonth()).strftime('%Y-%m-%d') for d in sigma_step_dates]
                    }
                }
                
                output_date_str: str = eval_date.strftime("%d-%m-%Y")
                param_output_path: str = os.path.join(params_output_dir, f"{output_date_str}.json")
                with open(param_output_path, 'w') as f:
                    json.dump(param_data, f, indent=4)
                print(f"--- Parameters successfully saved to: {param_output_path} ---")

                rmse_bps, results_df = evaluate_model_on_day(eval_date, zero_curve_df, vol_cube_df, predicted_params, TF_NN_CALIBRATION_SETTINGS)
                print(f"Validation RMSE for {eval_date}: {rmse_bps:.4f} bps")
                
                if not results_df.empty:
                    csv_output_path: str = os.path.join(predictions_output_dir, f"{output_date_str}.csv")                    
                    results_df.to_csv(csv_output_path, index=False)
                    print(f"--- Predictions successfully saved to: {csv_output_path} ---")
                else:
                    print(f"--- Evaluation for {eval_date} did not produce results to save. ---")
                
                test_results_for_plot[eval_date] = (rmse_bps, results_df)

            # 7. Plot results for the last day in the test set
            if test_results_for_plot:
                last_test_date: datetime.date = test_files[-1][0]
                print(f"\n--- Plotting calibration results for last test day: {last_test_date} ---")
                _, last_results_df = test_results_for_plot[last_test_date]
                if not last_results_df.empty: plot_calibration_results(last_results_df, last_test_date)
                else: print("Could not plot results as no valid data was returned from evaluation.")

    except (FileNotFoundError, ValueError) as e: print(f"\nERROR: {e}")
    except Exception as e: print(f"\nAn unexpected error occurred: {e}"); traceback.print_exc()
    