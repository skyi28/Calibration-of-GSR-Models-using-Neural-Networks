import datetime
import os
import sys
import time
import pandas as pd
import numpy as np
import QuantLib as ql
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from numpy.typing import NDArray
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from concurrent.futures import ThreadPoolExecutor
import bisect
import traceback
import json

#--------------------CONFIG--------------------
# Set to True to re-run data preparation, False to use existing processed/bootstrapped files.
PREPROCESS_CURVES: bool = False
BOOTSTRAP_CURVES: bool = False

FOLDER_SWAP_CURVES: str = r'data\EUR SWAP CURVE'    # Folder containing the raw swap curves.
FOLDER_ZERO_CURVES: str = r'data\EUR ZERO CURVE'    # Folder containing bootstrapped zero curves or in which the those will be stored.
FOLDER_VOLATILITY_CUBES: str = r'data\EUR BVOL CUBE'

# --- NEW: Folders for Neural Network results ---
FOLDER_RESULTS_NN: str = r'results\neural_network\predictions'
FOLDER_RESULTS_PARAMS_NN: str = r'results\neural_network\parameters'


#--------------------PREPROCESS CURVES (No changes)--------------------
if PREPROCESS_CURVES:
    print("--- Starting: Preprocessing Raw Swap Curves ---")
    processed_folder = os.path.join(FOLDER_SWAP_CURVES, 'processed')
    raw_folder = os.path.join(FOLDER_SWAP_CURVES, 'raw')
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
            processed_swap_curve.drop(['Term', 'Unit'], axis=1, inplace=True)
            processed_swap_curve['Date'] = processed_swap_curve_dates
            processed_swap_curve['Rate'] = swap_curve[['Bid', 'Ask']].mean(axis=1) / 100
            processed_swap_curve.to_csv(os.path.join(processed_folder, f"{swap_curve_date_str}.csv"), index=False)
    print("--- Finished: Preprocessing ---")


#--------------------BOOTSTRAP ZERO CURVES (No changes)--------------------
def _interpolate(x: float, x_points: list[float] | NDArray , y_points: list[float] | NDArray) -> float:
    return float(np.interp(x, x_points, y_points))

def bootstrap_zero_curve(
    processed_swap_curve: pd.DataFrame,
    valuation_date: datetime.datetime,
    freq: int = 2,
    day_count_convention: float = 365.25
) -> pd.DataFrame:
    df: pd.DataFrame = processed_swap_curve.copy()
    df['Tenor'] = df['Date'].apply(lambda d: round((d - valuation_date).days / day_count_convention, 2))
    df = df.sort_values(by='Tenor').reset_index(drop=True)
    known_tenors: list[float] = [0.0]
    known_dfs: list[float] = [1.0]
    zero_rates_list, discount_factors_list = [], []
    for index, row in df.iterrows():
        tenor, swap_rate = row['Tenor'], row['Rate']
        payment_tenors = np.arange(1/freq, tenor + 1/freq, 1/freq)
        sum_known_dfs = sum(_interpolate(float(t), known_tenors, known_dfs) for t in payment_tenors if t < tenor)
        coupon = swap_rate / freq
        new_df = (1 - coupon * sum_known_dfs) / (1 + coupon)
        zero_rate = -np.log(new_df) / tenor
        discount_factors_list.append(new_df)
        zero_rates_list.append(zero_rate)
        known_tenors.append(tenor)
        known_dfs.append(new_df)
    df['DiscountFactor'], df['ZeroRate'] = discount_factors_list, zero_rates_list
    df.drop('Rate', axis=1, inplace=True)
    cols = df.columns.tolist()
    new_order = ['Date', 'Tenor'] + [col for col in cols if col not in ('Date', 'Tenor')]
    return df[new_order]

if BOOTSTRAP_CURVES:
    print("--- Starting: Bootstrapping Zero Curves ---")
    preprocessed_folder = os.path.join(FOLDER_SWAP_CURVES, 'processed')
    if not os.path.exists(FOLDER_ZERO_CURVES):
        os.makedirs(FOLDER_ZERO_CURVES)
    for entry_name in os.listdir(preprocessed_folder):
        if entry_name.endswith('.csv'):
            swap_curve_date_str: str = entry_name.split('.csv')[0]
            swap_curve_date = datetime.datetime.strptime(swap_curve_date_str, '%d.%m.%Y')
            processed_swap_curve = pd.read_csv(os.path.join(preprocessed_folder, entry_name), parse_dates=['Date'])
            zero_curve = bootstrap_zero_curve(processed_swap_curve, swap_curve_date)
            zero_curve.to_csv(os.path.join(FOLDER_ZERO_CURVES, f"{swap_curve_date_str}.csv"), index=False)
    print("--- Finished: Bootstrapping ---")


#--------------------DATA DISCOVERY AND SPLITTING (No changes)--------------------
def load_and_split_data_files(
    zero_curve_folder: str,
    vol_cube_folder: str,
    train_split_percentage: float
) -> Tuple[List[Tuple[datetime.date, str, str]], List[Tuple[datetime.date, str, str]]]:
    print("\n--- Discovering and loading data files ---")
    vol_cube_xlsx_folder = os.path.join(vol_cube_folder, 'xlsx')
    if not os.path.exists(zero_curve_folder) or not os.path.exists(vol_cube_xlsx_folder):
        raise FileNotFoundError(f"Data folders not found. Searched for:\n- {zero_curve_folder}\n- {vol_cube_xlsx_folder}")
    available_files = []
    for entry_name in os.listdir(zero_curve_folder):
        if entry_name.endswith('.csv'):
            date_str = entry_name.replace('.csv', '')
            eval_date = datetime.datetime.strptime(date_str, '%d.%m.%Y').date()
            zero_path = os.path.join(zero_curve_folder, entry_name)
            vol_path = os.path.join(vol_cube_xlsx_folder, f"{date_str}.xlsx")
            if os.path.exists(vol_path):
                available_files.append((eval_date, zero_path, vol_path))
            else:
                print(f"Warning: Zero curve found for {date_str}, but no corresponding volatility cube. Skipping.")
    if not available_files:
        raise ValueError("No matching pairs of zero curve and volatility data found.")
    available_files.sort(key=lambda x: x[0])
    print(f"Found {len(available_files)} complete data sets from {available_files[0][0]} to {available_files[-1][0]}.")
    split_index = int(len(available_files) * (train_split_percentage / 100.0))
    train_files = available_files[:split_index]
    test_files = available_files[split_index:]
    if not train_files:
        raise ValueError("Training split resulted in 0 files. Adjust train_split_percentage or add more data.")
    print(f"Splitting data: {len(train_files)} days for training, {len(test_files)} days for validation.")
    return train_files, test_files

#--------------------NEW: HELPER FOR LOADING VOL CUBE--------------------
def load_volatility_cube(file_path: str) -> pd.DataFrame:
    """Loads and cleans the volatility cube DataFrame."""
    df = pd.read_excel(file_path, engine='openpyxl')
    df.rename(columns={df.columns[1]: 'Type'}, inplace=True) # The 'Type' column is usually the second one
    for col in df.columns:
        if 'Unnamed' in str(col): df.drop(col, axis=1, inplace=True)
    df['Expiry'] = df['Expiry'].ffill()
    return df

#--------------------HELPER AND PLOTTING FUNCTIONS (No changes)--------------------
def parse_tenor(tenor_str: str) -> ql.Period:
    tenor_str = tenor_str.strip().upper()
    if 'YR' in tenor_str: return ql.Period(int(tenor_str.replace('YR', '')), ql.Years)
    if 'MO' in tenor_str: return ql.Period(int(tenor_str.replace('MO', '')), ql.Months)
    raise ValueError(f"Could not parse tenor string: {tenor_str}")

def parse_tenor_to_years(tenor_str: str) -> float:
    tenor_str = tenor_str.strip().upper()
    if 'YR' in tenor_str: return float(int(tenor_str.replace('YR', '')))
    if 'MO' in tenor_str: return int(tenor_str.replace('MO', '')) / 12.0
    raise ValueError(f"Could not parse tenor string to years: {tenor_str}")

def create_ql_yield_curve(
    zero_curve_df: pd.DataFrame, eval_date: datetime.date
) -> ql.RelinkableYieldTermStructureHandle:
    ql_eval_date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
    dates = [ql_eval_date] + [ql.Date(d.day, d.month, d.year) for d in pd.to_datetime(zero_curve_df['Date'])]
    rates = [zero_curve_df['ZeroRate'].iloc[0]] + zero_curve_df['ZeroRate'].tolist()
    term_structure = ql.ZeroCurve(dates, rates, ql.Actual365Fixed(), ql.TARGET(), ql.Linear(), ql.Continuous, ql.Annual)
    term_structure.enableExtrapolation()
    handle = ql.RelinkableYieldTermStructureHandle()
    handle.linkTo(term_structure)
    return handle

def prepare_calibration_helpers(
    vol_cube_df: pd.DataFrame,
    term_structure_handle: ql.RelinkableYieldTermStructureHandle,
    min_expiry_years: float = 0.0,
    min_tenor_years: float = 0.0
) -> List[Tuple[ql.SwaptionHelper, str, str]]:
    helpers_with_info = []
    vols_df = vol_cube_df[vol_cube_df['Type'] == 'Vol'].set_index('Expiry')
    strikes_df = vol_cube_df[vol_cube_df['Type'] == 'Strike'].set_index('Expiry')
    swap_index = ql.Euribor6M(term_structure_handle)
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
            vol_handle = ql.QuoteHandle(ql.SimpleQuote(vol / 10000.0))
            helper = ql.SwaptionHelper(parse_tenor(expiry_str), parse_tenor(tenor_str), vol_handle, swap_index, ql.Period(6, ql.Months), swap_index.dayCounter(), swap_index.dayCounter(), term_structure_handle, ql.Normal)
            helpers_with_info.append((helper, expiry_str, tenor_str))
    return helpers_with_info

def plot_calibration_results(results_df: pd.DataFrame, eval_date: datetime.date):
    plot_data = results_df.dropna(subset=['MarketVol', 'ModelVol', 'Difference']).copy()
    if plot_data.empty:
        print(f"\nCould not generate plots for {eval_date}: No valid data points available.")
        return
    X = plot_data['Expiry'].values; Y = plot_data['Tenor'].values
    Z_market = plot_data['MarketVol'].values; Z_model = plot_data['ModelVol'].values
    Z_diff = plot_data['Difference'].values
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
    max_reasonable_diff = np.nanmax(np.abs(Z_diff)); ax3.set_zlim(-max_reasonable_diff, max_reasonable_diff)
    ax3.view_init(elev=30, azim=-120); plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()


#--------------------TENSORFLOW NEURAL NETWORK CALIBRATION HELPERS (No changes to these helpers)--------------------
def _format_time(seconds: float) -> str:
    s = int(round(seconds)); h, r = divmod(s, 3600); m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def _get_step_dates_from_expiries(ql_eval_date: ql.Date, included_expiries_yrs: List[float], num_segments: int) -> List[ql.Date]:
    if num_segments <= 1: return []
    unique_expiries = sorted(list(set(included_expiries_yrs)))
    if len(unique_expiries) < num_segments: num_segments = len(unique_expiries)
    if num_segments <= 1: return []
    indices = np.linspace(0, len(unique_expiries) - 1, num_segments + 1).astype(int)[1:-1]
    time_points_in_years = [unique_expiries[i] for i in indices]
    return [ql_eval_date + ql.Period(int(y * 365.25), ql.Days) for y in time_points_in_years]

def _expand_params_to_unified_timeline(initial_params_quotes: List[ql.SimpleQuote], param_step_dates: List[ql.Date], unified_step_dates: List[ql.Date]) -> List[ql.QuoteHandle]:
    initial_params_handles = [ql.QuoteHandle(q) for q in initial_params_quotes]
    if not unified_step_dates: return initial_params_handles
    if not param_step_dates: return [initial_params_handles[0]] * (len(unified_step_dates) + 1)
    expanded_handles = []; time_intervals = [float('-inf')] + [d.serialNumber() for d in unified_step_dates] + [float('inf')]
    for i in range(len(time_intervals) - 1):
        mid_point_serial = (time_intervals[i] + time_intervals[i+1]) / 2
        original_date_serials = [d.serialNumber() for d in param_step_dates]
        idx = bisect.bisect_right(original_date_serials, mid_point_serial)
        expanded_handles.append(initial_params_handles[idx])
    return expanded_handles

# MODIFIED: This function now accepts a pre-fitted scaler.
def prepare_nn_features(
    term_structure_handle: ql.RelinkableYieldTermStructureHandle,
    eval_date: ql.Date,
    scaler: StandardScaler,
    feature_tenors: List[float] = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
) -> np.ndarray:
    """
    Calculates zero rates for given tenors and scales them using a pre-fitted scaler.
    """
    day_counter = ql.Actual365Fixed()
    rates = [term_structure_handle.zeroRate(eval_date + ql.Period(int(ty * 365.25), ql.Days), day_counter, ql.Continuous).rate() for ty in feature_tenors]
    # Use the pre-fitted scaler to ONLY TRANSFORM the data. Do not fit again.
    return scaler.transform(np.array(rates).reshape(1, -1))

class ResidualParameterModel(tf.keras.Model):
    def __init__(self, total_params_to_predict, upper_bound, hidden_layer_size=64, batch_momentum=0.99, name=None):
        super().__init__(name=name, dtype='float64')
        self.upper_bound = tf.constant(upper_bound, dtype=tf.float64)
        self.hidden1 = tf.keras.layers.Dense(hidden_layer_size, dtype=tf.float64)
        # self.bn1 = tf.keras.layers.BatchNormalization(momentum=batch_momentum, dtype=tf.float64)
        self.hidden2 = tf.keras.layers.Dense(hidden_layer_size, dtype=tf.float64)
        # self.bn2 = tf.keras.layers.BatchNormalization(momentum=batch_momentum, dtype=tf.float64)
        self.output_layer = tf.keras.layers.Dense(total_params_to_predict, dtype=tf.float64, kernel_initializer='zeros', bias_initializer='zeros')

    def call(self, inputs, training=False):
        feature_vector, initial_logits = inputs
        x = self.hidden1(feature_vector)
        # x = self.bn1(x, training=training)
        x = tf.keras.activations.relu(x)
        x = self.hidden2(x)
        # x = self.bn2(x, training=training)
        x = tf.keras.activations.relu(x)
        delta_logits = self.output_layer(x)
        final_logits = initial_logits + delta_logits
        return self.upper_bound * tf.keras.activations.sigmoid(final_logits)


#--------------------REFACTORED: EVALUATION FUNCTION (No changes)--------------------
def evaluate_model_on_day(
    eval_date: datetime.date,
    zero_curve_df: pd.DataFrame,
    vol_cube_df: pd.DataFrame,
    calibrated_params: List[float],
    settings: dict
) -> Tuple[float, pd.DataFrame]:
    ql_eval_date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
    ql.Settings.instance().evaluationDate = ql_eval_date
    term_structure_handle = create_ql_yield_curve(zero_curve_df, eval_date)
    helpers_with_info = prepare_calibration_helpers(vol_cube_df, term_structure_handle, settings.get("min_expiry_years", 0.0), settings.get("min_tenor_years", 0.0))
    if not helpers_with_info: return float('nan'), pd.DataFrame()
    num_a_params = settings['num_a_segments'] if settings['optimize_a'] else 0
    calibrated_as = calibrated_params[:num_a_params]
    calibrated_sigmas = calibrated_params[num_a_params:]
    if not settings['optimize_a']: calibrated_as = settings['initial_guess'][:num_a_params] if num_a_params > 0 else []
    included_expiries_yrs = sorted(list(set([parse_tenor_to_years(expiry) for _, expiry, _ in helpers_with_info])))
    a_step_dates = _get_step_dates_from_expiries(ql_eval_date, included_expiries_yrs, settings['num_a_segments'])
    sigma_step_dates = _get_step_dates_from_expiries(ql_eval_date, included_expiries_yrs, settings['num_sigma_segments'])
    unified_step_dates = sorted(list(set(a_step_dates + sigma_step_dates)))
    reversion_quotes = [ql.SimpleQuote(p) for p in calibrated_as]
    sigma_quotes = [ql.SimpleQuote(p) for p in calibrated_sigmas]
    if not settings['optimize_a']: reversion_quotes = [ql.SimpleQuote(p) for p in settings['initial_guess'][:num_a_params]] if num_a_params > 0 else [ql.SimpleQuote(0.01)]
    expanded_reversion_handles = _expand_params_to_unified_timeline(reversion_quotes, a_step_dates, unified_step_dates)
    expanded_sigma_handles = _expand_params_to_unified_timeline(sigma_quotes, sigma_step_dates, unified_step_dates)
    final_model = ql.Gsr(term_structure_handle, unified_step_dates, expanded_sigma_handles, expanded_reversion_handles, 61.0)
    final_engine = ql.Gaussian1dSwaptionEngine(final_model, 64, 7.0, True, False, term_structure_handle)
    results_data = []; squared_errors = []
    for helper, expiry_str, tenor_str in helpers_with_info:
        helper.setPricingEngine(final_engine)
        market_vol_bps = helper.volatility().value() * 10000
        try:
            model_npv = helper.modelValue()
            model_vol = helper.impliedVolatility(model_npv, 1e-4, 500, 0.0001, 1.0)
            model_vol_bps = model_vol * 10000; error_bps = model_vol_bps - market_vol_bps
            squared_errors.append((model_vol - helper.volatility().value())**2)
        except (RuntimeError, ValueError):
            model_vol_bps, error_bps = float('nan'), float('nan')
        results_data.append({'ExpiryStr': expiry_str, 'TenorStr': tenor_str, 'MarketVol': market_vol_bps, 'ModelVol': model_vol_bps, 'Difference': error_bps})
    results_df = pd.DataFrame(results_data)
    if not results_df.empty:
        results_df['Expiry'] = results_df['ExpiryStr'].apply(parse_tenor_to_years); results_df['Tenor'] = results_df['TenorStr'].apply(parse_tenor_to_years)
    final_rmse_bps = np.sqrt(np.mean(squared_errors)) * 10000 if squared_errors else float('nan')
    return final_rmse_bps, results_df


#--------------------MAIN EXECUTION LOGIC--------------------
if __name__ == '__main__':
    try:
        # --- Create results directories if they don't exist ---
        os.makedirs(FOLDER_RESULTS_NN, exist_ok=True)
        os.makedirs(FOLDER_RESULTS_PARAMS_NN, exist_ok=True)

        TRAIN_SPLIT_PERCENTAGE = 80.0
        
        TF_NN_CALIBRATION_SETTINGS = {
            "num_a_segments": 1,
            "num_sigma_segments": 3,
            "optimize_a": True,
            "upper_bound": 0.1,
            "learning_rate_config": {"initial_learning_rate": 1e-2, "decay_steps": 100, "decay_rate": 0.95},
            "num_epochs": 15,
            "h_relative": 1e-7,
            "hidden_layer_size": 32,
            "initial_guess": [0.020566979477481182, 0.0002299520131905028, 0.00021618386141957264, 0.00017446618702943582],
            "gradient_clip_norm": 2.0,
            "num_threads": 8,
            "batch_momentum": 0.99,
            "min_expiry_years": 1.0,
            "min_tenor_years": 1.0
        }
        
        # 1. Load and split all available data
        train_files, test_files = load_and_split_data_files(FOLDER_ZERO_CURVES, FOLDER_VOLATILITY_CUBES, TRAIN_SPLIT_PERCENTAGE)
        if not test_files: print("Warning: No files left for testing/validation after split.")

        # 2. Prepare the feature scaler using only training data
        print("\n--- Preparing Feature Scaler using Training Data ---")
        all_training_features = []
        feature_tenors_for_scaling = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
        for eval_date, zero_path, _ in train_files:
            zero_curve_df = pd.read_csv(zero_path)
            ql_eval_date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
            term_structure_handle = create_ql_yield_curve(zero_curve_df, eval_date)
            day_counter = ql.Actual365Fixed()
            rates = [term_structure_handle.zeroRate(ql_eval_date + ql.Period(int(ty * 365.25), ql.Days), day_counter, ql.Continuous).rate() for ty in feature_tenors_for_scaling]
            all_training_features.append(rates)
        
        feature_scaler = StandardScaler()
        if not all_training_features:
            raise ValueError("Could not extract any features from the training data to fit the scaler.")
        feature_scaler.fit(np.array(all_training_features))
        print("Feature scaler has been fitted to the training data.")

        # OPTIMIZATION: Pre-load all training data into memory
        print("\n--- Pre-loading all training data into memory ---")
        loaded_train_data = []
        for i, (eval_date, zero_path, vol_path) in enumerate(train_files):
            sys.stdout.write(f"\rLoading file {i+1}/{len(train_files)}: {os.path.basename(zero_path)}")
            sys.stdout.flush()
            zero_curve_df = pd.read_csv(zero_path)
            vol_cube_df = load_volatility_cube(vol_path)
            loaded_train_data.append((eval_date, zero_curve_df, vol_cube_df))
        print("\nAll training data has been loaded.")

        # 3. Initialize the shared model and optimizer
        num_a_params_to_predict = TF_NN_CALIBRATION_SETTINGS['num_a_segments'] if TF_NN_CALIBRATION_SETTINGS['optimize_a'] else 0
        total_params_to_predict = num_a_params_to_predict + TF_NN_CALIBRATION_SETTINGS['num_sigma_segments']
        assert len(TF_NN_CALIBRATION_SETTINGS['initial_guess']) == total_params_to_predict, \
            f"Initial guess length must match total predictable parameters ({total_params_to_predict})"

        nn_model = ResidualParameterModel(total_params_to_predict, TF_NN_CALIBRATION_SETTINGS['upper_bound'],
                                          TF_NN_CALIBRATION_SETTINGS['hidden_layer_size'], TF_NN_CALIBRATION_SETTINGS['batch_momentum'])
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(**TF_NN_CALIBRATION_SETTINGS['learning_rate_config'])
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        p_scaled = np.clip(np.array(TF_NN_CALIBRATION_SETTINGS['initial_guess'], dtype=np.float64) / TF_NN_CALIBRATION_SETTINGS['upper_bound'], 1e-9, 1 - 1e-9)
        initial_logits = tf.constant([np.log(p_scaled / (1 - p_scaled))], dtype=tf.float64)

        # 4. --- TRAINING LOOP ---
        print("\n--- Starting Model Training ---")
        start_time = time.monotonic()
        for epoch in range(TF_NN_CALIBRATION_SETTINGS['num_epochs']):
            epoch_losses = []
            for day_idx, (eval_date, zero_curve_df, vol_cube_df) in enumerate(loaded_train_data):
                ql_eval_date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
                ql.Settings.instance().evaluationDate = ql_eval_date
                term_structure_handle = create_ql_yield_curve(zero_curve_df, eval_date)
                helpers_with_info = prepare_calibration_helpers(vol_cube_df, term_structure_handle,
                                                              TF_NN_CALIBRATION_SETTINGS['min_expiry_years'], TF_NN_CALIBRATION_SETTINGS['min_tenor_years'])
                if not helpers_with_info:
                    print(f"\nSkipping {eval_date} due to lack of valid swaption helpers.")
                    continue
                
                feature_vector = prepare_nn_features(term_structure_handle, ql_eval_date, feature_scaler)
                feature_vector = tf.constant(feature_vector, dtype=tf.float64)

                def _calculate_loss_for_day(params: np.ndarray) -> float:
                    try:
                        num_a = num_a_params_to_predict; a_params, sigma_params = params[:num_a], params[num_a:]
                        reversion_quotes = [ql.SimpleQuote(p) for p in a_params]
                        sigma_quotes = [ql.SimpleQuote(p) for p in sigma_params]
                        if not TF_NN_CALIBRATION_SETTINGS['optimize_a']: reversion_quotes = [ql.SimpleQuote(p) for p in TF_NN_CALIBRATION_SETTINGS['initial_guess'][:num_a]] if num_a > 0 else [ql.SimpleQuote(0.01)]
                        included_expiries = sorted(list(set([parse_tenor_to_years(e) for _, e, _ in helpers_with_info])))
                        a_steps = _get_step_dates_from_expiries(ql_eval_date, included_expiries, TF_NN_CALIBRATION_SETTINGS['num_a_segments'])
                        s_steps = _get_step_dates_from_expiries(ql_eval_date, included_expiries, TF_NN_CALIBRATION_SETTINGS['num_sigma_segments'])
                        unified_steps = sorted(list(set(a_steps + s_steps)))
                        exp_rev_h = _expand_params_to_unified_timeline(reversion_quotes, a_steps, unified_steps)
                        exp_sig_h = _expand_params_to_unified_timeline(sigma_quotes, s_steps, unified_steps)
                        thread_model = ql.Gsr(term_structure_handle, unified_steps, exp_sig_h, exp_rev_h, 61.0)
                        thread_engine = ql.Gaussian1dSwaptionEngine(thread_model, 64, 7.0, True, False, term_structure_handle)
                        squared_errors = []
                        for original_helper, _, _ in helpers_with_info:
                            original_helper.setPricingEngine(thread_engine); model_val = original_helper.modelValue()
                            implied_vol = original_helper.impliedVolatility(model_val, 1e-4, 500, 0.0001, 1.0)
                            squared_errors.append((implied_vol - original_helper.volatility().value())**2)
                        return np.mean(squared_errors) if squared_errors else 1e6
                    except (RuntimeError, ValueError): return 1e6

                @tf.custom_gradient
                def ql_loss_on_params(params_tensor):
                    loss_val = _calculate_loss_for_day(params_tensor.numpy()[0])
                    def grad_fn(dy):
                        base_params = params_tensor.numpy()[0]; h = TF_NN_CALIBRATION_SETTINGS['h_relative']
                        def _calc_single_grad(i):
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
                
                current_rmse = tf.sqrt(loss).numpy() * 10000; epoch_losses.append(current_rmse)
                progress = (day_idx + 1) / len(loaded_train_data); bar = ('=' * int(progress * 20)).ljust(20)
                sys.stdout.write(f"\rEpoch {epoch+1:2d}/{TF_NN_CALIBRATION_SETTINGS['num_epochs']} [{bar}] Day {day_idx+1:2d}/{len(loaded_train_data)} - RMSE: {current_rmse:7.2f} bps")
                sys.stdout.flush()

            avg_epoch_rmse = np.mean(epoch_losses) if epoch_losses else float('nan'); elapsed = time.monotonic() - start_time
            print(f"\nEpoch {epoch+1} Summary | Avg. Train RMSE: {avg_epoch_rmse:7.2f} bps | Time Elapsed: {_format_time(elapsed)}")
        
        print("\n--- Training Finished ---")

        # 5. --- VALIDATION/TESTING AND SAVING LOOP ---
        if test_files:
            print("\n--- Evaluating Model on Test Data and Saving Results ---")
            test_results_for_plot = {}
            for eval_date, zero_path, vol_path in test_files:
                print(f"\n--- Processing test day: {eval_date.strftime('%d-%m-%Y')} ---")
                zero_curve_df = pd.read_csv(zero_path)
                vol_cube_df = load_volatility_cube(vol_path)
                ql_eval_date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
                ql.Settings.instance().evaluationDate = ql_eval_date
                term_structure_handle = create_ql_yield_curve(zero_curve_df, eval_date)
                
                # Predict parameters with the trained model
                feature_vector = prepare_nn_features(term_structure_handle, ql_eval_date, feature_scaler)
                predicted_params = nn_model((tf.constant(feature_vector, dtype=tf.float64), initial_logits), training=False).numpy()[0]
                
                # Separate predicted 'a' and 'sigma' parameters
                num_a = num_a_params_to_predict
                calibrated_as = predicted_params[:num_a].tolist()
                calibrated_sigmas = predicted_params[num_a:].tolist()
                print("Predicted Parameters:")
                print(f"  Mean Reversion (a): {calibrated_as}")
                print(f"  Volatility (sigma): {calibrated_sigmas}")
                
                # --- NEW: Save Predicted Parameters to JSON ---
                # First, determine the step dates for this specific day's data
                helpers_with_info = prepare_calibration_helpers(vol_cube_df, term_structure_handle, TF_NN_CALIBRATION_SETTINGS['min_expiry_years'], TF_NN_CALIBRATION_SETTINGS['min_tenor_years'])
                if not helpers_with_info:
                    print(f"Skipping parameter/result saving for {eval_date} due to lack of valid swaption helpers.")
                    continue
                
                included_expiries_yrs = sorted(list(set([parse_tenor_to_years(expiry) for _, expiry, _ in helpers_with_info])))
                a_step_dates = _get_step_dates_from_expiries(ql_eval_date, included_expiries_yrs, TF_NN_CALIBRATION_SETTINGS['num_a_segments'])
                sigma_step_dates = _get_step_dates_from_expiries(ql_eval_date, included_expiries_yrs, TF_NN_CALIBRATION_SETTINGS['num_sigma_segments'])

                param_data = {
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
                
                output_date_str = eval_date.strftime("%d-%m-%Y")
                param_output_path = os.path.join(FOLDER_RESULTS_PARAMS_NN, f"{output_date_str}.json")
                with open(param_output_path, 'w') as f:
                    json.dump(param_data, f, indent=4)
                print(f"--- Parameters successfully saved to: {param_output_path} ---")

                # Evaluate the model to get the results dataframe and RMSE
                rmse_bps, results_df = evaluate_model_on_day(eval_date, zero_curve_df, vol_cube_df, predicted_params, TF_NN_CALIBRATION_SETTINGS)
                print(f"Validation RMSE for {eval_date}: {rmse_bps:.4f} bps")
                
                # --- NEW: Save Prediction Results to CSV ---
                if not results_df.empty:
                    csv_output_path = os.path.join(FOLDER_RESULTS_NN, f"{output_date_str}.csv")
                    results_df.to_csv(csv_output_path, index=False)
                    print(f"--- Predictions successfully saved to: {csv_output_path} ---")
                else:
                     print(f"--- Evaluation for {eval_date} did not produce results to save. ---")
                
                test_results_for_plot[eval_date] = (rmse_bps, results_df)

            # 6. Plot results for the last day in the test set
            if test_results_for_plot:
                last_test_date = test_files[-1][0]
                print(f"\n--- Plotting calibration results for last test day: {last_test_date} ---")
                _, last_results_df = test_results_for_plot[last_test_date]
                if not last_results_df.empty: plot_calibration_results(last_results_df, last_test_date)
                else: print("Could not plot results as no valid data was returned from evaluation.")

    except (FileNotFoundError, ValueError) as e: print(f"\nERROR: {e}")
    except Exception as e: print(f"\nAn unexpected error occurred: {e}"); traceback.print_exc()