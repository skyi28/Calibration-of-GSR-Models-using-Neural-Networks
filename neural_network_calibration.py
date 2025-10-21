# neural_network_calibration.py

"""
This script is designed to train a Neural Network for calibrating Hull-White GSR models.

Its primary role is to be a "model factory." Its purpose is to produce the best
possible trained model artifact, which can then be used by other scripts for evaluation.

This script's execution will:
1.  Discover and split the available data chronologically.
2.  Perform feature engineering, including PCA on interest rates.
3.  Optionally, run a full hyperparameter search using Keras Tuner with Hyperband.
4.  Train a final model on the full training set using the best hyperparameters.
5.  Save all necessary artifacts for prediction: the trained Keras model, the
    feature scaler, the PCA model, the initial logits tensor, and the final
    feature names.

All evaluation-specific logic has been removed from this script. Its sole focus
is on producing the deployable model.

WARNING: This script is computationally intensive and is intended for long-running
execution on powerful hardware.
"""
import datetime
import glob
import math
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
from sklearn.decomposition import PCA
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
import yfinance as yf
import random
import seaborn as sns

# -------------------- REPRODUCIBILITY SEED --------------------
SEED: int = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

#--------------------CONFIG--------------------
PREPROCESS_CURVES: bool = False
BOOTSTRAP_CURVES: bool = False

FOLDER_SWAP_CURVES: str = r'data/EUR SWAP CURVE'
FOLDER_ZERO_CURVES: str = r'data/EUR ZERO CURVE'
FOLDER_VOLATILITY_CUBES: str = r'data/EUR BVOL CUBE'
FOLDER_EXTERNAL_DATA: str = r'data/EXTERNAL'
FOLDER_MODELS: str = r'results/neural_network/models'
FOLDER_HYPERPARAMETERS: str = r'results/neural_network/hyperparameters'

#--------------------PREPROCESS CURVES (UNCHANGED)--------------------
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

#--------------------BOOTSTRAP ZERO CURVES (UNCHANGED)--------------------
def bootstrap_zero_curve_with_quantlib(
    processed_swap_curve: pd.DataFrame,
    valuation_date: datetime.datetime
) -> pd.DataFrame:
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
        if unit == 'MO': tenor_period = ql.Period(term_length, ql.Months)
        elif unit == 'YR': tenor_period = ql.Period(term_length, ql.Years)
        else: raise ValueError(f"Unsupported tenor unit: {row['Unit']}")
        helper: ql.SwapRateHelper = ql.SwapRateHelper(ql.QuoteHandle(ql.SimpleQuote(rate)), tenor_period, calendar, ql.Semiannual, ql.Unadjusted, day_count, swap_index)
        rate_helpers.append(helper)
    yield_curve: ql.PiecewiseLogLinearDiscount = ql.PiecewiseLogLinearDiscount(ql_valuation_date, rate_helpers, day_count)
    yield_curve.enableExtrapolation()
    results: list[dict] = []
    for row in df.itertuples():
        maturity_date = row.Date
        ql_date = ql.Date(maturity_date.day, maturity_date.month, maturity_date.year)
        tenor_frac = day_count.yearFraction(ql_valuation_date, ql_date)
        zero_rate = yield_curve.zeroRate(ql_date, day_count, ql.Continuous).rate()
        discount_factor = yield_curve.discount(ql_date)
        results.append({'Date': maturity_date, 'Tenor': tenor_frac, 'DiscountFactor': discount_factor, 'ZeroRate': zero_rate})
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

#--------------------DATA DISCOVERY AND SPLITTING (UNCHANGED)--------------------
def load_and_split_data_chronologically(
    zero_curve_folder: str,
    vol_cube_folder: str,
    train_split_percentage: float = 50.0,
    validation_split_percentage: float = 20.0,
) -> Tuple[List[Tuple[datetime.date, str, str]], List[Tuple[datetime.date, str, str]], List[Tuple[datetime.date, str, str]]]:
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
    train_files: List[Tuple[datetime.date, str, str]] = available_files[:train_end_index]
    validation_files: List[Tuple[datetime.date, str, str]] = available_files[train_end_index:validation_end_index]
    test_files: List[Tuple[datetime.date, str, str]] = available_files[validation_end_index:]
    if not train_files or not validation_files or not test_files:
        raise ValueError("Data splitting resulted in one or more empty sets. Adjust percentages or add more data.")
    print(f"Splitting data: {len(train_files)} for training, {len(validation_files)} for validation, and {len(test_files)} for testing.")
    return train_files, validation_files, test_files

#--------------------HELPER AND PLOTTING FUNCTIONS (UNCHANGED)--------------------
def load_volatility_cube(file_path: str) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_excel(file_path, engine='openpyxl')
    df.rename(columns={df.columns[1]: 'Type'}, inplace=True)
    for col in df.columns:
        if 'Unnamed' in str(col): df.drop(col, axis=1, inplace=True)
    df['Expiry'] = df['Expiry'].ffill()
    return df

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

def plot_and_save_correlation_matrix(
    raw_features_list: List[List[float]],
    feature_names: List[str],
    save_dir: str,
    title_suffix: str = "",
    show_plots: bool = False
):
    print(f"\n--- Generating Feature Correlation Matrix {title_suffix} ---")
    if not raw_features_list:
        print("Warning: Cannot generate correlation matrix. No training data provided.")
        return
    features_df = pd.DataFrame(raw_features_list, columns=feature_names)
    corr_matrix = features_df.corr()
    plt.figure(figsize=(20, 16))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
    plt.title(f"Feature Correlation Matrix {title_suffix}", fontsize=18)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    filename = f'feature_correlation_matrix{title_suffix.replace(" ", "_")}.png'
    plot_path = os.path.join(save_dir, filename)
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Feature correlation matrix saved to: {plot_path}")
    if show_plots:
        plt.show()
    plt.close(plt.gcf())

def plot_pca_component_loadings(
    pca_model: PCA,
    rate_tenors_in_years: List[float],
    save_dir: str,
    show_plots: bool = False
):
    print("\n--- Visualizing PCA Component Loadings for Interpretation ---")
    components = pca_model.components_
    if np.sum(components[0]) < 0: components[0] = -components[0]
    if components[1][-1] < 0: components[1] = -components[1]
    if np.sum(components[2, [0, -1]]) < 0: components[2] = -components[2]
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=True)
    fig.suptitle('PCA Component Loadings (Eigenvectors)', fontsize=16)
    axes[0].plot(rate_tenors_in_years, components[0], marker='o')
    axes[0].set_title('PC1 - Expected: Level', fontsize=14)
    axes[0].set_xlabel('Tenor (Years)'), axes[0].set_ylabel('Loading'), axes[0].grid(True)
    axes[1].plot(rate_tenors_in_years, components[1], marker='o')
    axes[1].set_title('PC2 - Expected: Slope', fontsize=14)
    axes[1].set_xlabel('Tenor (Years)'), axes[1].grid(True)
    axes[2].plot(rate_tenors_in_years, components[2], marker='o')
    axes[2].set_title('PC3 - Expected: Curvature', fontsize=14)
    axes[2].set_xlabel('Tenor (Years)'), axes[2].grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(save_dir, 'pca_component_loadings.png')
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"PCA component loadings plot saved to: {plot_path}")
    if show_plots:
        plt.show()
    plt.close(fig)

#--------------------PCA HELPER FUNCTIONS (UNCHANGED)--------------------
def fit_pca_on_rates(
    training_features_list: List[List[float]],
    rate_feature_indices: List[int],
    n_components: int = 3
) -> PCA:
    print(f"\n--- Fitting PCA on {len(rate_feature_indices)} rate features ---")
    rate_data = np.array(training_features_list)[:, rate_feature_indices]
    pca = PCA(n_components=n_components)
    pca.fit(rate_data)
    explained_variance = pca.explained_variance_ratio_
    print(f"PCA Components Explained Variance: {explained_variance}")
    print(f"Total variance explained by {n_components} components: {sum(explained_variance):.4f}")
    return pca

def apply_pca_to_features(
    raw_features: List[float],
    pca: PCA,
    rate_feature_indices: List[int]
) -> List[float]:
    raw_features_np = np.array(raw_features)
    rate_values = raw_features_np[rate_feature_indices].reshape(1, -1)
    non_rate_indices = [i for i in range(len(raw_features)) if i not in rate_feature_indices]
    non_rate_values = raw_features_np[non_rate_indices]
    principal_components = pca.transform(rate_values).flatten()
    return principal_components.tolist() + non_rate_values.tolist()

#--------------------TENSORFLOW NEURAL NETWORK CALIBRATION HELPERS (UNCHANGED)--------------------
def _format_time(seconds: float) -> str:
    s: int = int(round(seconds)); h, r = divmod(s, 3600); m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def _get_step_dates_from_expiries(ql_eval_date: ql.Date, included_expiries_yrs: List[float], num_segments: int) -> List[ql.Date]:
    if num_segments <= 1: return []
    unique_expiries: list[float] = sorted(list(set(included_expiries_yrs)))
    if len(unique_expiries) < num_segments: num_segments = len(unique_expiries)
    if num_segments <= 1: return []
    indices: NDArray = np.linspace(0, len(unique_expiries) - 1, num_segments + 1).astype(int)[1:-1]
    time_points_in_years: list[float] = [unique_expiries[i] for i in indices]
    return [ql_eval_date + ql.Period(int(y * 365.25), ql.Days) for y in time_points_in_years]

def _expand_params_to_unified_timeline(initial_params_quotes: List[ql.SimpleQuote], param_step_dates: List[ql.Date], unified_step_dates: List[ql.Date]) -> List[ql.QuoteHandle]:
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
    day_counter: ql.Actual365Fixed = ql.Actual365Fixed()
    rates: list[float] = [term_structure_handle.zeroRate(eval_date + ql.Period(int(ty * 365.25), ql.Days), day_counter, ql.Continuous).rate() for ty in feature_tenors]
    def get_rate(tenor_in_years: float) -> float:
        if tenor_in_years < 1.0:
            return term_structure_handle.zeroRate(eval_date + ql.Period(int(tenor_in_years * 12), ql.Months), day_counter, ql.Continuous).rate()
        return term_structure_handle.zeroRate(eval_date + ql.Period(int(tenor_in_years * 365.25), ql.Days), day_counter, ql.Continuous).rate()
    rate_3m, rate_1y, rate_2y, rate_5y = get_rate(0.25), get_rate(1.0), get_rate(2.0), get_rate(5.0)
    rate_10y, rate_20y, rate_30y = get_rate(10.0), get_rate(20.0), get_rate(30.0)
    slope_3m10s, slope_2s10s, slope_5s30s = rate_10y - rate_3m, rate_10y - rate_2y, rate_30y - rate_5y
    curvature_2y5y10y, curvature_1y2y5y = (2 * rate_5y) - rate_2y - rate_10y, (2 * rate_2y) - rate_1y - rate_5y
    curvature_10y20y30y = (2 * rate_20y) - rate_10y - rate_30y
    curvature_slope_ratio = curvature_2y5y10y / slope_2s10s if abs(slope_2s10s) > 1e-6 else 0.0
    py_date = datetime.date(eval_date.year(), eval_date.month(), eval_date.dayOfMonth())
    pd_timestamp = pd.to_datetime(py_date)
    move_value = external_data.loc[pd_timestamp, 'MOVE_Open']
    vix_value = external_data.loc[pd_timestamp, 'VIX_Open']
    eurusd_value = external_data.loc[pd_timestamp, 'EURUSD_Open']
    move_vix_ratio = move_value / vix_value if vix_value > 1e-6 else 1.0
    all_features = rates + [
        slope_3m10s, slope_2s10s, slope_5s30s,
        curvature_2y5y10y, curvature_1y2y5y, curvature_10y20y30y,
        curvature_slope_ratio,
        move_value, vix_value, move_vix_ratio, eurusd_value
    ]
    return all_features

def prepare_nn_features(
    term_structure_handle: ql.RelinkableYieldTermStructureHandle,
    eval_date: ql.Date,
    scaler: StandardScaler,
    external_data: pd.DataFrame,
    feature_tenors: List[float] = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0],
    pca_model: Optional[PCA] = None,
    rate_indices: Optional[List[int]] = None
) -> np.ndarray:
    raw_features = extract_raw_features(term_structure_handle, eval_date, external_data, feature_tenors)
    if pca_model is not None and rate_indices is not None:
        features_to_scale = apply_pca_to_features(raw_features, pca_model, rate_indices)
    else:
        features_to_scale = raw_features
    return scaler.transform(np.array(features_to_scale).reshape(1, -1))

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
        config = super().get_config()
        config.update(self._config)
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)

def evaluate_model_on_day(
    eval_date: datetime.date,
    zero_curve_df: pd.DataFrame,
    vol_cube_df: pd.DataFrame,
    calibrated_params: NDArray[np.float64],
    settings: dict
) -> Tuple[float, pd.DataFrame]:
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

# ------------------- HYPERPARAMETER TUNING AND TRAINING COMPONENTS (UNCHANGED) -------------------
def _calculate_loss_for_day(params: np.ndarray, ql_eval_date, term_structure_handle, helpers_with_info, settings) -> float:
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
            if error < 0:
                weighted_squared_errors.append((error**2) * underestimation_penalty)
            else:
                weighted_squared_errors.append(error**2)
        return float(np.mean(weighted_squared_errors)) if weighted_squared_errors else 1e6
    except (RuntimeError, ValueError):
        return 1e6

def _perform_training_step(model, optimizer, feature_vector, initial_logits, ql_eval_date, term_structure_handle, helpers_with_info, settings):
    @tf.custom_gradient
    def ql_loss_on_params(params_tensor):
        base_params = params_tensor.numpy()[0]
        loss_val = _calculate_loss_for_day(base_params, ql_eval_date, term_structure_handle, helpers_with_info, settings)
        def grad_fn(dy):
            h = settings['h_relative']
            gradient_method = settings.get('gradient_method', 'forward').lower()
            if gradient_method == 'central':
                def _calc_single_grad(i):
                    p_plus, p_minus = base_params.copy(), base_params.copy()
                    p_plus[i] += h; p_minus[i] -= h
                    loss_plus = _calculate_loss_for_day(p_plus, ql_eval_date, term_structure_handle, helpers_with_info, settings)
                    loss_minus = _calculate_loss_for_day(p_minus, ql_eval_date, term_structure_handle, helpers_with_info, settings)
                    return (loss_plus - loss_minus) / (2 * h)
            else: # Forward difference
                def _calc_single_grad(i):
                    p_plus = base_params.copy(); p_plus[i] += h
                    loss_plus = _calculate_loss_for_day(p_plus, ql_eval_date, term_structure_handle, helpers_with_info, settings)
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
    def build(self, hp: kt.HyperParameters) -> ResidualParameterModel:
        num_layers = hp.Int('num_layers', 1, 5)
        activation = hp.Choice('activation', ['relu', 'tanh'])
        use_dropout = hp.Boolean('use_dropout')
        dropout_rate = hp.Float('dropout_rate', 0.1, 0.5, step=0.1) if use_dropout else 0.0
        layers = [hp.Int(f'neurons_{i}', 16, 128, step=16) for i in range(num_layers)]
        total_params = TF_NN_CALIBRATION_SETTINGS['num_a_segments'] + TF_NN_CALIBRATION_SETTINGS['num_sigma_segments']
        upper_bound = TF_NN_CALIBRATION_SETTINGS['upper_bound']
        model = ResidualParameterModel(
            total_params_to_predict=total_params, upper_bound=upper_bound,
            layers=layers, activation=activation, use_dropout=use_dropout,
            dropout_rate=dropout_rate, dtype=tf.float64
        )
        return model

    def fit(self, hp, model, loaded_train_data, loaded_val_data, settings, feature_scaler, initial_logits, external_data, **kwargs):
        learning_rate = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        penalty = hp.Float("underestimation_penalty", 1.0, 4.0, step=0.5)
        trial_settings = settings.copy()
        trial_settings['underestimation_penalty'] = penalty
        epochs = kwargs.get('epochs', 1)
        trial = next((cb for cb in kwargs.get('callbacks', []) if hasattr(cb, 'trial')), None)
        trial_id = trial.trial.trial_id if trial else "unknown"
        pca_model = trial_settings.get('pca_model')
        rate_indices = trial_settings.get('rate_indices')
        for epoch in range(epochs):
            print(f"\nTrial {trial_id} | Starting Epoch {epoch + 1}/{epochs} | Penalty: {penalty:.2f}")
            epoch_losses = []
            for day_idx, (eval_date, zero_df, vol_df) in enumerate(loaded_train_data):
                print(f"\r  Training day {day_idx + 1}/{len(loaded_train_data)}...", end='', flush=True)
                ql_eval_date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
                ql.Settings.instance().evaluationDate = ql_eval_date
                term_structure = create_ql_yield_curve(zero_df, eval_date)
                helpers = prepare_calibration_helpers(vol_df, term_structure, trial_settings)
                if not helpers: continue
                feature_vec = prepare_nn_features(term_structure, ql_eval_date, feature_scaler, external_data, pca_model=pca_model, rate_indices=rate_indices)
                loss = _perform_training_step(model, optimizer, tf.constant(feature_vec, dtype=tf.float64), initial_logits, ql_eval_date, term_structure, helpers, trial_settings)
                if not np.isnan(loss): epoch_losses.append(np.sqrt(loss) * 10000)
            avg_epoch_rmse = np.mean(epoch_losses) if epoch_losses else float('inf')
            print(f"\n  Trial {trial_id} | Epoch {epoch+1}/{epochs} | Avg Train RMSE (Weighted): {avg_epoch_rmse:.2f} bps")
            val_losses = []
            for eval_date, zero_df, vol_df in loaded_val_data:
                term_structure = create_ql_yield_curve(zero_df, eval_date)
                ql_eval_date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
                feature_vec = prepare_nn_features(term_structure, ql_eval_date, feature_scaler, external_data, pca_model=pca_model, rate_indices=rate_indices)
                predicted_params = model((tf.constant(feature_vec, dtype=tf.float64), initial_logits), training=False).numpy()[0]
                rmse, _ = evaluate_model_on_day(eval_date, zero_df, vol_df, predicted_params, trial_settings)
                if not np.isnan(rmse): val_losses.append(rmse)
            avg_val_rmse = np.mean(val_losses) if val_losses else float('inf')
            print(f"  Trial {trial_id} | Epoch {epoch+1}/{epochs} | Avg Validation RMSE (Unweighted): {avg_val_rmse:.2f} bps")
        if trial:
            checkpoint_path = os.path.join(settings['hyperband_settings']['directory'] + os.sep + settings['hyperband_settings']['project_name'] + f"/trial_{trial_id}", "checkpoint.weights.h5")
            print(f"  Saving checkpoint for trial {trial_id} to {checkpoint_path}")
            model.save_weights(checkpoint_path)
        return { 'val_rmse': avg_val_rmse }

#--------------------TRAINING FUNCTION--------------------
def train_new_model(
    train_files: List,
    val_files: List,
    external_data: pd.DataFrame,
    settings: Dict,
    original_feature_names: List[str],
    feature_tenors: List[float]
) -> None:
    """
    Encapsulates the entire model training pipeline, from feature engineering to
    saving the final model artifact.
    """
    model_id = f"model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_save_dir = os.path.join(FOLDER_MODELS, model_id)
    os.makedirs(model_save_dir, exist_ok=True)
    print(f"\n{'='*25} STARTING NEW TRAINING RUN {'='*25}")
    print(f"Model artifacts will be saved to: {model_save_dir}")
    print(f"Training with {len(train_files)} files, validating with {len(val_files)} files.")

    print("\n--- Preparing Features, PCA, and Scaler using Training Data ---")
    raw_training_features = [extract_raw_features(create_ql_yield_curve(pd.read_csv(zp, parse_dates=['Date']), d), ql.Date(d.day, d.month, d.year), external_data, feature_tenors) for d, zp, _ in train_files]
    plot_and_save_correlation_matrix(raw_training_features, original_feature_names, model_save_dir, title_suffix="(Pre-PCA)", show_plots=settings['show_plots'])
    rate_indices = list(range(len(feature_tenors)))
    pca_model = fit_pca_on_rates(raw_training_features, rate_indices, n_components=3)
    settings['pca_model'], settings['rate_indices'] = pca_model, rate_indices
    plot_pca_component_loadings(pca_model, feature_tenors, model_save_dir, show_plots=settings['show_plots'])
    pca_training_features = [apply_pca_to_features(feats, pca_model, rate_indices) for feats in raw_training_features]
    feature_names = ['PC_Level', 'PC_Slope', 'PC_Curvature'] + original_feature_names[len(feature_tenors):]
    plot_and_save_correlation_matrix(pca_training_features, feature_names, model_save_dir, title_suffix="(Post-PCA)", show_plots=settings['show_plots'])
    feature_scaler = StandardScaler()
    feature_scaler.fit(np.array(pca_training_features))
    print("Feature scaler has been fitted to the PCA-transformed training data.")

    print("\n--- Pre-loading all data into memory ---")
    loaded_train_data = [(d, pd.read_csv(zp, parse_dates=['Date']), load_volatility_cube(vp)) for d, zp, vp in train_files]
    loaded_val_data = [(d, pd.read_csv(zp, parse_dates=['Date']), load_volatility_cube(vp)) for d, zp, vp in val_files]
    print("All training and validation data has been loaded.")

    num_params_to_predict = (settings['num_a_segments'] if settings['optimize_a'] else 0) + settings['num_sigma_segments']
    if len(settings['initial_guess']) != num_params_to_predict: raise ValueError(f"Length of 'initial_guess' must match the total number of parameters to predict.")
    p_scaled = np.clip(np.array(settings['initial_guess'], dtype=np.float64) / settings['upper_bound'], 1e-9, 1 - 1e-9)
    initial_logits = tf.constant([np.log(p_scaled / (1 - p_scaled))], dtype=tf.float64)

    best_hyperparameters = None
    if settings['perform_hyperparameter_tuning']:
        print("\n--- Starting Hyperparameter Tuning with Hyperband ---")
        hypermodel = HullWhiteHyperModel()
        tuner = kt.Hyperband(
            hypermodel=hypermodel,
            objective=kt.Objective("val_rmse", direction="min"),
            overwrite=False,
            **settings['hyperband_settings']
        )
        print("Searching for best hyperparameters...")
        tuner.search(loaded_train_data=loaded_train_data, loaded_val_data=loaded_val_data, settings=settings, feature_scaler=feature_scaler, initial_logits=initial_logits, external_data=external_data)
        best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    else:
        print("\n--- Loading Best Hyperparameters from File ---")
        list_of_files = glob.glob(os.path.join(FOLDER_HYPERPARAMETERS, '*.json'))
        if not list_of_files: raise FileNotFoundError("No hyperparameter files found.")
        latest_file = max(list_of_files, key=os.path.getctime)
        print(f"Loading hyperparameters from: {latest_file}")
        with open(latest_file, 'r') as f: loaded_hps = json.load(f)
        best_hyperparameters = kt.HyperParameters()
        for key, value in loaded_hps.items(): best_hyperparameters.Fixed(key, value)

    print("\n--- Training Final Model using Best Hyperparameters ---")
    settings['underestimation_penalty'] = best_hyperparameters.get('underestimation_penalty')
    final_model = HullWhiteHyperModel().build(best_hyperparameters)
    optimizer = tf.keras.optimizers.Adam(learning_rate=best_hyperparameters.get('learning_rate') or 0.001)
    train_loss_history, val_loss_history = [], []
    best_val_rmse, best_model_weights, best_epoch = float('inf'), None, -1
    patience_counter = 0

    start_time = time.monotonic()
    for epoch in range(settings['num_epochs']):
        epoch_train_losses = []
        for day_idx, (eval_date, zero_df, vol_df) in enumerate(loaded_train_data):
            ql_eval_date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
            ql.Settings.instance().evaluationDate = ql_eval_date
            term_structure = create_ql_yield_curve(zero_df, eval_date)
            helpers = prepare_calibration_helpers(vol_df, term_structure, settings)
            if not helpers: continue
            feature_vec = prepare_nn_features(term_structure, ql_eval_date, feature_scaler, external_data, pca_model=pca_model, rate_indices=rate_indices)
            loss = _perform_training_step(final_model, optimizer, tf.constant(feature_vec, dtype=tf.float64), initial_logits, ql_eval_date, term_structure, helpers, settings)
            current_rmse = np.sqrt(loss) * 10000
            epoch_train_losses.append(current_rmse)
            progress = (day_idx + 1) / len(loaded_train_data)
            bar = ('=' * int(progress * 20)).ljust(20)
            sys.stdout.write(f"\rEpoch {epoch+1:2d}/{settings['num_epochs']} [{bar}] Training Day {day_idx+1:3d}/{len(loaded_train_data)} - Weighted RMSE: {current_rmse:7.2f} bps")
            sys.stdout.flush()

        avg_epoch_train_rmse = np.mean(epoch_train_losses) if epoch_train_losses else float('nan')
        train_loss_history.append(avg_epoch_train_rmse)
        epoch_val_losses = []
        for eval_date, zero_df, vol_df in loaded_val_data:
            term_structure = create_ql_yield_curve(zero_df, eval_date)
            ql_eval_date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
            feature_vec = prepare_nn_features(term_structure, ql_eval_date, feature_scaler, external_data, pca_model=pca_model, rate_indices=rate_indices)
            predicted_params = final_model((tf.constant(feature_vec, dtype=tf.float64), initial_logits), training=False).numpy()[0]
            rmse, _ = evaluate_model_on_day(eval_date, zero_df, vol_df, predicted_params, settings)
            if not np.isnan(rmse): epoch_val_losses.append(rmse)
        avg_epoch_val_rmse = np.mean(epoch_val_losses) if epoch_val_losses else float('inf')
        val_loss_history.append(avg_epoch_val_rmse)
        elapsed = time.monotonic() - start_time
        summary_msg = (f"\nEpoch {epoch+1} Summary | Train RMSE (Weighted): {avg_epoch_train_rmse:7.2f} bps | Validation RMSE: {avg_epoch_val_rmse:7.2f} bps | Time: {_format_time(elapsed)}")
        print(summary_msg)
        if avg_epoch_val_rmse < best_val_rmse:
            best_val_rmse, best_model_weights, best_epoch = avg_epoch_val_rmse, final_model.get_weights(), epoch + 1
            patience_counter = 0
            print(f"  -> New best model found! Validation RMSE: {best_val_rmse:.2f} bps.")
        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience counter: {patience_counter}/{settings['early_stopping_patience']}")
            if patience_counter >= settings['early_stopping_patience']:
                print("Early stopping triggered.")
                break
    print("\n--- Final Training Finished ---")

    fig_hist, ax_hist = plt.subplots(figsize=(12, 6))
    ax_hist.plot(range(1, len(train_loss_history) + 1), train_loss_history, 'o-', color='#0000ff', label='Training RMSE (Weighted)')
    ax_hist.plot(range(1, len(val_loss_history) + 1), val_loss_history, 'o-', color='#ff0000', label='Validation RMSE')
    ax_hist.axvline(x=best_epoch, color='#00004c', linestyle='--', label=f'Best Model (Epoch {best_epoch})')
    ax_hist.set_title('Training and Validation Loss History')
    ax_hist.set_xlabel('Epoch')
    ax_hist.set_ylabel('RMSE (bps)')
    ax_hist.legend(); ax_hist.grid(True)
    fig_hist.tight_layout()
    history_plot_path = os.path.join(model_save_dir, 'training_history.png')
    fig_hist.savefig(history_plot_path)
    print(f"\n--- Training history plot saved to {history_plot_path} ---")
    if settings['show_plots']: plt.show()
    plt.close(fig_hist)

    if best_model_weights:
        print(f"\nRestoring model weights from Epoch {best_epoch} with best validation RMSE: {best_val_rmse:.2f} bps.")
        final_model.set_weights(best_model_weights)

    print(f"\n--- Saving best model artifact to {model_save_dir} ---")
    final_model.save(os.path.join(model_save_dir, 'model.keras'))
    joblib.dump(feature_scaler, os.path.join(model_save_dir, 'feature_scaler.joblib'))
    joblib.dump(pca_model, os.path.join(model_save_dir, 'pca_model.joblib'))
    np.save(os.path.join(model_save_dir, 'initial_logits.npy'), initial_logits.numpy())
    
    # <<< NEWLY ADDED LINE >>>
    # Save the feature names for later use in SHAP analysis
    with open(os.path.join(model_save_dir, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f)
    
    print("--- Model artifact saved successfully ---")
    print(f"{'='*27} TRAINING RUN ENDED {'='*28}\n")

#--------------------MAIN EXECUTION LOGIC--------------------
if __name__ == '__main__':
    try:
        print("="*80)
        print(" NEURAL NETWORK MODEL FACTORY ".center(80))
        print("This script's purpose is to train and save a model artifact for later evaluation.".center(80))
        print("="*80)
        
        os.makedirs(FOLDER_MODELS, exist_ok=True)
        os.makedirs(FOLDER_HYPERPARAMETERS, exist_ok=True)

        TF_NN_CALIBRATION_SETTINGS = {
            "perform_hyperparameter_tuning": False,
            "show_plots": False,
            "hyperband_settings": {"max_epochs": 100, "factor": 3, "directory": "results/neural_network/hyperband_tuner", "project_name": "hull_white_calibration"},
            "num_a_segments": 1, "num_sigma_segments": 7, "optimize_a": True,
            "instrument_batch_size_percentage": 100, "upper_bound": 0.1, "pricing_engine_integration_points": 32,
            "num_epochs": 200,
            "early_stopping_patience": 20,
            "h_relative": 1e-7,
            "initial_guess": [0.02, 0.0002, 0.0002, 0.00017, 0.00017, 0.00017, 0.00017, 0.00017],
            "gradient_method": "forward", "gradient_clip_norm": 2.0, "num_threads": os.cpu_count() or 1,
            "min_expiry_years": 2.0, "min_tenor_years": 2.0, "use_coterminal_only": False,
        }
        
        feature_tenors = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
        original_feature_names = [f'{int(t)}y_rate' for t in feature_tenors] + [
            'slope_3m10y', 'slope_2y10y', 'slope_5y30y',
            'curvature_2y5y10y', 'curvature_1y2y5y', 'curvature_10y20y30y',
            'curvature_slope_ratio',
            'MOVE_Open', 'VIX_Open', 'MOVE_VIX_Ratio', 'EURUSD_Open'
        ]
        
        initial_train_files, initial_val_files, _ = load_and_split_data_chronologically(
            FOLDER_ZERO_CURVES, FOLDER_VOLATILITY_CUBES
        )
        
        all_files = initial_train_files + initial_val_files
        all_files.sort(key=lambda x: x[0])
        
        external_data_csv_path = os.path.join(FOLDER_EXTERNAL_DATA, 'external_market_data.csv')
        start_date, end_date = all_files[0][0], all_files[-1][0]

        if not os.path.exists(external_data_csv_path):
            print(f"\n--- External market data not found. Downloading for range {start_date} to {end_date} ---")
            os.makedirs(FOLDER_EXTERNAL_DATA, exist_ok=True)
            try:
                tickers_to_download = ['^MOVE', '^VIX', 'EURUSD=X']
                raw_data = yf.download(tickers_to_download, start=start_date, end=end_date + datetime.timedelta(days=1))
                if raw_data.empty: raise ValueError("No data downloaded from yfinance.")
                external_data_df = raw_data['Open'].copy()
                external_data_df.rename(columns={'^MOVE': 'MOVE_Open', '^VIX': 'VIX_Open', 'EURUSD=X': 'EURUSD_Open'}, inplace=True)
                external_data_df.to_csv(external_data_csv_path)
                print(f"External market data saved to {external_data_csv_path}")
            except Exception as e:
                print(f"ERROR: Could not download external market data. {e}"), sys.exit(1)
        
        print("\n--- Loading and preparing external market data ---")
        external_data = pd.read_csv(external_data_csv_path, parse_dates=['Date'], index_col='Date')
        external_data = external_data.reindex(pd.date_range(start=start_date, end=end_date, freq='D')).ffill().bfill()
        print("External market data prepared.")
        
        print("\n--- WORKFLOW: TRAINING MODE ---")
        train_new_model(
            train_files=initial_train_files,
            val_files=initial_val_files,
            external_data=external_data,
            settings=TF_NN_CALIBRATION_SETTINGS,
            original_feature_names=original_feature_names,
            feature_tenors=feature_tenors
        )
        print("\n--- SCRIPT FINISHED ---")

    except (FileNotFoundError, ValueError) as e: print(f"\nERROR: {e}")
    except Exception as e: print(f"\nAn unexpected error occurred: {e}"); traceback.print_exc()