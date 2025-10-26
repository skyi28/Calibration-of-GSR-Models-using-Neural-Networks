# sensitivity_analysis_combined.py

"""
Combined Neural Network and Traditional Model Sensitivity Analysis Script

Objective:
This script performs a sensitivity analysis on both a trained Hull-White parameter
prediction Neural Network and a traditional QuantLib-based calibrator. It serves
as a powerful diagnostic tool to compare and validate the economic behavior of
the two distinct modeling approaches.

Methodology:
1.  A single, pre-trained Neural Network model and its artifacts are loaded.
2.  Market data for a specific analysis date is loaded, including the zero
    curve, volatility cube, and external market indicators (e.g., VIX).
3.  A "Base Case" prediction is generated using the original market data for
    both the Neural Network and the Traditional Calibrator.
4.  A series of pre-defined, economically meaningful shocks are applied to the
    input data.
    - Yield curve shocks affect both models.
    - Market stress shocks affect VIX/MOVE for the NN and the volatility
      cube for the traditional calibrator.
5.  Each model generates new parameter predictions for each shocked scenario.
6.  The results are compiled into a unified summary table and saved to a CSV.
7.  Grouped bar charts are generated to visualize the percentage change in
    predicted parameters for both models, side-by-side.
"""
import datetime
import os
import sys
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Dict, Any, List, Tuple, Optional, Union

# --- Import and Setup QuantLib ---
try:
    import QuantLib as ql
except ImportError:
    print("ERROR: QuantLib-Python is not installed. Please install it to run the traditional calibrator.")
    sys.exit(1)

# --- Ensure the script can find the other project modules ---
try:
    from neural_network_calibration import create_ql_yield_curve as create_ql_yield_curve_nn, ResidualParameterModel, extract_raw_features, apply_pca_to_features
except ImportError:
    print("ERROR: Could not import from 'neural_network_calibration'.")
    print("Ensure this script is in the same directory or the project is in your PYTHONPATH.")
    sys.exit(1)


# ==============================================================================
# --- SCRIPT CONFIGURATION ---
# ==============================================================================

# --- 1. Model and Data Selection ---
LATEST_MODEL_DIR = 'results/neural_network/models/model_20251021_004632'  # Set to None to auto-detect latest
ANALYSIS_DATE_STR = "04.08.2025"

# --- 2. Model Structure (Must match the trained model) ---
MODEL_PARAMETERS = {
    "num_a_segments": 1,
    "num_sigma_segments": 7,
    "optimize_a": True,
}
FEATURE_TENORS = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]

# --- 3. Traditional Calibration Settings ---
# These settings are used for the traditional calibrator runs.
TRADITIONAL_CALIBRATION_SETTINGS = {
    "num_a_segments": 1,
    "num_sigma_segments": 7,
    "optimize_a": True,
    "initial_a": 0.01821630830602023,
    "initial_sigma": [0.00021102917641460398, 0.00026161313022069355, 0.0002433782032732618, 0.00023207666803595517, 0.00019547828248440554, 0.00013296374054782453, 0.0001745673951321553], # A simple initial guess for sigma
    "pricing_engine_integration_points": 32,
    "min_expiry_years": 2.0,
    "min_tenor_years": 2.0
}

# --- 4. Scenario Definitions ---
SCENARIO_SETTINGS = {
    "YC_Shift_Up_50bps": {
        "type": "yield_curve_shift",
        "params": {"shift_bps": 50}
    },
    "YC_Shift_Down_50bps": {
        "type": "yield_curve_shift",
        "params": {"shift_bps": -50}
    },
    "YC_Twist_Steepen": {
        "type": "yield_curve_twist",
        "params": {
            "short_end_shift_bps": -25,
            "long_end_shift_bps": 25,
            "pivot_tenor_yrs": 3.0
        }
    },
    # "Market_Stress_25_Percent": {
    #     "type": "market_stress_shock",
    #     # This will be applied to VIX/MOVE for the NN
    #     "params_nn": {"stress_factor": 1.25},
    #     # and to the vol cube for the traditional calibrator
    #     "params_trad": {"vol_shock_factor": 1.25}
    # },
}

CHANGE_PLUS_COLOR = cm.get_cmap('coolwarm', 10)(1)
CHANGE_MINUS_COLOR = cm.get_cmap('coolwarm', 10)(9)

# --- 5. Paths and Directories ---
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FOLDER_ZERO_CURVES: str = os.path.join(DATA_DIR, 'EUR ZERO CURVE')
FOLDER_VOLATILITY_CUBES: str = os.path.join(DATA_DIR, 'EUR BVOL CUBE') # Added for traditional calib
FOLDER_EXTERNAL_DATA: str = os.path.join(DATA_DIR, 'EXTERNAL')
FOLDER_NN_MODELS: str = os.path.join(RESULTS_DIR, 'neural_network/models')
FOLDER_SENSITIVITY_RESULTS: str = os.path.join(RESULTS_DIR, 'sensitivity_analysis')


# ==============================================================================
# --- HELPER FUNCTIONS (NN & General) ---
# ==============================================================================

def load_nn_artifacts_standalone(model_dir: str) -> Dict[str, Any]:
    """Loads all necessary artifacts for the trained NN model."""
    print(f"\n--- Loading Neural Network artifacts from: {model_dir} ---")
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    artifacts = {
        'model': tf.keras.models.load_model(
            os.path.join(model_dir, 'model.keras'),
            custom_objects={'ResidualParameterModel': ResidualParameterModel}
        ),
        'scaler': joblib.load(os.path.join(model_dir, 'feature_scaler.joblib')),
        'pca_model': joblib.load(os.path.join(model_dir, 'pca_model.joblib')),
        'initial_logits': tf.constant(np.load(os.path.join(model_dir, 'initial_logits.npy')), dtype=tf.float64)
    }
    print("All NN artifacts loaded successfully.")
    return artifacts


# ==============================================================================
# --- HELPER FUNCTIONS (Traditional Calibration - Integrated) ---
# ==============================================================================

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

def create_ql_yield_curve_trad(
    zero_curve_df: pd.DataFrame, eval_date: datetime.date
) -> ql.RelinkableYieldTermStructureHandle:
    """Creates a QuantLib yield curve from a pandas DataFrame."""
    ql_eval_date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
    dates = [ql_eval_date] + [ql.Date(d.day, d.month, d.year) for d in pd.to_datetime(zero_curve_df['Date'])]
    rates = [zero_curve_df['ZeroRate'].iloc[0]] + zero_curve_df['ZeroRate'].tolist()
    term_structure = ql.ZeroCurve(dates, rates, ql.Actual365Fixed(), ql.TARGET(), ql.Linear(), ql.Continuous, ql.Annual)
    term_structure.enableExtrapolation()
    handle = ql.RelinkableYieldTermStructureHandle()
    handle.linkTo(term_structure)
    return handle

def prepare_calibration_helpers(
    vol_cube_df: pd.DataFrame, term_structure_handle: ql.RelinkableYieldTermStructureHandle,
    min_expiry_years: float, min_tenor_years: float
) -> List[Tuple[ql.SwaptionHelper, str, str]]:
    """Prepares a list of QuantLib SwaptionHelper objects from the volatility cube."""
    helpers_with_info = []
    vols_df = vol_cube_df[vol_cube_df['Type'] == 'Vol'].set_index('Expiry')
    strikes_df = vol_cube_df[vol_cube_df['Type'] == 'Strike'].set_index('Expiry')
    swap_index = ql.Euribor6M(term_structure_handle)

    for expiry_str in vols_df.index:
        for tenor_str in vols_df.columns:
            if tenor_str == 'Type': continue
            vol, strike = vols_df.loc[expiry_str, tenor_str], strikes_df.loc[expiry_str, tenor_str]
            if pd.isna(vol) or pd.isna(strike): continue

            if parse_tenor_to_years(expiry_str) < min_expiry_years or parse_tenor_to_years(tenor_str) < min_tenor_years: continue

            helper = ql.SwaptionHelper(parse_tenor(expiry_str), parse_tenor(tenor_str),
                                       ql.QuoteHandle(ql.SimpleQuote(vol / 10000.0)),
                                       swap_index, ql.Period(6, ql.Months),
                                       swap_index.dayCounter(), swap_index.dayCounter(), term_structure_handle, ql.Normal)
            helpers_with_info.append((helper, expiry_str, tenor_str))
    print(f"  -> Prepared {len(helpers_with_info)} swaption helpers for traditional calibration.")
    return helpers_with_info

def _get_step_dates_from_expiries(
    ql_eval_date: ql.Date, included_expiries_yrs: List[float], num_segments: int
) -> List[ql.Date]:
    if num_segments <= 1: return []
    unique_expiries = sorted(list(set(included_expiries_yrs)))
    if len(unique_expiries) < num_segments: num_segments = len(unique_expiries)
    if num_segments <= 1: return []
    indices = np.linspace(0, len(unique_expiries) - 1, num_segments + 1).astype(int)[1:-1]
    return [ql_eval_date + ql.Period(int(y * 365.25), ql.Days) for y in [unique_expiries[i] for i in indices]]

def _expand_params_to_unified_timeline(
    initial_params: List[ql.QuoteHandle], param_step_dates: List[ql.Date], unified_step_dates: List[ql.Date]
) -> List[ql.QuoteHandle]:
    if not unified_step_dates: return initial_params
    if not param_step_dates: return [initial_params[0]] * (len(unified_step_dates) + 1)
    expanded_handles = []
    time_intervals = [float('-inf')] + [d.serialNumber() for d in unified_step_dates] + [float('inf')]
    for i in range(len(time_intervals) - 1):
        mid_point_serial = (time_intervals[i] + time_intervals[i+1]) / 2
        import bisect
        original_date_serials = [d.serialNumber() for d in param_step_dates]
        idx = bisect.bisect_right(original_date_serials, mid_point_serial)
        expanded_handles.append(initial_params[idx])
    return expanded_handles


# ==============================================================================
# --- SHOCK APPLICATION FUNCTIONS ---
# ==============================================================================

def apply_yield_curve_shift(zero_df: pd.DataFrame, shift_bps: float) -> pd.DataFrame:
    """Applies a parallel shift to the zero curve."""
    df_shocked = zero_df.copy()
    df_shocked['ZeroRate'] += shift_bps / 10000.0
    return df_shocked

def apply_yield_curve_twist(
    zero_df: pd.DataFrame, short_end_shift_bps: float, long_end_shift_bps: float, pivot_tenor_yrs: float
) -> pd.DataFrame:
    """Applies a linear twist to the yield curve around a pivot point."""
    df_shocked = zero_df.copy()
    min_tenor, max_tenor = df_shocked['Tenor'].min(), df_shocked['Tenor'].max()
    shifts = []
    for tenor in df_shocked['Tenor']:
        if tenor <= pivot_tenor_yrs:
            shift = short_end_shift_bps * (pivot_tenor_yrs - tenor) / (pivot_tenor_yrs - min_tenor) if pivot_tenor_yrs > min_tenor else 0
        else:
            shift = long_end_shift_bps * (tenor - pivot_tenor_yrs) / (max_tenor - pivot_tenor_yrs) if max_tenor > pivot_tenor_yrs else 0
        shifts.append(shift)
    df_shocked['ZeroRate'] += np.array(shifts) / 10000.0
    return df_shocked

def apply_market_stress_shock_nn(external_data_day: pd.Series, stress_factor: float) -> pd.Series:
    """Increases VIX and MOVE indices by a given factor (for NN)."""
    shocked_series = external_data_day.copy()
    if 'MOVE_Open' in shocked_series.index: shocked_series['MOVE_Open'] *= stress_factor
    if 'VIX_Open' in shocked_series.index: shocked_series['VIX_Open'] *= stress_factor
    if 'MOVE_VIX_Ratio' in shocked_series.index and shocked_series.get('VIX_Open', 0) > 1e-6:
         shocked_series['MOVE_VIX_Ratio'] = shocked_series['MOVE_Open'] / shocked_series['VIX_Open']
    return shocked_series

def apply_market_stress_shock_trad(vol_cube_df: pd.DataFrame, vol_shock_factor: float) -> pd.DataFrame:
    """Applies a multiplicative shock to the volatilities in the vol cube (for Traditional)."""
    shocked_df = vol_cube_df.copy()
    vol_rows = shocked_df['Type'] == 'Vol'
    # Select only numeric columns to apply the shock
    numeric_cols = shocked_df.select_dtypes(include=np.number).columns
    shocked_df.loc[vol_rows, numeric_cols] *= vol_shock_factor
    return shocked_df


# ==============================================================================
# --- CORE PREDICTION AND ANALYSIS LOGIC ---
# ==============================================================================

def get_nn_prediction(
    nn_artifacts: Dict[str, Any], eval_date: datetime.date, zero_df: pd.DataFrame,
    external_data_for_day: pd.Series, feature_tenors: List[float]
) -> np.ndarray:
    """Generates a single NN parameter prediction for a given market state."""
    ql_eval_date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
    term_structure = create_ql_yield_curve_nn(zero_df, eval_date)
    external_df_for_extraction = pd.DataFrame(external_data_for_day).T
    external_df_for_extraction.index = pd.to_datetime(external_df_for_extraction.index)
    raw_features = extract_raw_features(term_structure, ql_eval_date, external_df_for_extraction, feature_tenors)
    pca_features = apply_pca_to_features(raw_features, nn_artifacts['pca_model'], list(range(len(feature_tenors))))
    scaled_features = nn_artifacts['scaler'].transform(np.array(pca_features).reshape(1, -1))
    predicted_params = nn_artifacts['model'](
        (tf.constant(scaled_features, dtype=tf.float64), nn_artifacts['initial_logits']), training=False
    ).numpy().flatten()
    return predicted_params

def get_traditional_prediction(
    eval_date: datetime.date, zero_df: pd.DataFrame, vol_cube_df: pd.DataFrame, settings: Dict[str, Any]
) -> np.ndarray:
    """Performs a single traditional calibration and returns the parameters."""
    ql_eval_date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
    ql.Settings.instance().evaluationDate = ql_eval_date
    term_structure_handle = create_ql_yield_curve_trad(zero_df, eval_date)
    
    helpers_with_info = prepare_calibration_helpers(
        vol_cube_df, term_structure_handle,
        min_expiry_years=settings.get('min_expiry_years', 0.0),
        min_tenor_years=settings.get('min_tenor_years', 0.0)
    )
    if not helpers_with_info: return np.full(settings['num_a_segments'] + settings['num_sigma_segments'], np.nan)

    included_expiries_yrs = sorted(list(set([parse_tenor_to_years(expiry) for _, expiry, _ in helpers_with_info])))
    a_step_dates = _get_step_dates_from_expiries(ql_eval_date, included_expiries_yrs, settings['num_a_segments'])
    sigma_step_dates = _get_step_dates_from_expiries(ql_eval_date, included_expiries_yrs, settings['num_sigma_segments'])
    unified_step_dates = sorted(list(set(a_step_dates + sigma_step_dates)))
    
    sigma_guesses = settings.get('initial_sigma', [0.01] * settings['num_sigma_segments'])
    sigma_handles = [ql.QuoteHandle(ql.SimpleQuote(s)) for s in sigma_guesses]
    reversion_handles = [ql.QuoteHandle(ql.SimpleQuote(settings.get('initial_a', 0.01)))] * settings['num_a_segments']

    expanded_sigma = _expand_params_to_unified_timeline(sigma_handles, sigma_step_dates, unified_step_dates)
    expanded_reversion = _expand_params_to_unified_timeline(reversion_handles, a_step_dates, unified_step_dates)

    model = ql.Gsr(term_structure_handle, unified_step_dates, expanded_sigma, expanded_reversion, 61.0)
    engine = ql.Gaussian1dSwaptionEngine(model, settings.get('pricing_engine_integration_points', 64), 7.0, True, False)
    
    calibration_helpers = [h for h, _, _ in helpers_with_info]
    for h in calibration_helpers: h.setPricingEngine(engine)
    
    method = ql.LevenbergMarquardt()
    end_criteria = ql.EndCriteria(400, 100, 1e-8, 1e-8, 1e-8)
    
    try:
        if settings['optimize_a']:
            model.calibrate(calibration_helpers, method, end_criteria, ql.PositiveConstraint())
        else:
            model.calibrateVolatilitiesIterative(calibration_helpers, method, end_criteria)
    except Exception as e:
        print(f"  -> WARNING: Traditional calibration failed for scenario. Error: {e}")
        return np.full(settings['num_a_segments'] + settings['num_sigma_segments'], np.nan)

    # --- Extract unique calibrated parameters ---
    def get_unique_calibrated_values(expanded_values: List[float], original_step_dates: List[ql.Date], unified_step_dates: List[ql.Date]) -> List[float]:
        if not original_step_dates: return [expanded_values[0]]
        indices_to_report = [0]
        original_serials = {d.serialNumber() for d in original_step_dates}
        for i, d_unified in enumerate(unified_step_dates):
            if d_unified.serialNumber() in original_serials: indices_to_report.append(i + 1)
        return list(dict.fromkeys([expanded_values[i] for i in sorted(list(set(indices_to_report)))]))

    final_as = get_unique_calibrated_values(list(model.reversion()), a_step_dates, unified_step_dates)
    final_sigmas = get_unique_calibrated_values(list(model.volatility()), sigma_step_dates, unified_step_dates)
    
    return np.array(final_as + final_sigmas)


def plot_parameter_sensitivity_bars_grouped(
    nn_results_df: pd.DataFrame, trad_results_df: pd.DataFrame, output_dir: str
):
    """Generates and saves grouped bar charts comparing NN and Traditional sensitivity."""
    nn_base = nn_results_df.loc['Base_Case']
    trad_base = trad_results_df.loc['Base_Case']
    
    # Calculate percentage change for both models
    nn_pct_change = (nn_results_df.drop('Base_Case') - nn_base).divide(nn_base.abs().replace(0, 1e-9)) * 100
    trad_pct_change = (trad_results_df.drop('Base_Case') - trad_base).divide(trad_base.abs().replace(0, 1e-9)) * 100
    
    scenarios = nn_pct_change.index
    param_names = nn_results_df.columns
    
    print("\n--- Plotting Combined Sensitivity Results ---")
    
    for param in param_names:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 7))

        nn_values = nn_pct_change[param]
        trad_values = trad_pct_change[param]

        x = np.arange(len(scenarios))  # the label locations
        width = 0.35  # the width of the bars

        rects1 = ax.barh(x + width/2, nn_values, width, label='Neural Network', color=CHANGE_PLUS_COLOR)
        rects2 = ax.barh(x - width/2, trad_values, width, label='Traditional', color=CHANGE_MINUS_COLOR)

        ax.set_title(f'Sensitivity of Parameter: {param}', fontsize=16, pad=20)
        ax.set_xlabel('Percentage Change (%) from Base Case', fontsize=12)
        ax.set_yticks(x)
        ax.set_yticklabels(scenarios)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.legend()
        
        ax.axvline(0, color='grey', linewidth=0.8)

        # Attach a text label above each bar in *rects*, displaying its height.
        def autolabel(rects, horiz_align):
            padding = 5
            for rect in rects:
                width = rect.get_width()
                label = f"{width:.2f}%"
                x_pos = width + (padding if width >= 0 else -padding)
                y_pos = rect.get_y() + rect.get_height() / 2
                ax.annotate(label, (x_pos, y_pos),
                            xytext=(0, 0), textcoords="offset points",
                            ha=horiz_align, va='center')

        autolabel(rects1, 'left')
        autolabel(rects2, 'left')
        
        left_limit, right_limit = ax.get_xlim()
        ax.set_xlim(left_limit * 1.25, right_limit * 1.25) # Give more space for labels

        fig.tight_layout()
        plot_path = os.path.join(output_dir, f'sensitivity_comparison_{param}.png')
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"  -> Saved comparison plot for {param} to {plot_path}")

# ==============================================================================
# --- MAIN EXECUTION BLOCK ---
# ==============================================================================
if __name__ == '__main__':
    print("="*80); print(" Combined Sensitivity Analysis (NN vs. Traditional) ".center(80)); print("="*80)
    
    try:
        os.makedirs(FOLDER_SENSITIVITY_RESULTS, exist_ok=True)
        
        # --- 1. Load NN Model ---
        if LATEST_MODEL_DIR:
            model_dir = LATEST_MODEL_DIR
        else:
            model_dirs = sorted([os.path.join(FOLDER_NN_MODELS, d) for d in os.listdir(FOLDER_NN_MODELS)])
            if not model_dirs: raise FileNotFoundError(f"No models found in {FOLDER_NN_MODELS}")
            model_dir = model_dirs[-1]
        
        nn_artifacts = load_nn_artifacts_standalone(model_dir)
        analysis_date = datetime.datetime.strptime(ANALYSIS_DATE_STR, "%d.%m.%Y").date()

        # --- 2. Load All Base Data ---
        print(f"\n--- Loading base data for analysis date: {analysis_date.strftime('%Y-%m-%d')} ---")
        zero_curve_path = os.path.join(FOLDER_ZERO_CURVES, f"{ANALYSIS_DATE_STR}.csv")
        vol_cube_path = os.path.join(FOLDER_VOLATILITY_CUBES, 'xlsx', f"{ANALYSIS_DATE_STR}.xlsx")
        external_data_path = os.path.join(FOLDER_EXTERNAL_DATA, 'external_market_data.csv')
        
        for path in [zero_curve_path, vol_cube_path, external_data_path]:
            if not os.path.exists(path): raise FileNotFoundError(f"Required data file not found: {path}")
            
        base_zero_df = pd.read_csv(zero_curve_path, parse_dates=['Date'])
        base_vol_cube_df = pd.read_excel(vol_cube_path, engine='openpyxl')
        base_vol_cube_df.rename(columns={'Unnamed: 1': 'Type'}, inplace=True)
        base_vol_cube_df['Expiry'] = base_vol_cube_df['Expiry'].ffill()
        external_data_full = pd.read_csv(external_data_path, parse_dates=['Date'], index_col='Date')
        
        if pd.to_datetime(analysis_date) not in external_data_full.index:
             external_data_full = external_data_full.reindex(external_data_full.index.union([pd.to_datetime(analysis_date)])).ffill()
        base_external_data_day = external_data_full.loc[pd.to_datetime(analysis_date)].copy()
        base_external_data_day.name = pd.to_datetime(analysis_date)
        print("All base data loaded successfully.")

        # --- 3. Run Scenarios for Both Models ---
        results = {'NN': {}, 'Traditional': {}}
        print("\n--- Running Scenarios ---")
        
        # A. Run the Base Case (un-shocked)
        print("\n  -> Scenario: Base_Case")
        print("     - Running Neural Network...")
        results['NN']['Base_Case'] = get_nn_prediction(nn_artifacts, analysis_date, base_zero_df, base_external_data_day, FEATURE_TENORS)
        print("     - Running Traditional Calibrator...")
        results['Traditional']['Base_Case'] = get_traditional_prediction(analysis_date, base_zero_df, base_vol_cube_df, TRADITIONAL_CALIBRATION_SETTINGS)
        
        # B. Loop through and run all defined shock scenarios
        for name, config in SCENARIO_SETTINGS.items():
            print(f"\n  -> Scenario: {name}")
            shocked_zero_df, shocked_external_data, shocked_vol_cube = base_zero_df.copy(), base_external_data_day.copy(), base_vol_cube_df.copy()
            
            if config['type'] == 'yield_curve_shift':
                shocked_zero_df = apply_yield_curve_shift(shocked_zero_df, **config['params'])
            elif config['type'] == 'yield_curve_twist':
                shocked_zero_df = apply_yield_curve_twist(shocked_zero_df, **config['params'])
            elif config['type'] == 'market_stress_shock':
                # Apply different shocks for each model type
                shocked_external_data = apply_market_stress_shock_nn(shocked_external_data, **config['params_nn'])
                shocked_vol_cube = apply_market_stress_shock_trad(shocked_vol_cube, **config['params_trad'])

            print("     - Running Neural Network...")
            results['NN'][name] = get_nn_prediction(nn_artifacts, analysis_date, shocked_zero_df, shocked_external_data, FEATURE_TENORS)
            print("     - Running Traditional Calibrator...")
            results['Traditional'][name] = get_traditional_prediction(analysis_date, shocked_zero_df, shocked_vol_cube, TRADITIONAL_CALIBRATION_SETTINGS)

        # --- 4. Process and Display Results ---
        num_a = MODEL_PARAMETERS['num_a_segments'] if MODEL_PARAMETERS['optimize_a'] else 0
        param_names = [f'a_{i+1}' for i in range(num_a)] + [f'sigma_{i+1}' for i in range(MODEL_PARAMETERS['num_sigma_segments'])]
        
        nn_results_df = pd.DataFrame.from_dict(results['NN'], orient='index', columns=param_names)
        trad_results_df = pd.DataFrame.from_dict(results['Traditional'], orient='index', columns=param_names)

        print("\n\n" + "="*80); print(" COMBINED SENSITIVITY ANALYSIS RESULTS ".center(80)); print("="*80)
        
        # Combine into a single DataFrame with multi-level columns
        combined_df = pd.concat([nn_results_df, trad_results_df], axis=1, keys=['Neural Network', 'Traditional'])

        # Calculate percentage changes for display
        nn_pct = (nn_results_df.drop('Base_Case') - nn_results_df.loc['Base_Case']).divide(nn_results_df.loc['Base_Case'].abs().replace(0,1e-9)) * 100
        trad_pct = (trad_results_df.drop('Base_Case') - trad_results_df.loc['Base_Case']).divide(trad_results_df.loc['Base_Case'].abs().replace(0,1e-9)) * 100
        pct_change_df = pd.concat([nn_pct, trad_pct], axis=1, keys=['Neural Network', 'Traditional'])

        # Build the final display table
        display_frames = []
        for model in ['Neural Network', 'Traditional']:
            for param in param_names:
                abs_col = combined_df[(model, param)].rename('Abs. Value')
                pct_col = pct_change_df[(model, param)].rename('% Change').reindex(abs_col.index)
                
                param_df = pd.concat([abs_col, pct_col], axis=1)
                param_df.columns = pd.MultiIndex.from_product([[model], [param], ['Abs. Value', '% Change']])
                display_frames.append(param_df)

        final_display_df = pd.concat(display_frames, axis=1).sort_index(axis=1)

        # Apply formatting for clean console output
        formatted_df = final_display_df.copy()
        for model in ['Neural Network', 'Traditional']:
             for param in param_names:
                formatted_df[(model, param, 'Abs. Value')] = formatted_df[(model, param, 'Abs. Value')].map('{:.6f}'.format)
                formatted_df[(model, param, '% Change')] = formatted_df[(model, param, '% Change')].map(lambda x: f"{x:+.2f}%" if pd.notna(x) else "-")
        
        print(formatted_df.to_string())
        print("="*80)

        # Save results to a single CSV
        results_csv_path = os.path.join(FOLDER_SENSITIVITY_RESULTS, 'sensitivity_results_comparison.csv')
        final_display_df.to_csv(results_csv_path)
        print(f"\nCombined numerical results saved to: {results_csv_path}")
        
        # --- 5. Visualize Results ---
        plot_parameter_sensitivity_bars_grouped(nn_results_df, trad_results_df, FOLDER_SENSITIVITY_RESULTS)
        
        print("\n--- SCRIPT FINISHED SUCCESSFULLY ---")

    except FileNotFoundError as e:
        print(f"\nERROR: A required data file could not be found.")
        print(e)
        print("Please check that data exists for the specified ANALYSIS_DATE_STR and all paths are correct.")
    except Exception as e:
        import traceback
        print(f"\nAn unexpected error occurred: {e}")
        traceback.print_exc()