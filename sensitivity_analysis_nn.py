# sensitivity_analysis_nn.py

"""
Neural Network Sensitivity Analysis Script

Objective:
This script performs a sensitivity analysis on a trained Hull-White parameter
prediction Neural Network. It is designed to be a standalone diagnostic tool
to understand and validate the model's economic behavior.

Methodology:
1.  A single, pre-trained Neural Network model artifact is loaded.
2.  Market data for a specific, representative analysis date is loaded.
3.  A "Base Case" prediction is generated using the original market data.
4.  A series of pre-defined, economically meaningful shocks are applied to
    the input data (e.g., yield curve shifts, market stress).
5.  The same, frozen NN model is used to generate new parameter predictions
    for each shocked scenario.
6.  The results are compiled into a summary table and visualized to show
    the change in predicted parameters relative to the Base Case.

This script intentionally does not use the traditional calibrator; its sole
focus is on analyzing the behavior of the neural network predictor.
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
from typing import Dict, Any, List
import QuantLib as ql

# --- Ensure the script can find the other project modules ---
try:
    from neural_network_calibration import create_ql_yield_curve, ResidualParameterModel, extract_raw_features, apply_pca_to_features
except ImportError:
    print("ERROR: Could not import from 'neural_network_calibration'.")
    print("Ensure this script is in the same directory or the project is in your PYTHONPATH.")
    sys.exit(1)

# ==============================================================================
# --- SCRIPT CONFIGURATION ---
# ==============================================================================

# --- 1. Model and Data Selection ---
# Set to a specific path or None to auto-detect the latest model
LATEST_MODEL_DIR = 'results/neural_network/models/model_20251021_004632'  # Example path; set to None to auto-detect
# Specify the single date for which the analysis will be run.
ANALYSIS_DATE_STR = "31.08.2025"

# --- 2. Model Structure (Must match the trained model) ---
# This is crucial for interpreting the model's output vector correctly.
MODEL_PARAMETERS = {
    "num_a_segments": 1,
    "num_sigma_segments": 7,
    "optimize_a": True, # Must match the 'optimize_a' setting during training
}
# The tenors used for the rate features during training
FEATURE_TENORS = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]


# --- 3. Scenario Definitions ---
# Define the shocks to be applied.
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
    "Market_Stress_25_Percent": {
        "type": "market_stress_shock",
        "params": {"stress_factor": 1.25} # 1.25 means a 25% increase
    },
}

CHANGE_PLUS_COLOR = cm.get_cmap('coolwarm', 10)(1)  # A nice blue
CHANGE_MINUS_COLOR = cm.get_cmap('coolwarm', 10)(9)  # A nice red

# --- 4. Paths and Directories ---
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FOLDER_ZERO_CURVES: str = os.path.join(DATA_DIR, 'EUR ZERO CURVE')
FOLDER_EXTERNAL_DATA: str = os.path.join(DATA_DIR, 'EXTERNAL')
FOLDER_NN_MODELS: str = os.path.join(RESULTS_DIR, 'neural_network/models')
FOLDER_SENSITIVITY_RESULTS: str = os.path.join(RESULTS_DIR, 'sensitivity_analysis')

# ==============================================================================
# --- HELPER FUNCTIONS ---
# ==============================================================================

def load_nn_artifacts_standalone(model_dir: str) -> Dict[str, Any]:
    """
    Loads all necessary artifacts for the trained NN model.
    This version does NOT depend on 'feature_names.json'.
    """
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
    min_tenor = df_shocked['Tenor'].min()
    max_tenor = df_shocked['Tenor'].max()
    shifts = []
    for tenor in df_shocked['Tenor']:
        if tenor <= pivot_tenor_yrs:
            shift = short_end_shift_bps * (pivot_tenor_yrs - tenor) / (pivot_tenor_yrs - min_tenor) if pivot_tenor_yrs > min_tenor else 0
        else:
            shift = long_end_shift_bps * (tenor - pivot_tenor_yrs) / (max_tenor - pivot_tenor_yrs) if max_tenor > pivot_tenor_yrs else 0
        shifts.append(shift)
    df_shocked['ZeroRate'] += np.array(shifts) / 10000.0
    return df_shocked

def apply_market_stress_shock(external_data_day: pd.Series, stress_factor: float) -> pd.Series:
    """Increases VIX and MOVE indices by a given factor."""
    shocked_series = external_data_day.copy()
    if 'MOVE_Open' in shocked_series.index:
        shocked_series['MOVE_Open'] *= stress_factor
    if 'VIX_Open' in shocked_series.index:
        shocked_series['VIX_Open'] *= stress_factor
    if 'MOVE_VIX_Ratio' in shocked_series.index and shocked_series.get('VIX_Open', 0) > 1e-6:
         shocked_series['MOVE_VIX_Ratio'] = shocked_series['MOVE_Open'] / shocked_series['VIX_Open']
    return shocked_series

# ==============================================================================
# --- CORE PREDICTION AND ANALYSIS LOGIC ---
# ==============================================================================

def get_nn_prediction(
    nn_artifacts: Dict[str, Any], eval_date: datetime.date, zero_df: pd.DataFrame,
    external_data_for_day: pd.Series, feature_tenors: List[float]
) -> np.ndarray:
    """Generates a single NN parameter prediction for a given market state."""
    ql_eval_date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
    term_structure = create_ql_yield_curve(zero_df, eval_date)
    
    # The extraction function requires the external data in a DataFrame format
    external_df_for_extraction = pd.DataFrame(external_data_for_day).T
    external_df_for_extraction.index = pd.to_datetime(external_df_for_extraction.index)
    
    raw_features = extract_raw_features(term_structure, ql_eval_date, external_df_for_extraction, feature_tenors)
    
    rate_indices = list(range(len(feature_tenors)))
    pca_features = apply_pca_to_features(raw_features, nn_artifacts['pca_model'], rate_indices)
    
    scaled_features = nn_artifacts['scaler'].transform(np.array(pca_features).reshape(1, -1))
    
    predicted_params = nn_artifacts['model'](
        (tf.constant(scaled_features, dtype=tf.float64), nn_artifacts['initial_logits']), training=False
    ).numpy().flatten()
    
    return predicted_params

def plot_parameter_sensitivity_bars(results_df: pd.DataFrame, output_dir: str):
    """Generates and saves bar charts visualizing the percentage change."""
    base_case = results_df.loc['Base_Case']
    scenarios = results_df.drop('Base_Case')
    
    # Use .abs() in the denominator to avoid issues with negative base case values
    pct_change_df = (scenarios - base_case).divide(base_case.abs().replace(0, 1e-9)) * 100

    # --- START: MODIFIED LOGIC FOR PLOT ORDER ---
    # Define the desired static order for the plot. Note: "YC_Shift_Down_50bps" is corrected from your request.
    scenario_order = list(reversed([
        'YC_Shift_Up_50bps',
        'YC_Shift_Down_50bps',
        'YC_Twist_Steepen',
        'Market_Stress_25_Percent'
    ]))
    # --- END: MODIFIED LOGIC FOR PLOT ORDER ---
    
    print("\n--- Plotting Sensitivity Results ---")
    for param in pct_change_df.columns:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Reorder the data according to the specified list instead of sorting by value
        ordered_data = pct_change_df[param].reindex(scenario_order)
        
        colors = [CHANGE_MINUS_COLOR if x < 0 else CHANGE_PLUS_COLOR for x in ordered_data]
        
        ordered_data.plot(kind='barh', ax=ax, color=colors)
        
        ax.set_title(f'Sensitivity of Parameter: {param}', fontsize=16, pad=20)
        ax.set_xlabel('Percentage Change (%) from Base Case', fontsize=12)
        ax.set_ylabel('Scenario', fontsize=12)
        
        # Manually add labels using ax.text() for compatibility with older Matplotlib versions.
        # This gives full control over text position and alignment.
        positive_padding = 0.3
        negative_padding = -0.3

        for patch in ax.patches:
            value = patch.get_width()
            y_pos = patch.get_y() + patch.get_height() / 2
            label = f"{value:.2f}%"

            if value >= 0:
                # For POSITIVE bars, place text to the right of the bar
                ax.text(value + positive_padding, y_pos, label, ha='left', va='center')
            else:
                # For NEGATIVE bars, place text to the left (inside the bar)
                ax.text(value + negative_padding, y_pos, label, ha='right', va='center', color='white', fontsize=9)

        # Dynamically adjust x-axis limits to give labels more space
        left_limit, right_limit = ax.get_xlim()
        ax.set_xlim(left_limit * 1.15, right_limit * 1.15)
        
        fig.tight_layout()
        plot_path = os.path.join(output_dir, f'sensitivity_{param}.png')
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"  -> Saved plot for {param} to {plot_path}")

# ==============================================================================
# --- MAIN EXECUTION BLOCK ---
# ==============================================================================
if __name__ == '__main__':
    print("="*80); print(" Neural Network Sensitivity Analysis ".center(80)); print("="*80)
    
    try:
        os.makedirs(FOLDER_SENSITIVITY_RESULTS, exist_ok=True)
        
        if LATEST_MODEL_DIR:
            model_dir = LATEST_MODEL_DIR
        else:
            model_dirs = [os.path.join(FOLDER_NN_MODELS, d) for d in os.listdir(FOLDER_NN_MODELS) if os.path.isdir(os.path.join(FOLDER_NN_MODELS, d))]
            if not model_dirs:
                raise FileNotFoundError(f"No trained NN models found in {FOLDER_NN_MODELS}")
            model_dir = max(model_dirs, key=os.path.getctime)
        
        nn_artifacts = load_nn_artifacts_standalone(model_dir)
        analysis_date = datetime.datetime.strptime(ANALYSIS_DATE_STR, "%d.%m.%Y").date()

        print(f"\n--- Loading base data for analysis date: {analysis_date.strftime('%Y-%m-%d')} ---")
        zero_curve_path = os.path.join(FOLDER_ZERO_CURVES, f"{ANALYSIS_DATE_STR}.csv")
        external_data_path = os.path.join(FOLDER_EXTERNAL_DATA, 'external_market_data.csv')
        
        if not os.path.exists(zero_curve_path):
            raise FileNotFoundError(f"Required zero curve file not found at: {zero_curve_path}")
        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"Required external data file not found at: {external_data_path}")
            
        base_zero_df = pd.read_csv(zero_curve_path, parse_dates=['Date'])
        external_data_full = pd.read_csv(external_data_path, parse_dates=['Date'], index_col='Date')
        
        # Handle cases where the exact date is not in the external data index
        if pd.to_datetime(analysis_date) not in external_data_full.index:
             # Use ffill to get the last available data if the date is a holiday/weekend
             external_data_full = external_data_full.reindex(external_data_full.index.union([pd.to_datetime(analysis_date)])).ffill()
        
        base_external_data_day = external_data_full.loc[pd.to_datetime(analysis_date)].copy()
        base_external_data_day.name = pd.to_datetime(analysis_date)
        print("Base data loaded successfully.")

        results = {}
        print("\n--- Running Scenarios ---")
        
        # A. Run the Base Case (un-shocked)
        print("  -> Scenario: Base_Case")
        results['Base_Case'] = get_nn_prediction(nn_artifacts, analysis_date, base_zero_df, base_external_data_day, FEATURE_TENORS)
        
        # B. Loop through and run all defined shock scenarios
        for name, config in SCENARIO_SETTINGS.items():
            print(f"  -> Scenario: {name}")
            shocked_zero_df, shocked_external_data = base_zero_df.copy(), base_external_data_day.copy()
            
            if config['type'] == 'yield_curve_shift':
                shocked_zero_df = apply_yield_curve_shift(shocked_zero_df, **config['params'])
            elif config['type'] == 'yield_curve_twist':
                shocked_zero_df = apply_yield_curve_twist(shocked_zero_df, **config['params'])
            elif config['type'] == 'market_stress_shock':
                shocked_external_data = apply_market_stress_shock(shocked_external_data, **config['params'])
            
            results[name] = get_nn_prediction(nn_artifacts, analysis_date, shocked_zero_df, shocked_external_data, FEATURE_TENORS)
            
        # Determine parameter names based on the configuration
        num_a = MODEL_PARAMETERS['num_a_segments'] if MODEL_PARAMETERS['optimize_a'] else 0
        num_sigma = MODEL_PARAMETERS['num_sigma_segments']
        param_names = [f'a_{i+1}' for i in range(num_a)] + [f'sigma_{i+1}' for i in range(num_sigma)]
        results_df = pd.DataFrame.from_dict(results, orient='index', columns=param_names)
        
        # --- MODIFIED: Create a detailed display table with relative changes ---
        print("\n\n" + "="*80)
        print(" SENSITIVITY ANALYSIS RESULTS ".center(80))
        print("="*80)
        
        # Separate the base case for calculations
        base_case = results_df.loc['Base_Case']
        scenarios_df = results_df.drop('Base_Case')
        
        # Calculate percentage change, avoiding division by zero
        epsilon = 1e-9
        pct_change_df = (scenarios_df - base_case).divide(base_case.abs() + epsilon) * 100
        
        # Build the final display DataFrame with multi-level columns
        display_frames = []
        for param in param_names:
            # Combine absolute value and percentage change for each parameter
            param_abs = results_df[[param]].rename(columns={param: 'Abs. Value'})
            param_pct = pct_change_df[[param]].rename(columns={param: '% Change'})
            
            # Re-add the base case to the percentage change frame so it aligns
            param_pct = param_pct.reindex(results_df.index) 
            
            combined = pd.concat([param_abs, param_pct], axis=1)
            combined.columns = pd.MultiIndex.from_product([[param], ['Abs. Value', '% Change']])
            display_frames.append(combined)

        # Concatenate all parameter frames horizontally
        final_display_df = pd.concat(display_frames, axis=1)
        
        # Apply formatting for clean console output
        formatted_df = final_display_df.copy()
        for param in param_names:
            formatted_df[(param, 'Abs. Value')] = formatted_df[(param, 'Abs. Value')].map('{:.6f}'.format)
            formatted_df[(param, '% Change')] = formatted_df[(param, '% Change')].map(lambda x: f"{x:+.2f}%" if pd.notna(x) else "-")

        print(formatted_df.to_string())
        print("="*80)
        
        # Save the raw, unformatted numerical results to CSV
        results_csv_path = os.path.join(FOLDER_SENSITIVITY_RESULTS, 'sensitivity_results_nn.csv')
        formatted_df.to_csv(results_csv_path)
        print(f"\nRaw numerical results saved to: {results_csv_path}")
        
        # Visualize results
        plot_parameter_sensitivity_bars(results_df, FOLDER_SENSITIVITY_RESULTS)
        
        print("\n--- SCRIPT FINISHED SUCCESSFULLY ---")

    except FileNotFoundError as e:
        print(f"\nERROR: A required data file could not be found.")
        print(e)
        print("Please check that the data exists for the specified ANALYSIS_DATE_STR.")
    except (ValueError, KeyError) as e:
        print(f"\nERROR: {e}")
        print("This could be due to a mismatch in configuration or missing data for the selected date.")
    except Exception as e:
        import traceback
        print(f"\nAn unexpected error occurred: {e}")
        traceback.print_exc()