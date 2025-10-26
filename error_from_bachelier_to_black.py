"""
This script extends a financial model calibration task. It reads a time series
of daily model errors (in Normal, or Bachelier, volatility terms) from a CSV
file. For each day, it converts these errors into their equivalent, vega-weighted
Black (lognormal) volatility terms.

The script processes each row of the input CSV, which represents a single day's
model performance. For each day, it:
1. Loads the corresponding historical financial data:
   - A EUR zero-coupon yield curve.
   - A swaption volatility cube.
2. Parses all swaptions from the cube for that day.
3. Calculates their respective vegas (sensitivity to volatility).
4. For each model error (for all defined strategies), it computes a portfolio-wide,
   vega-weighted average Black volatility. This provides a representative
   error figure in Black's model terms for that specific day.

The final output is a new CSV file containing the original data plus new
columns with the calculated Black volatilities for each strategy.

Key Libraries:
- pandas: For data manipulation and file I/O.
- QuantLib: For financial instrument pricing and yield curve construction.
- numpy: For numerical operations.
- tqdm: For displaying progress bars during processing.
"""
import datetime
import os
import sys
import pandas as pd
import numpy as np
import QuantLib as ql
from typing import List, Tuple, Optional, Dict
import random
from tqdm import tqdm

# -------------------- REPRODUCIBILITY SEED --------------------
# Set a seed for reproducibility of any random processes.
SEED: int = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# -------------------- CONFIGURATION --------------------
# Define file paths for input data.
FOLDER_ZERO_CURVES: str = r'data/EUR ZERO CURVE'
FOLDER_VOLATILITY_CUBES: str = r'data/EUR BVOL CUBE'

# Define a standard notional for vega calculation.
ASSUMED_NOTIONAL: float = 1.0

# Define input and output file paths for the time series data.
INPUT_CSV_PATH: str = r'results/comparison/daily_summary_results.csv'
OUTPUT_CSV_PATH: str = r'results/comparison/daily_summary_results_black.csv'

# --- NEW: Define all strategies to be processed ---
STRATEGIES = {
    'NN': {'input_col': 'RMSE_NN', 'output_prefix': 'BlackVol_NN'},
    'LM_Static': {'input_col': 'RMSE_LM_Static', 'output_prefix': 'BlackVol_LM_Static'},
    'LM_Pure_Rolling': {'input_col': 'RMSE_LM_Pure_Rolling', 'output_prefix': 'BlackVol_LM_Pure_Rolling'},
    'LM_Adaptive_Anchor': {'input_col': 'RMSE_LM_Adaptive_Anchor', 'output_prefix': 'BlackVol_LM_Adaptive_Anchor'}
}

# -------------------- DATA LOADING & HELPERS --------------------

def load_volatility_cube(file_path: str) -> pd.DataFrame:
    """
    Reads a volatility cube Excel file into a Pandas DataFrame.
    """
    df: pd.DataFrame = pd.read_excel(file_path, engine='openpyxl')
    df.rename(columns={df.columns[1]: 'Type'}, inplace=True)
    for col in df.columns:
        if 'Unnamed' in str(col): df.drop(col, axis=1, inplace=True)
    df['Expiry'] = df['Expiry'].ffill()
    return df

def parse_tenor(tenor_str: str) -> ql.Period:
    """
    Parses a string representing a tenor (e.g., '1Yr', '6Mo') into a QuantLib Period object.
    """
    tenor_str = str(tenor_str).strip().upper()
    if 'YR' in tenor_str: return ql.Period(int(tenor_str.replace('YR', '')), ql.Years)
    if 'MO' in tenor_str: return ql.Period(int(tenor_str.replace('MO', '')), ql.Months)
    raise ValueError(f"Could not parse tenor string: {tenor_str}")

def create_ql_yield_curve(
    zero_curve_df: pd.DataFrame, eval_date: datetime.date
) -> ql.RelinkableYieldTermStructureHandle:
    """
    Creates a QuantLib yield curve from a pandas DataFrame.
    """
    ql_eval_date: ql.Date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
    ql.Settings.instance().evaluationDate = ql_eval_date
    
    dates: list[ql.Date] = [ql_eval_date] + [ql.Date(d.day, d.month, d.year) for d in pd.to_datetime(zero_curve_df['Date'])]
    rates: list[float] = [zero_curve_df['ZeroRate'].iloc[0]] + zero_curve_df['ZeroRate'].tolist()
    
    term_structure: ql.ZeroCurve = ql.ZeroCurve(dates, rates, ql.Actual365Fixed(), ql.TARGET(), ql.Linear(), ql.Continuous, ql.Annual)
    term_structure.enableExtrapolation()
    
    handle: ql.RelinkableYieldTermStructureHandle = ql.RelinkableYieldTermStructureHandle()
    handle.linkTo(term_structure)
    return handle

# -------------------- CALCULATION LOGIC --------------------
def get_swaption_details_from_cube(vol_cube_df: pd.DataFrame) -> List[Dict]:
    """
    Extracts relevant information from a volatility cube DataFrame.
    """
    swaption_list = []
    df = vol_cube_df.set_index(['Expiry', 'Type'])
    for expiry in df.index.get_level_values('Expiry').unique():
        try:
            vols = df.loc[(expiry, 'Vol')]
            strikes = df.loc[(expiry, 'Strike')]
            for tenor_str, normal_vol_bps in vols.items():
                strike_rate = strikes[tenor_str]
                if pd.notna(normal_vol_bps) and normal_vol_bps > 0 and pd.notna(strike_rate):
                    swaption_list.append({
                        'expiry_str': expiry,
                        'tenor_str': tenor_str,
                        'normal_vol_bps': normal_vol_bps,
                        'forward_rate': strike_rate / 100.0
                    })
        except KeyError:
            continue
    return swaption_list

def calculate_vega_and_black_error(
    eval_date: datetime.date,
    swaption: Dict,
    yield_curve_handle: ql.RelinkableYieldTermStructureHandle,
    model_error_normal_bps: float
) -> Optional[Tuple[float, float]]:
    """
    Calculates the vega and Black volatility errors for a given swaption.
    """
    try:
        forward_rate = swaption['forward_rate']
        if forward_rate <= 0: return None

        ql_eval_date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
        ql.Settings.instance().evaluationDate = ql_eval_date
        
        expiry_period = parse_tenor(swaption['expiry_str'])
        tenor_period = parse_tenor(swaption['tenor_str'])
        
        market_normal_vol_decimal = swaption['normal_vol_bps'] / 10000.0
        model_error_normal_decimal = model_error_normal_bps / 10000.0
        
        individual_black_vol_error = model_error_normal_decimal / forward_rate

        swap_index = ql.Euribor6M(yield_curve_handle)
        
        underlying_swap = ql.MakeVanillaSwap(swapTenor=tenor_period,
                                             iborIndex=swap_index,
                                             fixedRate=forward_rate,
                                             forwardStart=expiry_period)

        exercise_date = ql.TARGET().advance(ql_eval_date, expiry_period)
        exercise = ql.EuropeanExercise(exercise_date)
        swaption_obj = ql.Swaption(underlying_swap, exercise)
        
        vol_handle = ql.QuoteHandle(ql.SimpleQuote(market_normal_vol_decimal))
        pricing_engine = ql.BachelierSwaptionEngine(yield_curve_handle, vol_handle)
        swaption_obj.setPricingEngine(pricing_engine)
        
        normal_vega = (swaption_obj.vega() / 100.0) * ASSUMED_NOTIONAL
        return (individual_black_vol_error, normal_vega)

    except Exception:
        return None

def convert_normal_to_black_for_date(
    eval_date: datetime.date,
    normal_error_bps: float,
    zero_curve_folder: str,
    vol_cube_folder: str
) -> Optional[Tuple[float, float]]:
    """
    Calculates the vega-weighted and simple average Black volatility errors.
    """
    date_str = eval_date.strftime('%d.%m.%Y')
    zero_path = os.path.join(zero_curve_folder, f"{date_str}.csv")
    vol_path = os.path.join(vol_cube_folder, 'xlsx', f"{date_str}.xlsx")

    if not os.path.exists(zero_path) or not os.path.exists(vol_path):
        return None

    try:
        zero_curve_df = pd.read_csv(zero_path, parse_dates=['Date'])
        vol_cube_df = load_volatility_cube(vol_path)
        yield_curve_handle = create_ql_yield_curve(zero_curve_df, eval_date)
        
        swaptions_to_process = get_swaption_details_from_cube(vol_cube_df)
        
        total_weighted_error, total_simple_error, total_vega = 0.0, 0.0, 0.0
        valid_swaption_count = 0
        
        for swaption in swaptions_to_process:
            result = calculate_vega_and_black_error(eval_date, swaption, yield_curve_handle, normal_error_bps)
            if result:
                individual_black_vol_error, normal_vega = result
                total_simple_error += individual_black_vol_error
                valid_swaption_count += 1
                total_weighted_error += individual_black_vol_error * normal_vega
                total_vega += normal_vega

        if total_vega == 0 or valid_swaption_count == 0:
            return None

        vega_weighted_error = total_weighted_error / total_vega
        simple_average_error = total_simple_error / valid_swaption_count
        
        return vega_weighted_error, simple_average_error

    except Exception as e:
        print(f"Warning: An unexpected error occurred while processing date {eval_date}: {e}")
        return None

# -------------------- MAIN EXECUTION BLOCK --------------------
if __name__ == "__main__":
    print("--- Starting: Time Series Model Error Conversion ---")
    
    # --- 1. Load Input Data ---
    try:
        print(f"Reading input data from: {INPUT_CSV_PATH}")
        df = pd.read_csv(INPUT_CSV_PATH)
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        print(f"Successfully loaded {len(df)} daily records.")
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Input file not found at '{INPUT_CSV_PATH}'.")
        sys.exit(1)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to read or parse input CSV. Details: {e}")
        sys.exit(1)

    # --- 2. Process Each Date and Strategy in the DataFrame ---
    results_data = {key: {'weighted': [], 'simple': []} for key in STRATEGIES}
    
    print("\n--- Processing each date to convert Normal volatility errors to Black ---")
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Converting Errors"):
        eval_date = row['Date']
        
        # Loop through each defined strategy to perform the conversion
        for key, config in STRATEGIES.items():
            input_col = config['input_col']
            
            if input_col in row and pd.notna(row[input_col]):
                normal_error_bps = row[input_col]
                conversion_results = convert_normal_to_black_for_date(
                    eval_date, normal_error_bps, FOLDER_ZERO_CURVES, FOLDER_VOLATILITY_CUBES
                )
                
                if conversion_results:
                    results_data[key]['weighted'].append(conversion_results[0])
                    results_data[key]['simple'].append(conversion_results[1])
                else:
                    results_data[key]['weighted'].append(None)
                    results_data[key]['simple'].append(None)
            else:
                # Append None if the input RMSE is missing for that day
                results_data[key]['weighted'].append(None)
                results_data[key]['simple'].append(None)

    # --- 3. Store Results and Save Output ---
    print("\n--- Aggregating results and saving output file ---")
    
    # Dynamically create new columns for each strategy
    for key, config in STRATEGIES.items():
        prefix = config['output_prefix']
        
        # The results are multiplied by 100 to express them in percentage points (%).
        df[f'{prefix}_Vega_Weighted'] = [vol * 100 if vol is not None else np.nan for vol in results_data[key]['weighted']]
        df[f'{prefix}_Simple_Avg'] = [vol * 100 if vol is not None else np.nan for vol in results_data[key]['simple']]

    # Perform basic error handling and summary.
    # Check the status of the first strategy as a proxy for all
    processed_count = df[f"{STRATEGIES['NN']['output_prefix']}_Vega_Weighted"].notna().sum()
    if processed_count == 0:
        print("\nCRITICAL WARNING: No dates could be processed for the primary strategy (NN).")
    else:
        print(f"\nSuccessfully processed {processed_count} out of {len(df)} dates.")
        
    # Save the updated DataFrame to a new CSV file.
    try:
        output_dir = os.path.dirname(OUTPUT_CSV_PATH)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"\nResults successfully saved to: {OUTPUT_CSV_PATH}")
    except Exception as e:
        print(f"\nCRITICAL ERROR: Failed to save the output file. Details: {e}")

    print("\n--- Script finished. ---")