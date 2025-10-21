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
4. For each model error (NN and LM), it computes a portfolio-wide,
   vega-weighted average Black volatility. This provides a representative
   error figure in Black's model terms for that specific day.

The final output is a new CSV file containing the original data plus two new
columns with the calculated Black volatilities.

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

# -------------------- DATA LOADING & HELPERS --------------------

def load_volatility_cube(file_path: str) -> pd.DataFrame:
    """Loads a volatility cube from an Excel file and formats it."""
    df: pd.DataFrame = pd.read_excel(file_path, engine='openpyxl')
    # Clean up dataframe by renaming columns, dropping empty ones, and forward-filling expiries.
    df.rename(columns={df.columns[1]: 'Type'}, inplace=True)
    for col in df.columns:
        if 'Unnamed' in str(col): df.drop(col, axis=1, inplace=True)
    df['Expiry'] = df['Expiry'].ffill()
    return df

def parse_tenor(tenor_str: str) -> ql.Period:
    """Converts a tenor string (e.g., '10YR', '6MO') into a QuantLib Period object."""
    tenor_str = str(tenor_str).strip().upper()
    if 'YR' in tenor_str: return ql.Period(int(tenor_str.replace('YR', '')), ql.Years)
    if 'MO' in tenor_str: return ql.Period(int(tenor_str.replace('MO', '')), ql.Months)
    raise ValueError(f"Could not parse tenor string: {tenor_str}")

def create_ql_yield_curve(
    zero_curve_df: pd.DataFrame, eval_date: datetime.date
) -> ql.RelinkableYieldTermStructureHandle:
    """Creates a QuantLib yield curve object from a pandas DataFrame."""
    ql_eval_date: ql.Date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
    ql.Settings.instance().evaluationDate = ql_eval_date
    
    dates: list[ql.Date] = [ql_eval_date] + [ql.Date(d.day, d.month, d.year) for d in pd.to_datetime(zero_curve_df['Date'])]
    rates: list[float] = [zero_curve_df['ZeroRate'].iloc[0]] + zero_curve_df['ZeroRate'].tolist()
    
    # Construct the zero curve with linear interpolation.
    term_structure: ql.ZeroCurve = ql.ZeroCurve(dates, rates, ql.Actual365Fixed(), ql.TARGET(), ql.Linear(), ql.Continuous, ql.Annual)
    term_structure.enableExtrapolation()
    
    # Use a RelinkableHandle to allow the curve to be updated easily if needed elsewhere.
    handle: ql.RelinkableYieldTermStructureHandle = ql.RelinkableYieldTermStructureHandle()
    handle.linkTo(term_structure)
    return handle

# -------------------- CALCULATION LOGIC --------------------

def get_swaption_details_from_cube(vol_cube_df: pd.DataFrame) -> List[Dict]:
    """Extracts individual swaption data (expiry, tenor, vol, strike) from a formatted cube."""
    swaption_list = []
    df = vol_cube_df.set_index(['Expiry', 'Type'])
    for expiry in df.index.get_level_values('Expiry').unique():
        try:
            vols = df.loc[(expiry, 'Vol')]
            strikes = df.loc[(expiry, 'Strike')]
            for tenor_str, normal_vol_bps in vols.items():
                strike_rate = strikes[tenor_str]
                # Only include swaptions with valid, positive volatility and strike data.
                if pd.notna(normal_vol_bps) and normal_vol_bps > 0 and pd.notna(strike_rate):
                    swaption_list.append({
                        'expiry_str': expiry,
                        'tenor_str': tenor_str,
                        'normal_vol_bps': normal_vol_bps,
                        'forward_rate': strike_rate / 100.0
                    })
        except KeyError:
            # This handles cases where an expiry might not have both 'Vol' and 'Strike' rows.
            continue
    return swaption_list

def calculate_vega_and_black_error(
    eval_date: datetime.date,
    swaption: Dict,
    yield_curve_handle: ql.RelinkableYieldTermStructureHandle,
    model_error_normal_bps: float
) -> Optional[Tuple[float, float]]:
    """Calculates the Normal vega and the equivalent Black volatility error for a single swaption."""
    try:
        forward_rate = swaption['forward_rate']
        # Black's model is not well-defined for non-positive forward rates.
        if forward_rate <= 0:
            return None

        ql_eval_date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
        ql.Settings.instance().evaluationDate = ql_eval_date
        
        expiry_period = parse_tenor(swaption['expiry_str'])
        tenor_period = parse_tenor(swaption['tenor_str'])
        
        market_normal_vol_decimal = swaption['normal_vol_bps'] / 10000.0
        model_error_normal_decimal = model_error_normal_bps / 10000.0
        
        # Approximate conversion from Normal vol to Black vol for an ATM option: Black Vol ≈ Normal Vol / Forward Rate
        individual_black_vol_error = model_error_normal_decimal / forward_rate

        swap_index = ql.Euribor6M(yield_curve_handle)
        
        # Create the underlying swap for the swaption.
        underlying_swap = ql.MakeVanillaSwap(swapTenor=tenor_period,
                                             iborIndex=swap_index,
                                             fixedRate=forward_rate,
                                             forwardStart=expiry_period)

        exercise_date = ql.TARGET().advance(ql_eval_date, expiry_period)
        exercise = ql.EuropeanExercise(exercise_date)
        swaption_obj = ql.Swaption(underlying_swap, exercise)
        
        # Set up the pricing engine using the Bachelier (Normal) model.
        vol_handle = ql.QuoteHandle(ql.SimpleQuote(market_normal_vol_decimal))
        pricing_engine = ql.BachelierSwaptionEngine(yield_curve_handle, vol_handle)
        swaption_obj.setPricingEngine(pricing_engine)
        
        # Calculate vega (per 1% change in vol) and scale by notional.
        normal_vega = (swaption_obj.vega() / 100.0) * ASSUMED_NOTIONAL
        return (individual_black_vol_error, normal_vega)

    except Exception:
        # Suppress verbose error printing for individual swaptions during batch processing.
        return None

def convert_normal_to_black_for_date(
    eval_date: datetime.date,
    normal_error_bps: float,
    zero_curve_folder: str,
    vol_cube_folder: str
) -> Optional[float]:
    """
    For a single date and a given normal volatility error, calculates the
    vega-weighted equivalent Black volatility.
    """
    # Construct file paths for the given date
    date_str = eval_date.strftime('%d.%m.%Y')
    zero_path = os.path.join(zero_curve_folder, f"{date_str}.csv")
    vol_path = os.path.join(vol_cube_folder, 'xlsx', f"{date_str}.xlsx")

    if not os.path.exists(zero_path) or not os.path.exists(vol_path):
        return None  # Return None if data for this date is missing

    try:
        # Load and build financial objects for the given date.
        zero_curve_df = pd.read_csv(zero_path, parse_dates=['Date'])
        vol_cube_df = load_volatility_cube(vol_path)
        yield_curve_handle = create_ql_yield_curve(zero_curve_df, eval_date)
        
        swaptions_to_process = get_swaption_details_from_cube(vol_cube_df)
        
        total_weighted_error = 0.0
        total_vega = 0.0
        
        for swaption in swaptions_to_process:
            result = calculate_vega_and_black_error(eval_date, swaption, yield_curve_handle, normal_error_bps)
            if result:
                individual_black_vol_error, normal_vega = result
                total_weighted_error += individual_black_vol_error * normal_vega
                total_vega += normal_vega

        if total_vega == 0:
            return None # Avoid division by zero if no valid vegas were found

        # Calculate and return the final portfolio-level Black volatility error.
        portfolio_black_vol_error = total_weighted_error / total_vega
        return portfolio_black_vol_error

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

    # --- 2. Process Each Date in the DataFrame ---
    black_vols_nn = []
    black_vols_lm = []
    
    print("\n--- Processing each date to convert Normal volatility errors to Black ---")
    # Use tqdm for a progress bar over the DataFrame rows.
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Converting Errors"):
        eval_date = row['Date']
        
        # Get the normal volatility errors for the current date.
        rmse_nn_bps = row['RMSE_NN_OutOfSample']
        rmse_lm_bps = row['RMSE_LM_OutOfSample']
        
        # Convert the NN model error.
        black_vol_nn = convert_normal_to_black_for_date(eval_date, rmse_nn_bps, FOLDER_ZERO_CURVES, FOLDER_VOLATILITY_CUBES)
        black_vols_nn.append(black_vol_nn)
        
        # Convert the LM model error.
        black_vol_lm = convert_normal_to_black_for_date(eval_date, rmse_lm_bps, FOLDER_ZERO_CURVES, FOLDER_VOLATILITY_CUBES)
        black_vols_lm.append(black_vol_lm)

    # --- 3. Store Results and Save Output ---
    print("\n--- Aggregating results and saving output file ---")
    
    # Add the new lists as columns to the DataFrame.
    # The results are multiplied by 100 to express them in percentage points (%).
    df['BlackVol_NN'] = [vol * 100 if vol is not None else np.nan for vol in black_vols_nn]
    df['BlackVol_LM'] = [vol * 100 if vol is not None else np.nan for vol in black_vols_lm]

    # Perform basic error handling and summary.
    processed_count = df['BlackVol_NN'].notna().sum()
    if processed_count == 0:
        print("\nCRITICAL WARNING: No dates could be processed.")
        print("Please check that the dates in the CSV match the filenames in the data folders.")
        print(f"Example expected file path for first date ({df['Date'].iloc[0]}):")
        print(f"  {os.path.join(FOLDER_ZERO_CURVES, df['Date'].iloc[0].strftime('%d.%m.%Y') + '.csv')}")
    else:
        print(f"\nSuccessfully processed {processed_count} out of {len(df)} dates.")
        
    # Save the updated DataFrame to a new CSV file.
    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(OUTPUT_CSV_PATH)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"\nResults successfully saved to: {OUTPUT_CSV_PATH}")
    except Exception as e:
        print(f"\nCRITICAL ERROR: Failed to save the output file. Details: {e}")

    print("\n--- Script finished. ---")