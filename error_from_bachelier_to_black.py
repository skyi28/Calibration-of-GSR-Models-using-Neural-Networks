"""
This script performs a financial model calibration task by converting a constant
model error from Normal (Bachelier) volatility terms to an equivalent,
vega-weighted Black (lognormal) volatility term.

The script reads historical financial data consisting of:
1. Daily EUR zero-coupon yield curves.
2. Corresponding daily swaption volatility cubes.

For each day, it parses all the swaptions from the volatility cube, calculates
their respective vegas (sensitivity to volatility), and then computes a
portfolio-wide, vega-weighted average Black volatility error. This provides a
single, representative error figure in Black's model terms, equivalent to the
input error in the Normal model.

The processing is parallelized across all available dates to improve performance.

Key Libraries:
- QuantLib: For financial instrument pricing and yield curve construction.
- pandas: For data manipulation and file I/O.
- concurrent.futures: For parallel processing of daily data.
"""
import datetime
import os
import sys
import pandas as pd
import numpy as np
import QuantLib as ql
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
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
# Alternative paths can be uncommented for testing purposes.
FOLDER_ZERO_CURVES: str = r'data/EUR ZERO CURVE'
# FOLDER_ZERO_CURVES: str = r'data\TESTS\EUcR ZERO CURVE'
FOLDER_VOLATILITY_CUBES: str = r'data/EUR BVOL CUBE'
# FOLDER_VOLATILITY_CUBES: str = r'data\TESTS\EUR BVOL CUBE'

# Define the constant model error in Normal volatility terms (Bachelier model).
MODEL_ERROR_NORMAL_BPS: float = 6.0
# Define a standard notional for vega calculation.
ASSUMED_NOTIONAL: float = 1.0

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

def load_all_data_files(
    zero_curve_folder: str,
    vol_cube_folder: str
) -> List[Tuple[datetime.date, str, str]]:
    """Discovers all matching pairs of zero curve and volatility cube files."""
    print("\n--- Discovering all available data files ---")
    vol_cube_xlsx_folder: str = os.path.join(vol_cube_folder, 'xlsx')
    if not os.path.exists(zero_curve_folder) or not os.path.exists(vol_cube_xlsx_folder):
        print(f"Checked paths:\n Zero Curves: {zero_curve_folder}\n Vol Cubes: {vol_cube_xlsx_folder}")
        raise FileNotFoundError("Data folders not found. Please check CONFIG paths.")

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
    print(f"Found {len(available_files)} complete data sets from {available_files[0][0]} to {available_files[-1][0]}.")
    return available_files

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

    except Exception as e:
        print(f"Error processing swaption {swaption} on {eval_date}: {e}")
        return None

def process_date_files(date_file_tuple: Tuple[datetime.date, str, str]) -> Dict:
    """Processes all swaptions for a single evaluation date."""
    eval_date, zero_path, vol_path = date_file_tuple
    
    # Load and build financial objects for the given date.
    zero_curve_df = pd.read_csv(zero_path, parse_dates=['Date'])
    vol_cube_df = load_volatility_cube(vol_path)
    yield_curve_handle = create_ql_yield_curve(zero_curve_df, eval_date)
    
    swaptions_to_process = get_swaption_details_from_cube(vol_cube_df)
    parsed_count = len(swaptions_to_process)
    valid_results = []
    
    for swaption in swaptions_to_process:
        result = calculate_vega_and_black_error(eval_date, swaption, yield_curve_handle, MODEL_ERROR_NORMAL_BPS)
        if result:
            valid_results.append(result)
            
    return {
        'parsed_count': parsed_count,
        'valid_count': len(valid_results),
        'results': valid_results
    }

# -------------------- MAIN EXECUTION BLOCK --------------------
if __name__ == "__main__":
    print("--- Starting: Vega-Weighted Model Error Conversion ---")
    print(f"Input model error is {MODEL_ERROR_NORMAL_BPS} bps in Normal (Bachelier) terms.")

    try:
        all_files = load_all_data_files(FOLDER_ZERO_CURVES, FOLDER_VOLATILITY_CUBES)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error during data discovery: {e}")
        sys.exit(1)

    all_results = []
    total_parsed = 0
    total_valid = 0

    # Use a process pool to parallelize the calculations across multiple CPU cores.
    with ProcessPoolExecutor() as executor:
        future_to_file = {executor.submit(process_date_files, file_tuple): file_tuple for file_tuple in all_files}
        print(f"\nSubmitting {len(all_files)} dates for processing...")
        
        # Process results as they are completed.
        for future in tqdm(as_completed(future_to_file), total=len(all_files), desc="Processing Dates"):
            try:
                report = future.result()
                total_parsed += report['parsed_count']
                total_valid += report['valid_count']
                if report['results']:
                    all_results.extend(report['results'])
            except Exception as exc:
                file_date = future_to_file[future][0]
                print(f'\nERROR: {file_date} generated an exception: {exc}')

    print("\n--- All dates processed. Aggregating results... ---")

    # --- Nicer Output Section ---
    print("\n" + "="*55)
    print(" " * 18 + "DIAGNOSTIC SUMMARY")
    print("="*55)
    print(f"  Total swaption points parsed from all files: {total_parsed:,}")
    print(f"  Total swaption points with valid forward rates: {total_valid:,}")
    print("="*55)

    if not all_results:
        print("\nCRITICAL ERROR: No valid swaption data could be processed.")
        if total_parsed > 0 and total_valid == 0:
            print("CONCLUSION: The script parsed your data, but 100% of the forward rates")
            print("            in the 'Strike' columns were negative or zero.")
        else:
            print("CONCLUSION: The script could not parse any valid swaption data from your files.")
        sys.exit(1)

    # Aggregate results for the final vega-weighted calculation.
    total_weighted_error = 0.0
    total_vega = 0.0
    for individual_black_vol_error, normal_vega in all_results:
        total_weighted_error += individual_black_vol_error * normal_vega
        total_vega += normal_vega

    if total_vega == 0:
        print("\nCRITICAL ERROR: Total portfolio vega is zero. Cannot divide by zero.")
        sys.exit(1)

    # Calculate the final portfolio-level Black volatility error.
    portfolio_black_vol_error = total_weighted_error / total_vega

    print("\n" + "#"*55)
    print(" " * 10 + "PORTFOLIO ERROR CONVERSION RESULTS")
    print("#"*55)
    print(f"  Input Model Error (Normal Vol) : {MODEL_ERROR_NORMAL_BPS:.2f} bps")
    print(f"  Total Number of Swaptions      : {len(all_results):,}")
    print(f"  Total Normal Vega (Notional=1) : {total_vega:,.4f}")
    print(f"-"*55)
    print(f"  Vega-Weighted Error (Black Vol): {portfolio_black_vol_error * 100:.4f} %")
    print("#"*55)
    print("\n--- Script finished successfully. ---")