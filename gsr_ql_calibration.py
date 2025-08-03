import datetime
import os
import pandas as pd
import numpy as np
import QuantLib as ql
from numpy.typing import NDArray
from typing import List, Tuple, Optional, Set
import json
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


#--------------------CONFIG--------------------
# Set to True to re-run data preparation, False to use existing processed/bootstrapped files.
PREPROCESS_CURVES: bool = False
BOOTSTRAP_CURVES: bool = False

FOLDER_SWAP_CURVES: str = r'data\EUR SWAP CURVE'    # Folder containing the raw swap curves.
FOLDER_ZERO_CURVES: str = r'data\EUR ZERO CURVE'    # Folder containing bootstrapped zero curves or in which the those will be stored.
FOLDER_VOLATILITY_CUBES: str = r'data\EUR BVOL CUBE'

FOLDER_RESULTS: str = r'results\traditional\predictions'
FOLDER_RESULTS_PARAMS: str = r'results\traditional\parameters'


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


#--------------------LOAD ZERO CURVES--------------------
zero_curves: list[pd.DataFrame] = []
if os.path.exists(FOLDER_ZERO_CURVES):
    for entry_name in os.listdir(FOLDER_ZERO_CURVES):
        if entry_name.endswith('.csv'):
            zero_curve_path = os.path.join(FOLDER_ZERO_CURVES, entry_name)
            zero_curve: pd.DataFrame = pd.read_csv(zero_curve_path)
            zero_curves.append(zero_curve)


#--------------------LOAD SWAPTION VOLATILITY CUBES--------------------
swaption_volatility_cubes: list[pd.DataFrame] = []
vol_xlsx_folder = os.path.join(FOLDER_VOLATILITY_CUBES, 'xlsx')
if os.path.exists(vol_xlsx_folder):
    for entry_name in os.listdir(vol_xlsx_folder):
        if entry_name.endswith('.xlsx'):
            vol_path = os.path.join(vol_xlsx_folder, entry_name)
            swaption_volatility_cube: pd.DataFrame = pd.read_excel(vol_path)
            swaption_volatility_cube.rename(columns={'Unnamed: 1': 'Type'}, inplace=True)
            for col in swaption_volatility_cube.columns:
                if 'Unnamed' in str(col):
                    swaption_volatility_cube.drop(col, axis=1, inplace=True)
            swaption_volatility_cube['Expiry'] = swaption_volatility_cube['Expiry'].ffill()
            swaption_volatility_cubes.append(swaption_volatility_cube)


#--------------------HULL-WHITE CALIBRATION WITH QUANTLIB--------------------

def parse_tenor(tenor_str: str) -> ql.Period:
    """Parses a tenor string (e.g., '1Yr', '6Mo') into a QuantLib Period object."""
    tenor_str = tenor_str.strip().upper()
    if 'YR' in tenor_str:
        num = int(tenor_str.replace('YR', ''))
        return ql.Period(num, ql.Years)
    elif 'MO' in tenor_str:
        num = int(tenor_str.replace('MO', ''))
        return ql.Period(num, ql.Months)
    raise ValueError(f"Could not parse tenor string: {tenor_str}")

def parse_tenor_to_years(tenor_str: str) -> float:
    """Parses a tenor string (e.g., '1Yr', '6Mo') into a float representing years."""
    tenor_str = tenor_str.strip().upper()
    if 'YR' in tenor_str:
        num = int(tenor_str.replace('YR', ''))
        return float(num)
    elif 'MO' in tenor_str:
        num = int(tenor_str.replace('MO', ''))
        return num / 12.0
    raise ValueError(f"Could not parse tenor string to years: {tenor_str}")

def create_ql_yield_curve(
    zero_curve_df: pd.DataFrame,
    eval_date: datetime.date
) -> ql.RelinkableYieldTermStructureHandle:
    """
    Creates a QuantLib YieldTermStructure from a bootstrapped zero curve DataFrame.
    """
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
    """
    Parses the swaption volatility cube and creates a list of QuantLib SwaptionHelper objects,
    optionally filtering by minimum expiry and tenor.
    """
    helpers_with_info = []
    vols_df = vol_cube_df[vol_cube_df['Type'] == 'Vol'].set_index('Expiry')
    strikes_df = vol_cube_df[vol_cube_df['Type'] == 'Strike'].set_index('Expiry')
    swap_index = ql.Euribor6M(term_structure_handle)

    filtered_count = 0
    total_count = 0

    for expiry_str in vols_df.index:
        for tenor_str in vols_df.columns:
            if tenor_str == 'Type': continue
            vol = vols_df.loc[expiry_str, tenor_str]
            strike = strikes_df.loc[expiry_str, tenor_str]
            if pd.isna(vol) or pd.isna(strike):
                continue

            total_count += 1

            expiry_in_years = parse_tenor_to_years(expiry_str)
            tenor_in_years = parse_tenor_to_years(tenor_str)

            if expiry_in_years < min_expiry_years or tenor_in_years < min_tenor_years:
                filtered_count += 1
                continue

            expiry_period = parse_tenor(expiry_str)
            tenor_period = parse_tenor(tenor_str)
            assert isinstance(vol, float), 'vol must be a float'
            vol_handle = ql.QuoteHandle(ql.SimpleQuote(vol / 10000.0))

            helper = ql.SwaptionHelper(expiry_period, tenor_period, vol_handle, swap_index, ql.Period(6, ql.Months),
                                       swap_index.dayCounter(), swap_index.dayCounter(), term_structure_handle, ql.Normal)
            helpers_with_info.append((helper, expiry_str, tenor_str))

    print(f"--- Preparing calibration instruments: {total_count - filtered_count} swaptions included ({filtered_count} filtered out). ---")

    return helpers_with_info

def plot_calibration_results(results_df: pd.DataFrame):
    """
    Shows 3 diagrams as 3D plots in the same figure after the calibration.
    This version includes a fix to prevent plot distortion from extreme outliers.
    """
    plot_data = results_df.dropna(subset=['MarketVol', 'ModelVol', 'Difference']).copy()

    if plot_data.empty:
        print("\nCould not generate plots: No valid data points available after calibration.")
        return

    X = plot_data['Expiry'].values
    Y = plot_data['Tenor'].values
    Z_market = plot_data['MarketVol'].values
    Z_model = plot_data['ModelVol'].values
    Z_diff = plot_data['Difference'].values

    fig = plt.figure(figsize=(24, 8))
    fig.suptitle('Hull-White Calibration Volatility Surfaces', fontsize=16)

    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.set_title('Observed Market Volatilities (bps)')
    surf1 = ax1.plot_trisurf(X, Y, Z_market, cmap=cm.viridis, antialiased=True, linewidth=0.1)
    ax1.set_xlabel('Expiry (Years)')
    ax1.set_ylabel('Tenor (Years)')
    ax1.set_zlabel('Volatility (bps)')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, pad=0.1)
    ax1.view_init(elev=30, azim=-120)

    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.set_title('Model Implied Volatilities (bps)')
    surf2 = ax2.plot_trisurf(X, Y, Z_model, cmap=cm.viridis, antialiased=True, linewidth=0.1)
    ax2.set_xlabel('Expiry (Years)')
    ax2.set_ylabel('Tenor (Years)')
    ax2.set_zlabel('Volatility (bps)')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, pad=0.1)
    ax2.zaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))
    ax2.view_init(elev=30, azim=-120)

    market_min, market_max = np.nanmin(Z_market), np.nanmax(Z_market)
    ax2.set_zlim(market_min * 0.9, market_max * 1.1)

    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.set_title('Difference (Market - Model) (bps)')
    surf3 = ax3.plot_trisurf(X, Y, Z_diff, cmap=cm.coolwarm_r, antialiased=True, linewidth=0.1)
    ax3.set_xlabel('Expiry (Years)')
    ax3.set_ylabel('Tenor (Years)')
    ax3.set_zlabel('Difference (bps)')
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10, pad=0.1)
    ax3.zaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
    ax3.view_init(elev=30, azim=-120)

    max_reasonable_diff = 50.0
    ax3.set_zlim(-max_reasonable_diff, max_reasonable_diff)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def _get_step_dates_from_expiries(
    ql_eval_date: ql.Date,
    included_expiries_yrs: List[float],
    num_segments: int
) -> List[ql.Date]:
    """
    Calculates the step dates for piecewise parameters based on available expiry dates.
    """
    if num_segments <= 1:
        return []

    if len(included_expiries_yrs) < num_segments:
         print(f"Warning: Not enough unique expiries ({len(included_expiries_yrs)}) to create "
               f"{num_segments - 1} steps. Reducing number of segments to {len(included_expiries_yrs)}.")
         num_segments = len(included_expiries_yrs)

    if num_segments <= 1:
        return []

    indices = np.linspace(0, len(included_expiries_yrs) - 1, num_segments + 1).astype(int)[1:-1]
    time_points_in_years = [included_expiries_yrs[i] for i in indices]

    step_dates = [ql_eval_date + ql.Period(int(y * 365.25), ql.Days) for y in time_points_in_years]
    return step_dates

# --- NEW HELPER FUNCTION TO MANAGE UNIFIED TIMELINE ---
def _expand_params_to_unified_timeline(
    initial_params: List[ql.QuoteHandle],
    param_step_dates: List[ql.Date],
    unified_step_dates: List[ql.Date]
) -> List[ql.QuoteHandle]:
    """
    Expands a list of parameter handles to match a new, unified timeline.
    """
    if not unified_step_dates:
        return initial_params

    if not param_step_dates:
        # If the parameter was constant, repeat its handle for each new segment.
        return [initial_params[0]] * (len(unified_step_dates) + 1)

    expanded_handles = []
    time_intervals = [float('-inf')] + [d.serialNumber() for d in unified_step_dates] + [float('inf')]

    for i in range(len(time_intervals) - 1):
        mid_point_serial = (time_intervals[i] + time_intervals[i+1]) / 2

        import bisect
        original_date_serials = [d.serialNumber() for d in param_step_dates]
        idx = bisect.bisect_right(original_date_serials, mid_point_serial)
        expanded_handles.append(initial_params[idx])

    return expanded_handles


# --- UPDATED CALIBRATION FUNCTION ---
def calibrate_hull_white(
    eval_date: datetime.date,
    zero_curve_df: pd.DataFrame,
    vol_cube_df: pd.DataFrame,
    num_a_segments: int = 1,
    num_sigma_segments: int = 1,
    optimize_a: bool = False,
    initial_a: float = 0.01,
    initial_sigma: float = 0.01,
    min_expiry_years: float = 0.0,
    min_tenor_years: float = 0.0
) -> Tuple[List[float], List[ql.Date], List[float], List[ql.Date], pd.DataFrame]:
    """
    Calibrates a one-factor Hull-White (via the GSR model) to a set of ATM swaptions.
    
    Returns:
        A tuple containing:
        - final_as (List[float]): The calibrated mean-reversion values.
        - a_step_dates (List[ql.Date]): The step dates for the 'a' parameter.
        - final_sigmas (List[float]): The calibrated volatility values.
        - sigma_step_dates (List[ql.Date]): The step dates for the 'sigma' parameter.
        - results_df (pd.DataFrame): DataFrame with detailed calibration errors.
    """
    ql_eval_date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
    ql.Settings.instance().evaluationDate = ql_eval_date

    term_structure_handle = create_ql_yield_curve(zero_curve_df, eval_date)

    helpers_with_info = prepare_calibration_helpers(
        vol_cube_df,
        term_structure_handle,
        min_expiry_years=min_expiry_years,
        min_tenor_years=min_tenor_years
    )
    if not helpers_with_info:
        raise ValueError("No valid swaption helpers could be created from the input data given the filtering criteria.")

    included_expiries_yrs = sorted(list(set([parse_tenor_to_years(expiry_str) for _, expiry_str, _ in helpers_with_info])))

    a_step_dates = _get_step_dates_from_expiries(ql_eval_date, included_expiries_yrs, num_a_segments)
    sigma_step_dates = _get_step_dates_from_expiries(ql_eval_date, included_expiries_yrs, num_sigma_segments)
    
    unified_step_dates = sorted(list(set(a_step_dates + sigma_step_dates)))
    if unified_step_dates:
        print(f"--- Using unified step dates at years: {[round(ql.Actual365Fixed().yearFraction(ql_eval_date, d), 2) for d in unified_step_dates]} ---")

    sigma_handles_initial = [ql.QuoteHandle(ql.SimpleQuote(initial_sigma)) for _ in range(num_sigma_segments)]
    if optimize_a:
        reversion_handles_initial = [ql.QuoteHandle(ql.SimpleQuote(initial_a)) for _ in range(num_a_segments)]
    else:
        reversion_handles_initial = [ql.QuoteHandle(ql.SimpleQuote(initial_a))]

    expanded_sigma_handles = _expand_params_to_unified_timeline(sigma_handles_initial, sigma_step_dates, unified_step_dates)
    expanded_reversion_handles = _expand_params_to_unified_timeline(reversion_handles_initial, a_step_dates, unified_step_dates)

    model = ql.Gsr(term_structure_handle, unified_step_dates, expanded_sigma_handles, expanded_reversion_handles, 61.0)
    
    calibration_helpers = [h for h, _, _ in helpers_with_info]
    engine = ql.Gaussian1dSwaptionEngine(model, 64, 7.0, True, False, term_structure_handle)
    for helper in calibration_helpers:
        helper.setPricingEngine(engine)

    method = ql.LevenbergMarquardt()
    end_criteria = ql.EndCriteria(400, 100, 1.0e-8, 1.0e-8, 1.0e-8)
    
    if optimize_a:
        model.calibrate(calibration_helpers, method, end_criteria, ql.PositiveConstraint())
    else:
        model.calibrateVolatilitiesIterative(calibration_helpers, method, end_criteria)
    
    calibrated_expanded_as = list(model.reversion())
    calibrated_expanded_sigmas = list(model.volatility())

    def get_unique_calibrated_values(expanded_values: List[float], original_step_dates: List[ql.Date], unified_step_dates: List[ql.Date]) -> List[float]:
        if not original_step_dates:
            return [expanded_values[0]] 
        
        indices_to_report = [0]
        original_serials = {d.serialNumber() for d in original_step_dates}
        for i, d_unified in enumerate(unified_step_dates):
            if d_unified.serialNumber() in original_serials:
                indices_to_report.append(i + 1)
        
        unique_vals = list(dict.fromkeys([expanded_values[i] for i in sorted(list(set(indices_to_report)))]))
        return unique_vals

    final_as = get_unique_calibrated_values(calibrated_expanded_as, a_step_dates, unified_step_dates)
    final_sigmas = get_unique_calibrated_values(calibrated_expanded_sigmas, sigma_step_dates, unified_step_dates)

    def print_parameter_results(name: str, values: list, step_dates: list, status: str):
        print(f"Calibrated {name} values ({len(values)} segments): {status}")
        if not step_dates:
            print(f"  {name}_1 (constant): {values[0]:.6f}")
            return
            
        times = [0.0] + [ql.Actual365Fixed().yearFraction(ql_eval_date, d) for d in step_dates]
        for i, val in enumerate(values):
            end_time_str = f"{times[i+1]:.1f}Y" if i < len(times) - 1 else "inf"
            print(f"  {name}_{i+1} (t in [{times[i]:.1f}Y, {end_time_str}]): {val:.6f}")

    print("\n--- Hull-White (GSR) Calibration Results ---")
    print(f"Evaluation Date: {eval_date.strftime('%Y-%m-%d')}")
    print("-" * 45)
    print_parameter_results("Mean Reversion (a)", final_as, a_step_dates, '(Calibrated)' if optimize_a else '(Fixed)')
    print("-" * 45)
    print_parameter_results("Volatility (sigma)", final_sigmas, sigma_step_dates, '(Calibrated)')
    print("-" * 45)
    
    print("\n--- Calibration Diagnostics (Market vs. Model) ---")
    print(f"{'Expiry':>6} | {'Tenor':>6} | {'Market Vol (bps)':>18} | {'Model Vol (bps)':>16} | {'Error (bps)':>14}")
    print("-" * 80)
    
    vols_df = vol_cube_df[vol_cube_df['Type'] == 'Vol'].set_index('Expiry')
    
    results_data = []
    for helper, expiry_str_orig, tenor_str_orig in helpers_with_info:
        market_vol_bps = vols_df.loc[expiry_str_orig, tenor_str_orig]
        model_npv = helper.modelValue()
        try:
            implied_vol = helper.impliedVolatility(model_npv, 1e-4, 500, 0.0001, 1.0)
            model_vol_bps = implied_vol * 10000
            error_bps = model_vol_bps - market_vol_bps
        except RuntimeError:
            model_vol_bps = float('nan')
            error_bps = float('nan')
        
        expiry_str = expiry_str_orig.replace("YR", "Y").replace("MO", "M")
        tenor_str = tenor_str_orig.replace("YR", "Y").replace("MO", "M")
        
        print(f"{expiry_str:>6} | {tenor_str:>6} | {market_vol_bps:18.4f} | {model_vol_bps:16.4f} | {error_bps:14.4f}")
        
        results_data.append({
            'ExpiryStr': expiry_str_orig,
            'TenorStr': tenor_str_orig,
            'MarketVol': market_vol_bps,
            'ModelVol': model_vol_bps,
            'Difference': error_bps
        })
        
    results_df = pd.DataFrame(results_data)
    results_df['Expiry'] = results_df['ExpiryStr'].apply(parse_tenor_to_years)
    results_df['Tenor'] = results_df['TenorStr'].apply(parse_tenor_to_years)
    
    # --- NEW: Return all parameter information ---
    return final_as, a_step_dates, final_sigmas, sigma_step_dates, results_df


# --- UPDATED: Function to run calibration for a single day and save all results ---
def run_and_save_calibration_for_date(
    eval_date_str: str,
    calibration_settings: dict,
    show_plot: bool = False
):
    """
    Loads data, runs calibration, saves the results DataFrame,
    and saves the model parameters to a JSON file.
    """
    print(f"\n{'='*25} STARTING CALIBRATION FOR: {eval_date_str} {'='*25}")
    try:
        eval_date = datetime.datetime.strptime(eval_date_str, "%d.%m.%Y").date()

        # Define file paths
        zero_curve_path = os.path.join(FOLDER_ZERO_CURVES, f"{eval_date_str}.csv")
        vol_cube_path = os.path.join(FOLDER_VOLATILITY_CUBES, "xlsx", f"{eval_date_str}.xlsx")

        # Check for file existence
        if not os.path.exists(zero_curve_path) or not os.path.exists(vol_cube_path):
            raise FileNotFoundError(f"Data files for {eval_date_str} not found. Skipping.")

        # Load data
        zero_curve_df = pd.read_csv(zero_curve_path)
        vol_cube_df = pd.read_excel(vol_cube_path, engine='openpyxl')

        # Preprocess volatility cube
        vol_cube_df.rename(columns={'Unnamed: 1': 'Type'}, inplace=True)
        for col in vol_cube_df.columns:
            if 'Unnamed' in str(col):
                vol_cube_df.drop(col, axis=1, inplace=True)
        vol_cube_df['Expiry'] = vol_cube_df['Expiry'].ffill()

        # --- UPDATED: Unpack all results from calibration function ---
        calibrated_as, a_step_dates, calibrated_sigmas, sigma_step_dates, results_df = calibrate_hull_white(
            eval_date=eval_date,
            zero_curve_df=zero_curve_df,
            vol_cube_df=vol_cube_df,
            **calibration_settings
        )

        output_date_str = eval_date.strftime("%d-%m-%Y")

        # Save results DataFrame
        if not results_df.empty:
            output_path = os.path.join(FOLDER_RESULTS, f"{output_date_str}.csv")
            results_df.to_csv(output_path, index=False)
            print(f"\n--- Results for {eval_date_str} successfully saved to: {output_path} ---")
            
            if show_plot:
                plot_calibration_results(results_df)
        else:
            print(f"\n--- Calibration for {eval_date_str} did not produce results to save. ---")

        # --- NEW: Save calibrated parameters to JSON ---
        if calibrated_as and calibrated_sigmas:
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

            param_output_path = os.path.join(FOLDER_RESULTS_PARAMS, f"{output_date_str}.json")
            with open(param_output_path, 'w') as f:
                json.dump(param_data, f, indent=4)
            print(f"--- Parameters for {eval_date_str} successfully saved to: {param_output_path} ---")


    except Exception as e:
        import traceback
        print(f"\n--- ERROR during calibration for {eval_date_str}: {e} ---")
        traceback.print_exc()
        
    print(f"\n{'='*25} FINISHED CALIBRATION FOR: {eval_date_str} {'='*25}")

if __name__ == '__main__':
    # --- Ensure the results directories exist ---
    os.makedirs(FOLDER_RESULTS, exist_ok=True)
    os.makedirs(FOLDER_RESULTS_PARAMS, exist_ok=True)
    
    # --- Define calibration settings ---
    CALIBRATION_SETTINGS = {
        "num_a_segments": 1,
        "num_sigma_segments": 3,
        "optimize_a": True,
        "initial_a": 0.02,
        "initial_sigma": 0.005,
        "min_expiry_years": 1.0,
        "min_tenor_years": 1.0
    }

    # --- Find all dates with available data ---
    try:
        vol_files_folder = os.path.join(FOLDER_VOLATILITY_CUBES, 'xlsx')

        zero_curve_dates: Set[str] = {
            f.replace('.csv', '') for f in os.listdir(FOLDER_ZERO_CURVES) if f.endswith('.csv')
        }
        vol_cube_dates: Set[str] = {
            f.replace('.xlsx', '') for f in os.listdir(vol_files_folder) if f.endswith('.xlsx')
        }

        available_dates: List[str] = sorted(list(zero_curve_dates.intersection(vol_cube_dates)))

        if not available_dates:
            print("No matching data found. Please check your data folders:")
            print(f"Zero Curves: {FOLDER_ZERO_CURVES}")
            print(f"Volatility Cubes: {vol_files_folder}")
        else:
            print(f"Found {len(available_dates)} matching data sets. Starting batch calibration...")
            
            for date_str in available_dates:
                run_and_save_calibration_for_date(
                    date_str,
                    CALIBRATION_SETTINGS,
                    show_plot=False
                )

            print("\n\nBatch calibration process finished for all available dates.")

    except FileNotFoundError as e:
        print(f"\nError: A required data folder was not found.")
        print(f"Details: {e}")
    except Exception as e:
        import traceback
        print(f"\nAn unexpected error occurred during the main process: {e}")
        traceback.print_exc()