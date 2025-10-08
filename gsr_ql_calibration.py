import datetime
import os
import pandas as pd
import numpy as np
import QuantLib as ql
from numpy.typing import NDArray
from typing import List, Tuple, Optional, Set, Union
import sys
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#--------------------CONFIG--------------------
PREPROCESS_CURVES: bool = False
BOOTSTRAP_CURVES: bool = False

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

FOLDER_SWAP_CURVES: str = os.path.join(DATA_DIR, 'EUR SWAP CURVE')
FOLDER_ZERO_CURVES: str = os.path.join(DATA_DIR, 'EUR ZERO CURVE')
FOLDER_VOLATILITY_CUBES: str = os.path.join(DATA_DIR, 'EUR BVOL CUBE')

# NEW: Define a single output folder for the consolidated results
FOLDER_RESULTS_TRADITIONAL: str = os.path.join(RESULTS_DIR, 'traditional')


#--------------------PREPROCESS CURVES--------------------
if PREPROCESS_CURVES:
    print("--- Starting: Preprocessing Raw Swap Curves ---")
    processed_folder: str = os.path.join(FOLDER_SWAP_CURVES, 'processed')
    raw_folder: str = os.path.join(FOLDER_SWAP_CURVES, 'raw')
    os.makedirs(processed_folder, exist_ok=True)
    if not os.path.exists(raw_folder):
        print(f"Warning: Raw data folder not found at {raw_folder}")
        os.makedirs(raw_folder, exist_ok=True)


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
            ql.Annual,
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
    os.makedirs(FOLDER_ZERO_CURVES, exist_ok=True)


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
            swaption_volatility_cube: pd.DataFrame = pd.read_excel(vol_path, engine='openpyxl')
            swaption_volatility_cube.rename(columns={'Unnamed: 1': 'Type'}, inplace=True)
            for col in swaption_volatility_cube.columns:
                if 'Unnamed' in str(col):
                    swaption_volatility_cube.drop(col, axis=1, inplace=True)
            swaption_volatility_cube['Expiry'] = swaption_volatility_cube['Expiry'].ffill()
            swaption_volatility_cubes.append(swaption_volatility_cube)


#--------------------HULL-WHITE CALIBRATION WITH QUANTLIB--------------------
def parse_tenor(tenor_str: str) -> ql.Period:
    """
    Parses a tenor string into a QuantLib Period object.

    Args:
        tenor_str (str): A string representing the tenor, e.g., '5YR' for 5 years or '6MO' for 6 months.

    Returns:
        ql.Period: A QuantLib Period object representing the parsed tenor.

    Raises:
        ValueError: If the tenor string cannot be parsed.
    """
    tenor_str = tenor_str.strip().upper()
    if 'YR' in tenor_str:
        num = int(tenor_str.replace('YR', ''))
        return ql.Period(num, ql.Years)
    elif 'MO' in tenor_str:
        num = int(tenor_str.replace('MO', ''))
        return ql.Period(num, ql.Months)
    raise ValueError(f"Could not parse tenor string: {tenor_str}")

def parse_tenor_to_years(tenor_str: str) -> float:
    """
    Parses a tenor string into a float representing years.

    Args:
        tenor_str (str): A string representing the tenor, e.g., '5YR' for 5 years or '6MO' for 6 months.

    Returns:
        float: The parsed tenor in years.

    Raises:
        ValueError: If the tenor string cannot be parsed.
    """
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
    Creates a QuantLib ZeroCurve object from a Pandas DataFrame containing a zero curve, and a datetime.date object representing the evaluation date.

    Args:
        zero_curve_df (pd.DataFrame): A Pandas DataFrame containing the zero curve data.
            The DataFrame should have columns 'Date' and 'ZeroRate'.
        eval_date (datetime.date): The evaluation date.

    Returns:
        ql.RelinkableYieldTermStructureHandle: A handle to the created QuantLib ZeroCurve object.
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
    Parses the swaption volatility cube and creates a list of QuantLib SwaptionHelper objects, optionally filtering by minimum expiry and tenor.

    Args:
        vol_cube_df (pd.DataFrame): The DataFrame containing the swaption volatility cube.
        term_structure_handle (ql.RelinkableYieldTermStructureHandle): The yield curve to use for pricing.
        min_expiry_years (float, optional): The minimum expiry time in years. Defaults to 0.0.
        min_tenor_years (float, optional): The minimum tenor time in years. Defaults to 0.0.

    Returns:
        List[Tuple[ql.SwaptionHelper, str, str]]: A list of tuples, where each tuple contains a SwaptionHelper,
            the expiry string and the tenor string.
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

def plot_calibration_results(results_df: pd.DataFrame, eval_date: datetime.date, save_dir: str):
    """
    Shows 3 diagrams as 3D plots in the same figure after the calibration.

    Plots the observed market volatilities, model implied volatilities, and the difference between them.

    Args:
        results_df (pd.DataFrame): The DataFrame with the calibration results.
        eval_date (datetime.date): The evaluation date for the plot title.
        save_dir (str): The directory to save the plot image.
    """
    plot_data = results_df.dropna(subset=['MarketVol', 'ModelVol', 'Difference_bps']).copy()

    if plot_data.empty:
        print(f"\nCould not generate plots for {eval_date}: No valid data points available.")
        return

    X = plot_data['Expiry'].values
    Y = plot_data['Tenor'].values
    Z_market = plot_data['MarketVol'].values
    Z_model = plot_data['ModelVol'].values
    Z_diff = plot_data['Difference_bps'].values

    fig = plt.figure(figsize=(24, 8))
    fig.suptitle(f'Hull-White Calibration Volatility Surfaces for {eval_date}', fontsize=16)

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
    
    market_min, market_max = np.nanmin(Z_market), np.nanmax(Z_market)
    ax2.set_zlim(market_min * 0.9, market_max * 1.1)
    ax2.view_init(elev=30, azim=-120)

    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.set_title('Difference (Market - Model) (bps)')
    surf3 = ax3.plot_trisurf(X, Y, Z_diff, cmap=cm.coolwarm_r, antialiased=True, linewidth=0.1)
    ax3.set_xlabel('Expiry (Years)')
    ax3.set_ylabel('Tenor (Years)')
    ax3.set_zlabel('Difference (bps)')
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10, pad=0.1)
    
    max_reasonable_diff = np.nanmax(np.abs(Z_diff))
    ax3.set_zlim(-max_reasonable_diff, max_reasonable_diff)
    ax3.view_init(elev=30, azim=-120)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_dir, f'CalibrationPlot_{eval_date}.png'))
    plt.show()

def _get_step_dates_from_expiries(
    ql_eval_date: ql.Date,
    included_expiries_yrs: List[float],
    num_segments: int
) -> List[ql.Date]:
    """
    Calculates the step dates for piecewise parameters based on available expiry dates.
    
    Args:
        ql_eval_date (ql.Date): The evaluation date of the yield curve.
        included_expiries_yrs (List[float]): The sorted list of expiry years.
        num_segments (int): The number of segments to split the expiry years into.
    
    Returns:
        List[ql.Date]: The step dates of the piecewise parameters.
    """
    if num_segments <= 1:
        return []
    
    unique_expiries = sorted(list(set(included_expiries_yrs)))

    if len(unique_expiries) < num_segments:
         print(f"Warning: Not enough unique expiries ({len(unique_expiries)}) to create "
               f"{num_segments - 1} steps. Reducing number of segments to {len(unique_expiries)}.")
         num_segments = len(unique_expiries)

    if num_segments <= 1:
        return []

    indices = np.linspace(0, len(unique_expiries) - 1, num_segments + 1).astype(int)[1:-1]
    time_points_in_years = [unique_expiries[i] for i in indices]

    step_dates = [ql_eval_date + ql.Period(int(y * 365.25), ql.Days) for y in time_points_in_years]
    return step_dates

def _expand_params_to_unified_timeline(
    initial_params: List[ql.QuoteHandle],
    param_step_dates: List[ql.Date],
    unified_step_dates: List[ql.Date]
) -> List[ql.QuoteHandle]:
    """
    Expands a list of initial parameter quotes to align with a unified timeline.

    Args:
        initial_params (List[ql.QuoteHandle]): The initial parameter quotes as QuoteHandles.
        param_step_dates (List[ql.Date]): The original step dates associated with the initial parameters.
        unified_step_dates (List[ql.Date]): The new, unified step dates to align the parameter quotes with.

    Returns:
        List[ql.QuoteHandle]: A list of QuoteHandles that represent the initial parameters expanded to the unified timeline.
    """
    if not unified_step_dates:
        return initial_params

    if not param_step_dates:
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

def calibrate_hull_white(
    eval_date: datetime.date,
    zero_curve_df: pd.DataFrame,
    vol_cube_df: pd.DataFrame,
    num_a_segments: int = 1,
    num_sigma_segments: Union[int, str] = 'auto',
    optimize_a: bool = False,
    initial_a: float = 0.01,
    initial_sigma: float = 0.01,
    min_expiry_years: float = 0.0,
    min_tenor_years: float = 0.0,
    pricing_engine_integration_points: int = 64
) -> Tuple[Optional[List[float]], Optional[List[float]], Optional[pd.DataFrame], float]:
    """
    Calibrate a Hull-White model to a set of swaption market quotes.

    Parameters
    ----------
    eval_date : datetime.date
        The evaluation date for the calibration.
    zero_curve_df : pd.DataFrame
        The zero curve data frame as produced by `create_zero_curve`.
    vol_cube_df : pd.DataFrame
        The swaption volatility cube data frame as produced by `create_vol_cube`.
    num_a_segments : int, optional
        The number of mean reversion segments to use. Defaults to 1.
    num_sigma_segments : int or str, optional
        The number of volatility segments to use. If 'auto', this is set to the number
        of unique expiries in the input data. Defaults to 'auto'.
    optimize_a : bool, optional
        Whether to optimize the mean reversion parameter. Defaults to False.
    initial_a : float, optional
        The initial guess for the mean reversion parameter. Defaults to 0.01.
    initial_sigma : float, optional
        The initial guess for the volatility parameter. Defaults to 0.01.
    min_expiry_years : float, optional
        The minimum expiry time (in years) for which to include swaptions in the calibration.
        Defaults to 0.0.
    min_tenor_years : float, optional
        The minimum tenor time (in years) for which to include swaptions in the calibration.
        Defaults to 0.0.
    pricing_engine_integration_points : int, optional
        The number of integration points to use in the Gaussian1dSwaptionEngine. Defaults to 64.

    Returns
    -------
    as : List[float]
        The calibrated mean reversion parameters.
    a_step_dates : List[ql.Date]
        The step dates associated with the mean reversion parameters.
    sigmas : List[float]
        The calibrated volatility parameters.
    sigma_step_dates : List[ql.Date]
        The step dates associated with the volatility parameters.
    results_df : pd.DataFrame
        A data frame containing the calibration results, including market and model volatilities.
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
        print("No valid swaption helpers could be created from the input data given the filtering criteria.")
        return None, None, None, float('nan')

    included_expiries_yrs = sorted(list(set([parse_tenor_to_years(expiry_str) for _, expiry_str, _ in helpers_with_info])))

    final_num_sigma_segments: int
    if isinstance(num_sigma_segments, str) and num_sigma_segments.lower() == 'auto':
        final_num_sigma_segments = len(included_expiries_yrs)
        print(f"--- 'auto' setting: Number of volatility segments set to {final_num_sigma_segments} (matching unique expiries). ---")
        if final_num_sigma_segments == 0:
            print("Automatic volatility parameter determination failed: No swaption expiries remain after filtering.")
            return None, None, None, float('nan')
        sigma_step_dates = [ql_eval_date + ql.Period(int(y * 365.25), ql.Days) for y in included_expiries_yrs[:-1]]
    else:
        final_num_sigma_segments = int(num_sigma_segments)
        sigma_step_dates = _get_step_dates_from_expiries(ql_eval_date, included_expiries_yrs, final_num_sigma_segments)
        print(f"--- Fixed setting: Number of volatility segments set to {final_num_sigma_segments}. ---")
    
    a_step_dates = _get_step_dates_from_expiries(ql_eval_date, included_expiries_yrs, num_a_segments)

    unified_step_dates = sorted(list(set(a_step_dates + sigma_step_dates)))
    if unified_step_dates:
        print(f"--- Using unified step dates at years: {[round(ql.Actual365Fixed().yearFraction(ql_eval_date, d), 2) for d in unified_step_dates]} ---")

    sigma_handles_initial = [ql.QuoteHandle(ql.SimpleQuote(initial_sigma)) for _ in range(final_num_sigma_segments)]
    if optimize_a:
        reversion_handles_initial = [ql.QuoteHandle(ql.SimpleQuote(initial_a)) for _ in range(num_a_segments)]
    else:
        reversion_handles_initial = [ql.QuoteHandle(ql.SimpleQuote(initial_a))]

    expanded_sigma_handles = _expand_params_to_unified_timeline(sigma_handles_initial, sigma_step_dates, unified_step_dates)
    expanded_reversion_handles = _expand_params_to_unified_timeline(reversion_handles_initial, a_step_dates, unified_step_dates)

    model = ql.Gsr(term_structure_handle, unified_step_dates, expanded_sigma_handles, expanded_reversion_handles, 61.0)
    
    calibration_helpers = [h for h, _, _ in helpers_with_info]
    engine = ql.Gaussian1dSwaptionEngine(model, pricing_engine_integration_points, 7.0, True, False, term_structure_handle)
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
        """
        Extracts and returns unique calibrated values corresponding to original step dates.

        This function takes expanded calibration values and maps them back to the original
        step dates by identifying the indices that align with the original steps in the
        unified timeline. It ensures that only unique values are retained in their order
        of appearance.

        Args:
            expanded_values (List[float]): The list of expanded calibration values.
            original_step_dates (List[ql.Date]): The original step dates associated with the calibration.
            unified_step_dates (List[ql.Date]): The unified step dates encompassing all segments.

        Returns:
            List[float]: A list of unique calibrated values corresponding to the original step dates.
        """
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
    
    vols_df = vol_cube_df[vol_cube_df['Type'] == 'Vol'].set_index('Expiry')
    
    results_data = []
    squared_errors = []
    for helper, expiry_str, tenor_str in helpers_with_info:
        market_vol: float = helper.volatility().value()
        market_vol_bps: float = market_vol * 10000
        try:
            model_npv: float = helper.modelValue()
            model_vol: float = helper.impliedVolatility(model_npv, 1e-4, 500, 0.0001, 1.0)
            model_vol_bps: float = model_vol * 10000
            error_bps: float = model_vol_bps - market_vol_bps
            squared_errors.append(error_bps**2)
        except (RuntimeError, ValueError):
            model_vol_bps, error_bps = float('nan'), float('nan')

        results_data.append({
            'ExpiryStr': expiry_str, 'TenorStr': tenor_str,
            'MarketVol': market_vol_bps, 'ModelVol': model_vol_bps,
            'Difference_bps': error_bps
        })
        
    results_df = pd.DataFrame(results_data)
    results_df['Expiry'] = results_df['ExpiryStr'].apply(parse_tenor_to_years)
    results_df['Tenor'] = results_df['TenorStr'].apply(parse_tenor_to_years)
    
    rmse_bps: float = np.sqrt(np.mean(squared_errors)) if squared_errors else float('nan')
    
    print(f"Daily RMSE for {eval_date}: {rmse_bps:.4f} bps")

    return final_as, final_sigmas, results_df, rmse_bps


if __name__ == '__main__':
    os.makedirs(FOLDER_RESULTS_TRADITIONAL, exist_ok=True)
    
    CALIBRATION_SETTINGS = {
        # The number of mean reversion parameters to calibrate.
        "num_a_segments": 1,
        # The number of volatility parameters to calibrate.
        # Can be an integer or 'auto'. If 'auto', the number of parameters will be equal to the number of unique expiries.
        "num_sigma_segments": "3",
        # If True, the mean reversion parameter 'a' will be calibrated.
        # Otherwise, it will be fixed to the value of 'initial_a'.
        "optimize_a": True,
        # The initial value for the mean reversion parameter 'a'.
        "initial_a": 0.02,
        # The initial value for the volatility parameter 'sigma'.
        "initial_sigma": 0.005,
        # The number of integration points used in the pricing engine.
        "pricing_engine_integration_points": 32,
        # The minimum expiry in years for the swaptions to be included in the calibration.
        "min_expiry_years": 1.0,
        # The minimum tenor in years for the swaptions to be included in the calibration.
        "min_tenor_years": 1.0
    }

    try:
        vol_files_folder = os.path.join(FOLDER_VOLATILITY_CUBES, 'xlsx')

        if not os.path.exists(FOLDER_ZERO_CURVES):
            os.makedirs(FOLDER_ZERO_CURVES)
            print(f"Created missing directory: {FOLDER_ZERO_CURVES}")
        if not os.path.exists(vol_files_folder):
            os.makedirs(vol_files_folder)
            print(f"Created missing directory: {vol_files_folder}")

        zero_curve_dates: Set[str] = {f.replace('.csv', '') for f in os.listdir(FOLDER_ZERO_CURVES) if f.endswith('.csv')}
        vol_cube_dates: Set[str] = {f.replace('.xlsx', '') for f in os.listdir(vol_files_folder) if f.endswith('.xlsx')}

        available_dates: List[str] = sorted(list(zero_curve_dates.intersection(vol_cube_dates)))

        if not available_dates:
            print("No matching data found. Please check your data folders:")
            print(f"Zero Curves: {FOLDER_ZERO_CURVES}")
            print(f"Volatility Cubes: {vol_files_folder}")
        else:
            print(f"Found {len(available_dates)} matching data sets. Starting batch calibration...")
            
            all_results_dfs = []
            for date_str in available_dates:
                start_time = time.monotonic()
                print(f"\n{'='*25} STARTING CALIBRATION FOR: {date_str} {'='*25}")
                try:
                    eval_date = datetime.datetime.strptime(date_str, "%d.%m.%Y").date()
                    zero_curve_path = os.path.join(FOLDER_ZERO_CURVES, f"{date_str}.csv")
                    vol_cube_path = os.path.join(vol_files_folder, f"{date_str}.xlsx")

                    zero_curve_df = pd.read_csv(zero_curve_path)
                    vol_cube_df = pd.read_excel(vol_cube_path, engine='openpyxl')
                    vol_cube_df.rename(columns={'Unnamed: 1': 'Type'}, inplace=True)
                    for col in list(vol_cube_df.columns):
                        if 'Unnamed' in str(col): vol_cube_df.drop(col, axis=1, inplace=True)
                    vol_cube_df['Expiry'] = vol_cube_df['Expiry'].ffill()

                    calibrated_as, calibrated_sigmas, results_df, rmse_bps = calibrate_hull_white(
                        eval_date=eval_date,
                        zero_curve_df=zero_curve_df,
                        vol_cube_df=vol_cube_df,
                        **CALIBRATION_SETTINGS
                    )
                    end_time = time.monotonic()
                    elapsed = end_time - start_time
                    print(f"Finished calibration for {date_str} in {elapsed:.2f} seconds.")
                    
                    if results_df is not None and not results_df.empty:
                        day_results_df = results_df.copy()
                        day_results_df['EvaluationDate'] = eval_date
                        day_results_df['DailyRMSE_bps'] = rmse_bps
                        day_results_df['PredictionTimeSeconds'] = elapsed
                        
                        if calibrated_as:
                            for i, a_val in enumerate(calibrated_as):
                                day_results_df[f'a_{i+1}'] = a_val
                        if calibrated_sigmas:
                            for i, s_val in enumerate(calibrated_sigmas):
                                day_results_df[f'sigma_{i+1}'] = s_val
                        
                        all_results_dfs.append(day_results_df)

                except Exception as e:
                    import traceback
                    print(f"\n--- ERROR during calibration for {date_str}: {e} ---")
                    traceback.print_exc()
        
                print(f"{'='*25} FINISHED CALIBRATION FOR: {date_str} {'='*25}")

            if all_results_dfs:
                master_results_df = pd.concat(all_results_dfs, ignore_index=True)
                results_save_path = os.path.join(FOLDER_RESULTS_TRADITIONAL, 'evaluation_results.csv')
                master_results_df.to_csv(results_save_path, index=False)
                print(f"\n\n--- Comprehensive evaluation results saved to: {results_save_path} ---")

                last_eval_date = all_results_dfs[-1]['EvaluationDate'].iloc[0]
                print(f"\n--- Plotting calibration results for last day: {last_eval_date} ---")
                plot_calibration_results(all_results_dfs[-1], last_eval_date, FOLDER_RESULTS_TRADITIONAL)

            else:
                print("\n\nBatch calibration process finished, but no results were generated.")

    except Exception as e:
        import traceback
        print(f"\nAn unexpected error occurred during the main process: {e}")
        traceback.print_exc()