# traditional_calibration.py

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
from matplotlib import cm

#--------------------CONFIG--------------------
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

FOLDER_SWAP_CURVES: str = os.path.join(DATA_DIR, 'EUR SWAP CURVE')
FOLDER_ZERO_CURVES: str = os.path.join(DATA_DIR, 'EUR ZERO CURVE')
FOLDER_VOLATILITY_CUBES: str = os.path.join(DATA_DIR, 'EUR BVOL CUBE')
FOLDER_RESULTS_TRADITIONAL: str = os.path.join(RESULTS_DIR, 'traditional')

#--------------------HELPER FUNCTIONS (UNCHANGED)--------------------
def parse_tenor(tenor_str: str) -> ql.Period:
    tenor_str = tenor_str.strip().upper()
    if 'YR' in tenor_str:
        num = int(tenor_str.replace('YR', ''))
        return ql.Period(num, ql.Years)
    elif 'MO' in tenor_str:
        num = int(tenor_str.replace('MO', ''))
        return ql.Period(num, ql.Months)
    raise ValueError(f"Could not parse tenor string: {tenor_str}")

def parse_tenor_to_years(tenor_str: str) -> float:
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
    ax1.set_xlabel('Expiry (Years)'); ax1.set_ylabel('Tenor (Years)'); ax1.set_zlabel('Volatility (bps)')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, pad=0.1)
    ax1.view_init(elev=30, azim=-120)
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.set_title('Model Implied Volatilities (bps)')
    surf2 = ax2.plot_trisurf(X, Y, Z_model, cmap=cm.viridis, antialiased=True, linewidth=0.1)
    ax2.set_xlabel('Expiry (Years)'); ax2.set_ylabel('Tenor (Years)'); ax2.set_zlabel('Volatility (bps)')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, pad=0.1)
    market_min, market_max = np.nanmin(Z_market), np.nanmax(Z_market)
    ax2.set_zlim(market_min * 0.9, market_max * 1.1)
    ax2.view_init(elev=30, azim=-120)
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.set_title('Difference (Model - Market) (bps)')
    surf3 = ax3.plot_trisurf(X, Y, Z_diff, cmap=cm.coolwarm, antialiased=True, linewidth=0.1)
    ax3.set_xlabel('Expiry (Years)'); ax3.set_ylabel('Tenor (Years)'); ax3.set_zlabel('Difference (bps)')
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10, pad=0.1)
    max_reasonable_diff = np.nanmax(np.abs(Z_diff))
    ax3.set_zlim(-max_reasonable_diff, max_reasonable_diff)
    ax3.view_init(elev=30, azim=-120)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_dir, f'CalibrationPlot_{eval_date}.png'))
    plt.close(fig)

def _get_step_dates_from_expiries(
    ql_eval_date: ql.Date,
    included_expiries_yrs: List[float],
    num_segments: int
) -> List[ql.Date]:
    if num_segments <= 1: return []
    unique_expiries = sorted(list(set(included_expiries_yrs)))
    if len(unique_expiries) < num_segments:
         print(f"Warning: Not enough unique expiries ({len(unique_expiries)}) to create "
               f"{num_segments - 1} steps. Reducing number of segments to {len(unique_expiries)}.")
         num_segments = len(unique_expiries)
    if num_segments <= 1: return []
    indices = np.linspace(0, len(unique_expiries) - 1, num_segments + 1).astype(int)[1:-1]
    time_points_in_years = [unique_expiries[i] for i in indices]
    step_dates = [ql_eval_date + ql.Period(int(y * 365.25), ql.Days) for y in time_points_in_years]
    return step_dates

def _expand_params_to_unified_timeline(
    initial_params: List[ql.QuoteHandle],
    param_step_dates: List[ql.Date],
    unified_step_dates: List[ql.Date]
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

#--------------------REFACTORED CALIBRATION FUNCTION--------------------
def calibrate_on_subset(
    eval_date: datetime.date,
    zero_curve_df: pd.DataFrame,
    list_of_calibration_helpers: List[Tuple[ql.SwaptionHelper, str, str]],
    term_structure_handle: ql.RelinkableYieldTermStructureHandle,
    num_a_segments: int = 1,
    num_sigma_segments: Union[int, str] = 'auto',
    optimize_a: bool = False,
    initial_a: float = 0.01,
    initial_sigma: float = 0.01,
    pricing_engine_integration_points: int = 64
) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """
    Calibrates a Hull-White model using a pre-defined subset of swaption helpers.
    This is the core modular function for the LM method.

    Args:
        eval_date: The evaluation date for the calibration.
        zero_curve_df: The zero curve data frame.
        list_of_calibration_helpers: A list of (SwaptionHelper, expiry_str, tenor_str)
            tuples to be used for calibration.
        term_structure_handle: A handle to the QuantLib yield curve.
        ... (other calibration settings)

    Returns:
        A tuple containing (calibrated_a_params, calibrated_sigma_params).
    """
    ql_eval_date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
    ql.Settings.instance().evaluationDate = ql_eval_date

    if not list_of_calibration_helpers:
        print("No swaption helpers provided for calibration.")
        return None, None

    included_expiries_yrs = sorted(list(set([parse_tenor_to_years(expiry_str) for _, expiry_str, _ in list_of_calibration_helpers])))
    
    final_num_sigma_segments: int
    if isinstance(num_sigma_segments, str) and num_sigma_segments.lower() == 'auto':
        final_num_sigma_segments = len(included_expiries_yrs)
        if final_num_sigma_segments == 0:
            print("Automatic volatility parameter determination failed: No swaption expiries remain.")
            return None, None
        sigma_step_dates = [ql_eval_date + ql.Period(int(y * 365.25), ql.Days) for y in included_expiries_yrs[:-1]]
    else:
        final_num_sigma_segments = int(num_sigma_segments)
        sigma_step_dates = _get_step_dates_from_expiries(ql_eval_date, included_expiries_yrs, final_num_sigma_segments)
    
    a_step_dates = _get_step_dates_from_expiries(ql_eval_date, included_expiries_yrs, num_a_segments)
    unified_step_dates = sorted(list(set(a_step_dates + sigma_step_dates)))
    
    # Ensure initial_sigma is a list if multiple segments are used
    if isinstance(initial_sigma, list):
        sigma_guesses = initial_sigma
    else:
        sigma_guesses = [initial_sigma] * final_num_sigma_segments

    if len(sigma_guesses) != final_num_sigma_segments:
        print(f"Warning: Length of initial_sigma guess ({len(sigma_guesses)}) does not match number of sigma segments ({final_num_sigma_segments}). Using first element for all.")
        sigma_guesses = [sigma_guesses[0]] * final_num_sigma_segments
        
    sigma_handles_initial = [ql.QuoteHandle(ql.SimpleQuote(s)) for s in sigma_guesses]
    if optimize_a:
        reversion_handles_initial = [ql.QuoteHandle(ql.SimpleQuote(initial_a)) for _ in range(num_a_segments)]
    else:
        reversion_handles_initial = [ql.QuoteHandle(ql.SimpleQuote(initial_a))]

    expanded_sigma_handles = _expand_params_to_unified_timeline(sigma_handles_initial, sigma_step_dates, unified_step_dates)
    expanded_reversion_handles = _expand_params_to_unified_timeline(reversion_handles_initial, a_step_dates, unified_step_dates)
    
    model = ql.Gsr(term_structure_handle, unified_step_dates, expanded_sigma_handles, expanded_reversion_handles, 61.0)
    
    calibration_helpers = [h for h, _, _ in list_of_calibration_helpers]
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
        if not original_step_dates: return [expanded_values[0]] 
        indices_to_report = [0]
        original_serials = {d.serialNumber() for d in original_step_dates}
        for i, d_unified in enumerate(unified_step_dates):
            if d_unified.serialNumber() in original_serials:
                indices_to_report.append(i + 1)
        unique_vals = list(dict.fromkeys([expanded_values[i] for i in sorted(list(set(indices_to_report)))]))
        return unique_vals

    final_as = get_unique_calibrated_values(calibrated_expanded_as, a_step_dates, unified_step_dates)
    final_sigmas = get_unique_calibrated_values(calibrated_expanded_sigmas, sigma_step_dates, unified_step_dates)

    return final_as, final_sigmas

#--------------------ORIGINAL CALIBRATION FUNCTION (FOR STANDALONE RUN)--------------------
def calibrate_hull_white_full(
    eval_date: datetime.date,
    zero_curve_df: pd.DataFrame,
    vol_cube_df: pd.DataFrame,
    num_a_segments: int = 1,
    num_sigma_segments: Union[int, str] = 'auto',
    optimize_a: bool = False,
    initial_a: float = 0.01,
    initial_sigma: Union[float, List[float]] = 0.01,
    min_expiry_years: float = 0.0,
    min_tenor_years: float = 0.0,
    pricing_engine_integration_points: int = 64
) -> Tuple[Optional[List[float]], Optional[List[float]], Optional[pd.DataFrame], float]:
    
    ql_eval_date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
    ql.Settings.instance().evaluationDate = ql_eval_date

    term_structure_handle = create_ql_yield_curve(zero_curve_df, eval_date)

    helpers_with_info = prepare_calibration_helpers(
        vol_cube_df, term_structure_handle, min_expiry_years=min_expiry_years, min_tenor_years=min_tenor_years
    )
    if not helpers_with_info:
        print("No valid swaption helpers for calibration.")
        return None, None, None, float('nan')

    final_as, final_sigmas = calibrate_on_subset(
        eval_date, zero_curve_df, helpers_with_info, term_structure_handle,
        num_a_segments, num_sigma_segments, optimize_a, initial_a, initial_sigma,
        pricing_engine_integration_points
    )

    if final_as is None or final_sigmas is None:
        return None, None, None, float('nan')

    # --- Evaluation part (calculating RMSE on the calibration set) ---
    all_params = (final_as if optimize_a else [initial_a]*num_a_segments) + final_sigmas
    num_a_params = num_a_segments if optimize_a else 0
    
    # Rebuild model with final params to get in-sample error
    included_expiries_yrs = sorted(list(set([parse_tenor_to_years(expiry) for _, expiry, _ in helpers_with_info])))
    a_step_dates = _get_step_dates_from_expiries(ql_eval_date, included_expiries_yrs, num_a_segments)
    
    if isinstance(num_sigma_segments, str) and num_sigma_segments.lower() == 'auto':
        final_num_sigma_segments = len(included_expiries_yrs)
        sigma_step_dates = [ql_eval_date + ql.Period(int(y * 365.25), ql.Days) for y in included_expiries_yrs[:-1]]
    else:
        final_num_sigma_segments = int(num_sigma_segments)
        sigma_step_dates = _get_step_dates_from_expiries(ql_eval_date, included_expiries_yrs, final_num_sigma_segments)
        
    unified_step_dates = sorted(list(set(a_step_dates + sigma_step_dates)))
    reversion_quotes = [ql.SimpleQuote(p) for p in final_as]
    sigma_quotes = [ql.SimpleQuote(p) for p in final_sigmas]
    
    expanded_reversion_handles = _expand_params_to_unified_timeline([ql.QuoteHandle(q) for q in reversion_quotes], a_step_dates, unified_step_dates)
    expanded_sigma_handles = _expand_params_to_unified_timeline([ql.QuoteHandle(q) for q in sigma_quotes], sigma_step_dates, unified_step_dates)
    
    final_model = ql.Gsr(term_structure_handle, unified_step_dates, expanded_sigma_handles, expanded_reversion_handles, 61.0)
    final_engine = ql.Gaussian1dSwaptionEngine(final_model, pricing_engine_integration_points, 7.0, True, False, term_structure_handle)

    results_data = []
    squared_errors = []
    for helper, expiry_str, tenor_str in helpers_with_info:
        helper.setPricingEngine(final_engine)
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
        results_data.append({'ExpiryStr': expiry_str, 'TenorStr': tenor_str, 'MarketVol': market_vol_bps, 'ModelVol': model_vol_bps, 'Difference_bps': error_bps})
        
    results_df = pd.DataFrame(results_data)
    results_df['Expiry'] = results_df['ExpiryStr'].apply(parse_tenor_to_years)
    results_df['Tenor'] = results_df['TenorStr'].apply(parse_tenor_to_years)
    rmse_bps: float = np.sqrt(np.mean(squared_errors)) if squared_errors else float('nan')
    print(f"Daily In-Sample RMSE for {eval_date}: {rmse_bps:.4f} bps")

    return final_as, final_sigmas, results_df, rmse_bps

if __name__ == '__main__':
    os.makedirs(FOLDER_RESULTS_TRADITIONAL, exist_ok=True)
    CALIBRATION_SETTINGS = {
        "num_a_segments": 1, "num_sigma_segments": 7, "optimize_a": True,
        "initial_a": 0.02,
        "initial_sigma": [0.0002, 0.0002, 0.00017, 0.00017, 0.00017, 0.00017, 0.00017],
        "pricing_engine_integration_points": 32,
        "min_expiry_years": 2.0, "min_tenor_years": 2.0
    }
    try:
        vol_files_folder = os.path.join(FOLDER_VOLATILITY_CUBES, 'xlsx')
        if not os.path.exists(FOLDER_ZERO_CURVES) or not os.path.exists(vol_files_folder):
             raise FileNotFoundError("Data folders not found.")
        zero_curve_dates: Set[str] = {f.replace('.csv', '') for f in os.listdir(FOLDER_ZERO_CURVES) if f.endswith('.csv')}
        vol_cube_dates: Set[str] = {f.replace('.xlsx', '') for f in os.listdir(vol_files_folder) if f.endswith('.xlsx')}
        available_dates: List[str] = sorted(list(zero_curve_dates.intersection(vol_cube_dates)))

        if not available_dates:
            print("No matching data found.")
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

                    calibrated_as, calibrated_sigmas, results_df, rmse_bps = calibrate_hull_white_full(
                        eval_date=eval_date, zero_curve_df=zero_curve_df, vol_cube_df=vol_cube_df, **CALIBRATION_SETTINGS
                    )
                    end_time = time.monotonic()
                    elapsed = end_time - start_time
                    print(f"Finished calibration for {date_str} in {elapsed:.2f} seconds.")
                    
                    if results_df is not None and not results_df.empty:
                        day_results_df = results_df.copy()
                        day_results_df['EvaluationDate'], day_results_df['DailyRMSE_bps'] = eval_date, rmse_bps
                        day_results_df['PredictionTimeSeconds'] = elapsed
                        if calibrated_as:
                            for i, a_val in enumerate(calibrated_as): day_results_df[f'a_{i+1}'] = a_val
                        if calibrated_sigmas:
                            for i, s_val in enumerate(calibrated_sigmas): day_results_df[f'sigma_{i+1}'] = s_val
                        all_results_dfs.append(day_results_df)

                except Exception as e:
                    import traceback
                    print(f"\n--- ERROR during calibration for {date_str}: {e} ---")
                    traceback.print_exc()
                print(f"{'='*25} FINISHED CALIBRATION FOR: {date_str} {'='*25}")
            
            if all_results_dfs:
                master_results_df = pd.concat(all_results_dfs, ignore_index=True)
                results_save_path = os.path.join(FOLDER_RESULTS_TRADITIONAL, 'evaluation_results_standalone.csv')
                master_results_df.to_csv(results_save_path, index=False)
                print(f"\n\n--- Comprehensive evaluation results saved to: {results_save_path} ---")
                last_eval_date = all_results_dfs[-1]['EvaluationDate'].iloc[0]
                print(f"\n--- Plotting calibration results for last day: {last_eval_date} ---")
                plot_calibration_results(all_results_dfs[-1], last_eval_date, FOLDER_RESULTS_TRADITIONAL)
    except Exception as e:
        import traceback
        print(f"\nAn unexpected error occurred during the main process: {e}")
        traceback.print_exc()