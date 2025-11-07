"""
Script for Visualizing Hull-White Calibration Comparison Results

Objective:
This script reads the results from the comparison of four distinct calibration
strategies (NN, LM-Static, LM-Pure-Rolling, LM-Adaptive-Anchor), calculates
summary metrics, and produces a comprehensive set of visualizations for analysis.

The script first calculates daily metrics on-the-fly, then generates two
categories of visualizations:
1.  High-level daily performance summaries (RMSE, parameter evolution).
2.  Granular, instrument-level analysis (heatmaps, scatter plots, bucketed errors).

All generated tables (as .txt and .csv) and plots (as .png) are saved in the
results/comparison directory.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
import datetime
import QuantLib as ql
from tqdm.auto import tqdm
tqdm.pandas()

# --- CONFIGURATION ---
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
COMPARISON_DIR = os.path.join(RESULTS_DIR, 'comparison')

# Input files
SWAPTION_CSV = os.path.join(COMPARISON_DIR, 'per_swaption_holdout_results.csv')
SUMMARY_CSV = os.path.join(COMPARISON_DIR, 'daily_summary_results.csv')
SUMMARY_CSV_BLACK = os.path.join(COMPARISON_DIR, 'daily_summary_results_black.csv')

# Data Folders needed for on-the-fly vega calculation
FOLDER_ZERO_CURVES = os.path.join(DATA_DIR, 'EUR ZERO CURVE')

# --- PLOT STYLING ---
# Central dictionary to define styles AND naming conventions for all strategies
STRATEGIES = {
    'NN': {
        'name': 'Neural Network',
        'color': cm.get_cmap('coolwarm')(0.15),
        'style': '-',
        'output_prefix': 'BlackVol_NN'
    },
    'LM_Static': {
        'name': 'LM (Static)',
        'color': cm.get_cmap('coolwarm')(0.75),
        'style': ':',
        'output_prefix': 'BlackVol_LM_Static'
    },
    'LM_Pure_Rolling': {
        'name': 'LM (Pure Rolling)',
        'color': cm.get_cmap('coolwarm')(0.99),
        'style': '--',
        'output_prefix': 'BlackVol_LM_Pure_Rolling'
    },
    'LM_Adaptive_Anchor': {
        'name': 'LM (Adaptive Anchor)',
        'color': cm.get_cmap('coolwarm')(0.5),
        'style': '-.',
        'output_prefix': 'BlackVol_LM_Adaptive_Anchor'
    }
}
sns.set_theme(style="whitegrid", palette=[s['color'] for s in STRATEGIES.values()])
sns.set_theme(style="whitegrid", palette=[s['color'] for s in STRATEGIES.values()])

# --- HELPER FUNCTIONS FOR VEGA CALCULATION & PARSING ---
def parse_tenor(tenor_str: str) -> ql.Period:
    """
    Parses a tenor string (e.g., '10YR', '6MO') into a QuantLib Period object.

    Args:
        tenor_str (str): The tenor string to be parsed.

    Returns:
        ql.Period: The parsed tenor as a QuantLib Period object.

    Raises:
        ValueError: If the parsing fails due to an invalid tenor string.
    """
    try:
        tenor_str = str(tenor_str).strip().upper()
        if 'YR' in tenor_str: return ql.Period(int(tenor_str.replace('YR', '')), ql.Years)
        if 'MO' in tenor_str: return ql.Period(int(tenor_str.replace('MO', '')), ql.Months)
    except (ValueError, AttributeError):
        raise ValueError(f"Could not parse tenor string: {tenor_str}")
    raise ValueError(f"Could not parse tenor string: {tenor_str}")

def parse_tenor_to_years(tenor_str: str) -> float:
    """
    Parses a tenor string (e.g., '10YR', '6MO') into a float representing years.

    Args:
        tenor_str (str): The tenor string to be parsed.

    Returns:
        float: The parsed tenor in years. If the parsing fails, returns np.nan.

    Raises:
        ValueError: If the parsing fails due to an invalid tenor string.
    """
    try:
        tenor_str = str(tenor_str).strip().upper()
        if 'YR' in tenor_str: return float(tenor_str.replace('YR', ''))
        if 'MO' in tenor_str: return float(tenor_str.replace('MO', '')) / 12.0
    except (ValueError, AttributeError):
        return np.nan
    raise ValueError(f"Could not parse tenor string to years: {tenor_str}")

def create_ql_yield_curve(
    zero_curve_df: pd.DataFrame, eval_date: datetime.date) -> ql.RelinkableYieldTermStructureHandle:
    """
    Creates a QuantLib yield curve from a pandas DataFrame containing a time series of daily zero rates.

    Args:
        zero_curve_df (pd.DataFrame): A pandas DataFrame containing a time series of daily zero rates.
        eval_date (datetime.date): The date at which the yield curve should be evaluated.

    Returns:
        ql.RelinkableYieldTermStructureHandle: A QuantLib yield curve handle constructed from the input time series.
    """
    ql_eval_date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
    dates = [ql_eval_date] + [ql.Date(d.day, d.month, d.year) for d in pd.to_datetime(zero_curve_df['Date'])]
    rates = [zero_curve_df['ZeroRate'].iloc[0]] + zero_curve_df['ZeroRate'].tolist()
    term_structure = ql.ZeroCurve(dates, rates, ql.Actual365Fixed(), ql.TARGET(), ql.Linear(), ql.Continuous, ql.Annual)
    term_structure.enableExtrapolation()
    handle = ql.RelinkableYieldTermStructureHandle()
    handle.linkTo(term_structure)
    return handle

def calculate_swaption_vega(swaption_row: pd.Series, yield_curve_handle: ql.RelinkableYieldTermStructureHandle) -> float:
    """
    Calculates the vega value of a swaption given the market normal volatility and yield curve.

    Args:
        swaption_row (pd.Series): A pandas Series containing the details of the swaption.
        yield_curve_handle (ql.RelinkableYieldTermStructureHandle): A QuantLib yield curve handle.

    Returns:
        float: The vega value of the swaption. If any error occurs, returns 0.0.
    """
    try:
        ql_eval_date = ql.Settings.instance().evaluationDate
        swap_index = ql.Euribor6M(yield_curve_handle)
        expiry_period = parse_tenor(swaption_row['ExpiryStr'])
        tenor_period = parse_tenor(swaption_row['TenorStr'])
        dummy_swap = ql.MakeVanillaSwap(tenor_period, swap_index, 0.0, expiry_period)
        forward_rate = dummy_swap.fairRate()
        if forward_rate <= 0: return 0.0
        underlying_swap = ql.MakeVanillaSwap(tenor_period, swap_index, forward_rate, expiry_period)
        exercise_date = ql.TARGET().advance(ql_eval_date, expiry_period)
        exercise = ql.EuropeanExercise(exercise_date)
        swaption_obj = ql.Swaption(underlying_swap, exercise)
        market_normal_vol = swaption_row['MarketVol_bps'] / 10000.0
        vol_handle = ql.QuoteHandle(ql.SimpleQuote(market_normal_vol))
        pricing_engine = ql.BachelierSwaptionEngine(yield_curve_handle, vol_handle)
        swaption_obj.setPricingEngine(pricing_engine)
        vega_value = swaption_obj.vega()
        return vega_value if not np.isnan(vega_value) else 0.0
    except Exception:
        return 0.0

def calculate_daily_rmse_metrics(df_swaption: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates daily Root Mean Squared Error (RMSE) metrics for the given strategies.

    Parameters:
        df_swaption (pd.DataFrame): A pandas DataFrame containing the results of the daily swaption calibration.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the daily RMSE metrics for each strategy.

    Notes:
        The RMSE metrics are calculated using both unweighted and vega-weighted approaches. The vega-weighted approach uses the vega values of the swaptions to weight the squared errors.
    """
    print("--- Calculating daily RMSE metrics (unweighted and vega-weighted) ---")
    
    error_cols = {
        'NN': 'NN_Error_bps',
        'LM_Static': 'LM_Static_Error_bps',
        'LM_Pure_Rolling': 'LM_Pure_Rolling_Error_bps',
        'LM_Adaptive_Anchor': 'LM_Adaptive_Anchor_Error_bps'
    }

    unique_dates = pd.to_datetime(df_swaption['EvaluationDate']).unique()
    yield_curves = {}
    for dt in tqdm(unique_dates, desc="Loading Curves"):
        eval_date = dt.date()
        date_str = eval_date.strftime('%d.%m.%Y')
        zero_path = os.path.join(FOLDER_ZERO_CURVES, f"{date_str}.csv")
        if os.path.exists(zero_path):
            zero_df = pd.read_csv(zero_path, parse_dates=['Date'])
            yield_curves[eval_date] = create_ql_yield_curve(zero_df, eval_date)

    daily_results = []
    grouped = df_swaption.groupby('EvaluationDate')
    for date_str, group in tqdm(grouped, total=len(grouped), desc="Calculating Daily Metrics"):
        eval_date = pd.to_datetime(date_str).date()
        if eval_date not in yield_curves: continue
        
        yield_curve_handle = yield_curves[eval_date]
        ql.Settings.instance().evaluationDate = ql.Date(eval_date.day, eval_date.month, eval_date.year)
        vegas = group.apply(lambda row: calculate_swaption_vega(row, yield_curve_handle), axis=1)
        
        day_metrics = {'Date': eval_date}
        total_vega = vegas.sum()
        
        for key, col_name in error_cols.items():
            if col_name in group.columns:
                day_metrics[f'RMSE_{key}_Unweighted'] = np.sqrt((group[col_name]**2).mean())
                if total_vega > 0:
                    mse_weighted = ((group[col_name]**2) * vegas).sum() / total_vega
                    day_metrics[f'RMSE_{key}_VegaWeighted'] = np.sqrt(mse_weighted)
                else:
                    day_metrics[f'RMSE_{key}_VegaWeighted'] = np.nan
        daily_results.append(day_metrics)
        
    return pd.DataFrame(daily_results)

# --- VISUALIZATION FUNCTIONS ---
def generate_summary_tables(df: pd.DataFrame, save_dir: str):
    """
    Generates summary tables for the given DataFrame and saves them to csv and txt files in the given directory.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the summary data.
        save_dir (str): The directory where the summary tables will be saved.

    Returns:
        None

    Notes:
        The generated summary tables are:

        Table 1: Aggregate Out-of-Sample Performance Metrics (RMSE in bps)
        Table 2: Average Prediction Time
        Table 3: Parameter Stability Statistics

    """
    print("\n" + "="*80)
    print(" GENERATING SUMMARY TABLES ".center(80, "="))
    print("="*80)

    # --- Table 1: Performance Metrics (RMSE) ---
    rmse_cols = [col for col in df.columns if 'RMSE' in col]
    if rmse_cols:
        rmse_stats = df[rmse_cols].agg(['mean', 'std', 'median', 'min', 'max']).T
        rmse_stats.columns = ['Mean', 'Std Dev', 'Median', 'Min', 'Max']

        index_map = {f'RMSE_{key}_Unweighted': f'{props["name"]} (Unweighted)' for key, props in STRATEGIES.items()}
        index_map.update({f'RMSE_{key}_VegaWeighted': f'{props["name"]} (Vega-Weighted)' for key, props in STRATEGIES.items()})

        original_index = rmse_stats.index
        new_index = [index_map.get(item, item) for item in original_index]
        rmse_stats.index = new_index
        rmse_stats.sort_index(inplace=True)

        print("\n--- Table 1: Aggregate Out-of-Sample Performance Metrics (RMSE in bps) ---")
        print(rmse_stats.to_string(float_format="%.4f"))

        rmse_stats.to_csv(os.path.join(save_dir, 'table1_performance_metrics.csv'))
        with open(os.path.join(save_dir, 'table1_performance_metrics.txt'), 'w') as f:
            f.write("Table 1: Aggregate Out-of-Sample Performance Metrics (RMSE in bps)\n")
            f.write(rmse_stats.to_string(float_format="%.4f"))
        print(f"\nSaved Table 1 to '{save_dir}'")
    else:
        print("\nSkipping Table 1: No RMSE columns found in the summary DataFrame.")

    # --- Table 2: Average Prediction Time ---
    df = pd.read_csv(SUMMARY_CSV, parse_dates=['Date'])
    time_cols = [col for col in df.columns if col.startswith('Time_') and col.endswith('_sec')]
    if time_cols:
        time_stats = df[time_cols].mean().to_frame(name='Average Time (sec)')
        time_stats['Std Dev'] = df[time_cols].std().to_frame(name='Std Dev (sec)')
        time_index_map = {f'Time_{key}_sec': props["name"] for key, props in STRATEGIES.items()}
        
        original_time_index = time_stats.index
        new_time_index = [time_index_map.get(item, item) for item in original_time_index]
        time_stats.index = new_time_index
        time_stats.sort_index(inplace=True, ascending=False)

        print("\n--- Table 2: Average Prediction Time ---")
        print(time_stats.to_string(float_format="%.4f"))

        time_stats.to_csv(os.path.join(save_dir, 'table2_prediction_time.csv'))
        with open(os.path.join(save_dir, 'table2_prediction_time.txt'), 'w') as f:
            f.write("Table 2: Average Prediction Time\n")
            f.write(time_stats.to_string(float_format="%.4f"))
        print(f"\nSaved Table 2 to '{save_dir}'")
    else:
        print("\nSkipping Table 2: No time columns found in the summary DataFrame.")

    # --- Table 3: Parameter Stability Statistics ---
    print("\n--- Table 3: Parameter Stability Statistics ---")
    all_param_stats = []
    
    for key, props in STRATEGIES.items():
        param_cols = [col for col in df.columns if col.startswith(f'{key}_a_') or col.startswith(f'{key}_sigma_')]
        if not param_cols:
            continue
        param_stats = df[param_cols].describe().T # Transpose for better readability
        param_stats['Strategy'] = props['name']
        all_param_stats.append(param_stats)

    if all_param_stats:
        # Concatenate all stats into one multi-indexed DataFrame
        final_param_table = pd.concat(all_param_stats)
        final_param_table = final_param_table.set_index('Strategy', append=True).swaplevel(0, 1)
        final_param_table.sort_index(inplace=True)
        
        desired_columns = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
        final_param_table = final_param_table[[col for col in desired_columns if col in final_param_table.columns]]
        print(final_param_table.to_string(float_format="%.6f"))
        
        # Save the table to csv and txt files
        final_param_table.to_csv(os.path.join(save_dir, 'table3_parameter_stability.csv'))
        with open(os.path.join(save_dir, 'table3_parameter_stability.txt'), 'w') as f:
            f.write("Table 3: Parameter Stability Statistics\n")
            f.write(final_param_table.to_string(float_format="%.6f"))
        print(f"\nSaved Table 3 to '{save_dir}'")
    else:
        print("\nSkipping Table 3: No parameter columns found in the summary DataFrame.")

def plot_daily_rmse(df: pd.DataFrame, save_path: str):
    """
    Plot: Daily RMSE Over Time (Unweighted vs. Vega-Weighted)

    Plots the daily out-of-sample RMSE for all models, showing both the
    vega-weighted and simple average errors in log-normal (Black) volatility terms.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the daily comparison results for Black volatility.
    save_path : str
        The path where the plot will be saved.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for key, props in STRATEGIES.items():
        if f'RMSE_{key}_Unweighted' in df.columns:
            ax.plot(df['Date'], df[f'RMSE_{key}_Unweighted'], label=f"{props['name']} (Unweighted)", 
                    color=props['color'], linestyle='-', marker='o', markersize=4, zorder=10)
        if f'RMSE_{key}_VegaWeighted' in df.columns:
            ax.plot(df['Date'], df[f'RMSE_{key}_VegaWeighted'], label=f"{props['name']} (Vega-Weighted)", 
                    color=props['color'], linestyle='--')
            
    ax.set_title('Daily RMSE Over Time (Unweighted vs. Vega-Weighted)', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('RMSE (bps)', fontsize=12)
    ax.legend(ncol=2)
    ax.grid(True, which='both', linestyle='--')
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved Plot 1: {save_path}")
    
def plot_daily_rmse_black(df: pd.DataFrame, save_path: str):
    """
    Plot: Daily Out-of-Sample RMSE Over Time (Black Volatility) - Updated for all strategies

    Plots the daily out-of-sample RMSE for all models, showing both the
    vega-weighted and simple average errors in log-normal (Black) volatility terms.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the daily comparison results for Black volatility.
    save_path : str
        The path where the plot will be saved.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    for key, props in STRATEGIES.items():
        # Column names are constructed dynamically based on the strategy prefix
        weighted_col = f"{props['output_prefix']}_Vega_Weighted"
        simple_avg_col = f"{props['output_prefix']}_Simple_Avg"

        if weighted_col in df.columns:
            ax.plot(df['Date'], df[weighted_col], label=f"{props['name']} (Vega Weighted)",
                    color=props['color'], marker='o', linestyle='-', markersize=4, zorder=10)
        
        if simple_avg_col in df.columns:
            ax.plot(df['Date'], df[simple_avg_col], label=f"{props['name']} (Simple Average)",
                    color=props['color'], linestyle=':', marker=None)

    ax.set_title('Daily Out-of-Sample RMSE Over Time (Black Volatility)', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('RMSE (Black Volatility) in %', fontsize=12)
    ax.legend(ncol=2)
    ax.grid(True, which='both', linestyle='--')
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved Black Volatility RMSE Plot: {save_path}")

def plot_rmse_distribution(df: pd.DataFrame, save_path: str):
    """
    Plot: Distribution of Daily RMSE (Unweighted vs. Vega-Weighted)

    Plots a violin plot of the distribution of daily RMSE for all models, comparing
    the distribution of the unweighted and vega-weighted results.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the daily comparison results for all models.
    save_path : str
        The path where the plot will be saved.

    Returns
    -------
    None
    """
    unweighted_cols = {f'RMSE_{key}_Unweighted': props['name'] for key, props in STRATEGIES.items() if f'RMSE_{key}_Unweighted' in df.columns}
    weighted_cols = {f'RMSE_{key}_VegaWeighted': props['name'] for key, props in STRATEGIES.items() if f'RMSE_{key}_VegaWeighted' in df.columns}
    
    df_unweighted = df[list(unweighted_cols.keys())].melt(var_name='Variable', value_name='Daily RMSE (bps)')
    df_unweighted['Weighting'] = 'Unweighted'
    df_unweighted['Model'] = df_unweighted['Variable'].map(unweighted_cols)

    df_weighted = df[list(weighted_cols.keys())].melt(var_name='Variable', value_name='Daily RMSE (bps)')
    df_weighted['Weighting'] = 'Vega-Weighted'
    df_weighted['Model'] = df_weighted['Variable'].map(weighted_cols)

    df_melted = pd.concat([df_unweighted, df_weighted])

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.violinplot(data=df_melted, x='Model', y='Daily RMSE (bps)', hue='Weighting', 
                   ax=ax, cut=0, split=True, palette={'Unweighted': '#a1c4fd', 'Vega-Weighted': '#4a89dc'}, order=[p['name'] for p in STRATEGIES.values()])
    ax.set_title('Distribution of Daily RMSE (Unweighted vs. Vega-Weighted)', fontsize=16)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Daily RMSE (bps)', fontsize=12)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved Plot 2: {save_path}")

def plot_parameter_evolution(df: pd.DataFrame, save_path: str):
    """
    Plot: Comparison of Model Parameter Evolution

    Plots the evolution of the calibrated model parameters over time, comparing
    the different models. The plot consists of a 4x2 grid of subplots, with each
    subplot showing the evolution of one parameter. The x-axis represents the date of
    the calibration, and the y-axis represents the value of the parameter.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the daily comparison results for all models.
    save_path : str
        The path where the plot will be saved.

    Returns
    -------
    None
    """
    param_stems = sorted(list(set([
        '_'.join(col.split('_')[-2:]) for col in df.columns
        if ('_a_' in col or '_sigma_' in col) and col.startswith(tuple(STRATEGIES.keys()))
    ])))

    if not param_stems:
        print("Could not generate Plot 3: No parameter columns found.")
        return

    fig, axes = plt.subplots(4, 2, figsize=(18, 20), sharex=True)
    fig.suptitle('Comparison of Model Parameter Evolution', fontsize=20, y=0.99)

    for ax, stem in zip(axes.flat, param_stems):
        for key, props in STRATEGIES.items():
            param_col = f"{key}_{stem}"
            if param_col in df.columns:
                ax.plot(df['Date'], df[param_col], label=props['name'],
                        color=props['color'], linestyle=props['style'], alpha=0.9)

        ax.set_title(f"Evolution of Parameter: {stem}", fontsize=14)
        ax.set_ylabel('Parameter Value')
        ax.grid(True, which='both', linestyle='--')
        ax.legend()

    for ax in axes[3, :]: # This selects the last row of axes
        ax.set_xlabel('Date', fontsize=12)

    fig.autofmt_xdate()
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved Plot 3: {save_path}")

def plot_volatility_surface(df: pd.DataFrame, save_path_prefix: str):
    """
    Plots the volatility surface reconstruction for a given evaluation date.

    Parameters
    ----------
    df (pd.DataFrame): DataFrame containing the results of the Hull-White calibration process.
    save_path_prefix (str): Prefix of the path where the plot will be saved.

    Returns
    -------
    None
    """
    if df.empty: return
    latest_date = pd.to_datetime(df['EvaluationDate']).max()
    date_str = latest_date.strftime('%Y-%m-%d')
    day_df = df[df['EvaluationDate'] == date_str].copy()
    if day_df.empty: return

    day_df['Expiry'] = day_df['ExpiryStr'].apply(parse_tenor_to_years)
    day_df['Tenor'] = day_df['TenorStr'].apply(parse_tenor_to_years)
    day_df.dropna(subset=['Expiry', 'Tenor'], inplace=True)

    fig = plt.figure(figsize=(24, 8))
    fig.suptitle(f'Volatility Surface Reconstruction on Hold-Out Set ({date_str})', fontsize=16)
    
    titles = ['Market Volatility (Ground Truth)', STRATEGIES['NN']['name'], STRATEGIES['LM_Adaptive_Anchor']['name']]
    z_vars = ['MarketVol_bps', 'NN_ModelVol_bps', 'LM_Adaptive_Anchor_ModelVol_bps']

    for i in range(3):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        ax.set_title(titles[i])
        if z_vars[i] in day_df.columns:
            ax.plot_trisurf(day_df['Expiry'], day_df['Tenor'], day_df[z_vars[i]], cmap=cm.coolwarm, antialiased=True)
            ax.set_xlabel('Expiry (Years)'); ax.set_ylabel('Tenor (Years)'); ax.set_zlabel('Volatility (bps)')
            ax.view_init(elev=30, azim=-120)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = f"{save_path_prefix}_{date_str}.png"
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved Plot 5: {save_path}")

def plot_error_heatmaps(df: pd.DataFrame, save_path: str):
    """
    Generates a 2x2 grid of heatmaps representing the mean prediction errors of the Hull-White calibration models across the volatility surface.

    Parameters
    ----------
    df (pd.DataFrame): DataFrame containing the results of the Hull-White calibration process.
    save_path (str): Path where the plot will be saved.

    Returns
    -------
    None
    """
    df_copy = df.copy()
    df_copy['Expiry'] = df_copy['ExpiryStr'].apply(parse_tenor_to_years)
    df_copy['Tenor'] = df_copy['TenorStr'].apply(parse_tenor_to_years)
    df_copy.dropna(subset=['Expiry', 'Tenor'], inplace=True)

    expiry_bins = pd.cut(df_copy['Expiry'], bins=np.arange(0, 31, 5), right=False)
    tenor_bins = pd.cut(df_copy['Tenor'], bins=np.arange(0, 31, 5), right=False)
    
    error_cols = {
        'NN': 'NN_Error_bps', 'LM_Static': 'LM_Static_Error_bps',
        'LM_Pure_Rolling': 'LM_Pure_Rolling_Error_bps', 'LM_Adaptive_Anchor': 'LM_Adaptive_Anchor_Error_bps'
    }
    
    pivots = {key: df_copy.pivot_table(index=expiry_bins, columns=tenor_bins, values=col, aggfunc='mean')
              for key, col in error_cols.items() if col in df_copy.columns}

    if not pivots:
        print("Could not generate Plot 6: No error columns found in swaption data.")
        return
        
    vmax = max(abs(p.min().min()) for p in pivots.values()) + max(p.max().max() for p in pivots.values()) / 2
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), sharex=True, sharey=True)
    fig.suptitle('Mean Prediction Errors (Model - Market) in bps across Volatility Surface', fontsize=16)

    for ax, key in zip(axes.flat, STRATEGIES.keys()):
        if key in pivots:
            sns.heatmap(pivots[key], ax=ax, cmap='coolwarm', annot=True, fmt=".2f", vmin=-vmax, vmax=vmax)
            ax.set_title(STRATEGIES[key]['name'])
            ax.set_xlabel('Tenor Bins'); ax.set_ylabel('Expiry Bins')
        else: ax.set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved Plot 6: {save_path}")
    
def plot_error_heatmaps_weighted(df: pd.DataFrame, save_path: str):
    """
    Generates a 2x2 grid of heatmaps representing the vega-weighted mean squared error contribution of the Hull-White calibration models across the volatility surface.

    Parameters
    ----------
    df (pd.DataFrame): DataFrame containing the results of the Hull-White calibration process.
    save_path (str): Path where the plot will be saved.

    Returns
    -------
    None
    """
    if 'Vega' not in df.columns or df['Vega'].isnull().all():
        print("Could not generate weighted heatmaps: 'Vega' column not found or is empty.")
        return

    df_copy = df.copy()
    df_copy['Expiry'] = df_copy['ExpiryStr'].apply(parse_tenor_to_years)
    df_copy['Tenor'] = df_copy['TenorStr'].apply(parse_tenor_to_years)
    df_copy.dropna(subset=['Expiry', 'Tenor', 'Vega'], inplace=True)
    
    expiry_bins = pd.cut(df_copy['Expiry'], bins=np.arange(0, 31, 5), right=False)
    tenor_bins = pd.cut(df_copy['Tenor'], bins=np.arange(0, 31, 5), right=False)
    
    error_cols = {
        'NN': 'NN_Error_bps', 'LM_Static': 'LM_Static_Error_bps',
        'LM_Pure_Rolling': 'LM_Pure_Rolling_Error_bps', 'LM_Adaptive_Anchor': 'LM_Adaptive_Anchor_Error_bps'
    }

    pivots = {}
    for key, col in error_cols.items():
        if col in df_copy.columns:
            df_copy[f'weighted_error_{key}'] = df_copy[col] * df_copy['Vega'] / df_copy['Vega'].mean()
            pivots[key] = df_copy.pivot_table(index=expiry_bins, columns=tenor_bins, values=f'weighted_error_{key}', aggfunc='mean')

    if not pivots:
        print("Could not generate Plot 6b: No error columns found for weighted heatmap.")
        return
        
    vmax = max(p.max().max() for p in pivots.values())
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), sharex=True, sharey=True)
    fig.suptitle('Vega-Weighted Mean Squared Error Contribution across Volatility Surface', fontsize=16)

    for ax, key in zip(axes.flat, STRATEGIES.keys()):
        if key in pivots:
            sns.heatmap(pivots[key], ax=ax, cmap='Reds', annot=True, fmt=".2f", vmin=0, vmax=vmax)
            ax.set_title(STRATEGIES[key]['name'])
            ax.set_xlabel('Tenor Bins'); ax.set_ylabel('Expiry Bins')
        else: ax.set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved Plot 6b: {save_path}")

def plot_scatter_comparison(df: pd.DataFrame, save_path: str):
    """
    Generates a 2x2 grid of scatter plots comparing the model implied volatilities to the observed market volatilities across all hold-out swaptions.

    Parameters
    ----------
    df (pd.DataFrame): A pandas DataFrame containing the results of the Hull-White calibration process.
    save_path (str): The path where the plot will be saved.

    Returns
    -------
    None
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex=True, sharey=True)
    fig.suptitle('Model vs. Market Volatility on All Hold-Out Swaptions', fontsize=16)

    model_vol_cols = {
        'NN': 'NN_ModelVol_bps', 'LM_Static': 'LM_Static_ModelVol_bps',
        'LM_Pure_Rolling': 'LM_Pure_Rolling_ModelVol_bps', 'LM_Adaptive_Anchor': 'LM_Adaptive_Anchor_ModelVol_bps'
    }
    
    max_val = df['MarketVol_bps'].max() * 1.05
    min_val = df['MarketVol_bps'].min() * 0.95
    
    for ax, key in zip(axes.flat, STRATEGIES.keys()):
        col_name = model_vol_cols[key]
        if col_name in df.columns:
            ax.scatter(df['MarketVol_bps'], df[col_name], alpha=0.3, color=STRATEGIES[key]['color'], s=10)
            ax.set_title(STRATEGIES[key]['name'])
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='y=x (Perfect Fit)')
            ax.set_xlabel('Market Volatility (bps)'); ax.set_ylabel('Model Volatility (bps)')
            ax.legend(); ax.grid(True)
        else: ax.set_visible(False)

    axes[0, 0].set_xlim(min_val, max_val)
    axes[0, 0].set_ylim(min_val, max_val)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved Plot 10: {save_path}")

def plot_error_by_bucket(df: pd.DataFrame, save_path: str, bucket_col: str):
    """
    Generates a boxplot of the prediction errors across different buckets of a chosen feature (e.g. expiry, tenor, market volatility).

    Parameters
    ----------
    df (pd.DataFrame): A pandas DataFrame containing the results of the Hull-White calibration process.
    save_path (str): The path where the plot will be saved.
    bucket_col (str): The column name of the feature to bucket the data by.

    Returns
    -------
    None
    """
    df_copy = df.copy()
    if bucket_col == 'Expiry':
        df_copy['Bucket'] = df_copy['ExpiryStr'].apply(parse_tenor_to_years)
        title = 'Error Distribution by Swaption Expiry Bucket'
    elif bucket_col == 'Tenor':
        df_copy['Bucket'] = df_copy['TenorStr'].apply(parse_tenor_to_years)
        title = 'Error Distribution by Swaption Tenor Bucket'
    elif bucket_col == 'Volatility':
        try:
            labels = ['Low Vol (0-25%)', 'Mid Vol (25-75%)', 'High Vol (75-100%)']
            df_copy['Bucket'] = pd.qcut(df_copy['MarketVol_bps'], q=[0, 0.25, 0.75, 1.0], labels=labels)
            title = 'Error Distribution by Market Volatility Level'
        except (ValueError, IndexError):
            print(f"Could not generate plot for {bucket_col}: Not enough unique points for quantile binning.")
            return
    else: return
    
    if bucket_col != 'Volatility':
        bins = [0, 3, 7, 31]; labels = ['Short (0-3Y)', 'Medium (3-7Y)', 'Long (7Y+)']
        df_copy['Bucket'] = pd.cut(df_copy['Bucket'], bins=bins, labels=labels, right=False)

    error_cols = [
        'NN_Error_bps', 'LM_Static_Error_bps', 
        'LM_Pure_Rolling_Error_bps', 'LM_Adaptive_Anchor_Error_bps'
    ]
    
    df_melted = df_copy.melt(id_vars=['Bucket'], value_vars=error_cols, var_name='Model', value_name='Error (bps)')
    
    model_map = {
        'NN_Error_bps': STRATEGIES['NN']['name'], 'LM_Static_Error_bps': STRATEGIES['LM_Static']['name'],
        'LM_Pure_Rolling_Error_bps': STRATEGIES['LM_Pure_Rolling']['name'], 'LM_Adaptive_Anchor_Error_bps': STRATEGIES['LM_Adaptive_Anchor']['name']
    }
    df_melted['Model'] = df_melted['Model'].map(model_map)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.boxplot(data=df_melted, x='Bucket', y='Error (bps)', hue='Model', 
                palette=[p['color'] for p in STRATEGIES.values()], ax=ax, hue_order=[p['name'] for p in STRATEGIES.values()])
    ax.axhline(0, color='black', linestyle='--')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(f'{bucket_col} Bucket/Quantile', fontsize=12)
    ax.set_ylabel('Error (bps)', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved Plot for bucket '{bucket_col}': {save_path}")

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    try:
        print(f"--- Loading per-swaption data from {SWAPTION_CSV} ---")
        swaption_df = pd.read_csv(SWAPTION_CSV)
        print("Per-swaption data loaded successfully.")

        summary_df_calculated = calculate_daily_rmse_metrics(swaption_df)
        summary_df_calculated['Date'] = pd.to_datetime(summary_df_calculated['Date'])
        
        # --- Generate Part 1 Visuals ---
        generate_summary_tables(summary_df_calculated, COMPARISON_DIR)
        plot_daily_rmse(summary_df_calculated, os.path.join(COMPARISON_DIR, 'plot1_daily_rmse_combined.png'))
        plot_rmse_distribution(summary_df_calculated, os.path.join(COMPARISON_DIR, 'plot2_rmse_distribution_combined.png'))
        
        try:
            summary_df_from_file = pd.read_csv(SUMMARY_CSV, parse_dates=['Date'])
            plot_parameter_evolution(summary_df_from_file, os.path.join(COMPARISON_DIR, 'plot3_parameter_evolution.png'))
        except FileNotFoundError:
            print(f"INFO: Summary file '{SUMMARY_CSV}' not found. Skipping parameter evolution plot.")
            
        try:
            summary_df_black = pd.read_csv(SUMMARY_CSV_BLACK, parse_dates=['Date'])
            plot_daily_rmse_black(summary_df_black, os.path.join(COMPARISON_DIR, 'plot4_daily_rmse_black_vol.png'))
        except FileNotFoundError:
            print(f"\nINFO: Black volatility summary file not found at '{SUMMARY_CSV_BLACK}'. Skipping corresponding plot.")

        # --- PRE-CALCULATE VEGAS FOR ALL GRANULAR PLOTS ---
        print("\n--- Pre-calculating vegas for all swaptions for granular analysis ---")
        unique_dates = pd.to_datetime(swaption_df['EvaluationDate']).unique()
        yield_curves = {}
        for dt in tqdm(unique_dates, desc="Loading Curves"):
            eval_date = dt.date()
            date_str = eval_date.strftime('%d.%m.%Y')
            zero_path = os.path.join(FOLDER_ZERO_CURVES, f"{date_str}.csv")
            if os.path.exists(zero_path):
                zero_df = pd.read_csv(zero_path, parse_dates=['Date'])
                yield_curves[eval_date] = create_ql_yield_curve(zero_df, eval_date)

        def get_vega_for_row(row):
            """
            Calculates the vega value for a given swaption row.

            Parameters:
                row (pd.Series): A pandas Series containing the details of the swaption.

            Returns:
                float: The vega value of the swaption. If any error occurs, returns np.nan.
            """
            eval_date = pd.to_datetime(row['EvaluationDate']).date()
            if eval_date in yield_curves:
                yield_curve_handle = yield_curves[eval_date]
                ql.Settings.instance().evaluationDate = ql.Date(eval_date.day, eval_date.month, eval_date.year)
                return calculate_swaption_vega(row, yield_curve_handle)
            return np.nan

        swaption_df['Vega'] = swaption_df.progress_apply(get_vega_for_row, axis=1)
        
        # --- Generate Part 2 Visuals ---
        plot_volatility_surface(swaption_df, os.path.join(COMPARISON_DIR, 'plot5_volatility_surface'))
        plot_error_heatmaps(swaption_df, os.path.join(COMPARISON_DIR, 'plot6_error_heatmaps.png'))
        plot_error_heatmaps_weighted(swaption_df, os.path.join(COMPARISON_DIR, 'plot6b_error_heatmaps_weighted.png'))
        plot_scatter_comparison(swaption_df, os.path.join(COMPARISON_DIR, 'plot10_scatter_comparison.png'))
        plot_error_by_bucket(swaption_df, os.path.join(COMPARISON_DIR, 'plot7_error_by_expiry.png'), 'Expiry')
        plot_error_by_bucket(swaption_df, os.path.join(COMPARISON_DIR, 'plot8_error_by_tenor.png'), 'Tenor')
        plot_error_by_bucket(swaption_df, os.path.join(COMPARISON_DIR, 'plot9_error_by_volatility.png'), 'Volatility')
                
        print("\n--- All visualizations have been generated successfully. ---")

    except FileNotFoundError as e:
        print(f"\nERROR: Could not find a required data file: {e.filename}")
        print(f"Please ensure '{SWAPTION_CSV}', '{SUMMARY_CSV}' and the zero curve data folder '{FOLDER_ZERO_CURVES}' exist.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()