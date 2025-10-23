"""
Script for Visualizing Hull-White Calibration Comparison Results

Objective:
This script reads per-swaption results, calculates daily summary metrics
(both unweighted and vega-weighted RMSE), and produces a comprehensive
set of tables and plots to visualize the model comparison.

The script first calculates daily metrics on-the-fly, then generates two
categories of visualizations:
1.  High-level daily performance summaries (RMSE, distributions).
2.  Granular, instrument-level analysis (heatmaps, scatter plots, etc.).

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
from tqdm import tqdm

# --- CONFIGURATION ---
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
COMPARISON_DIR = os.path.join(RESULTS_DIR, 'comparison')

# Input files
# This is the primary input now for daily metrics
SWAPTION_CSV = os.path.join(COMPARISON_DIR, 'per_swaption_holdout_results.csv')
# This is a secondary input for parameter plots
ORIGINAL_SUMMARY_CSV = os.path.join(COMPARISON_DIR, 'daily_summary_results.csv')

# Data Folders needed for on-the-fly vega calculation
FOLDER_ZERO_CURVES = os.path.join(DATA_DIR, 'EUR ZERO CURVE')

# --- PLOT STYLING ---
NN_COLOR = cm.get_cmap('coolwarm', 10)(1)  # A nice blue
LM_COLOR = cm.get_cmap('coolwarm', 10)(9)  # A nice red
MODEL_PALETTE = {"Neural Network": NN_COLOR, "Levenberg-Marquardt": LM_COLOR}
sns.set_theme(style="whitegrid")

# --- HELPER FUNCTIONS FOR VEGA CALCULATION & PARSING ---

def parse_tenor(tenor_str: str) -> ql.Period:
    """Parses a string like '1Yr' or '6Mo' into a QuantLib Period object."""
    try:
        tenor_str = str(tenor_str).strip().upper()
        if 'YR' in tenor_str: return ql.Period(int(tenor_str.replace('YR', '')), ql.Years)
        if 'MO' in tenor_str: return ql.Period(int(tenor_str.replace('MO', '')), ql.Months)
    except (ValueError, AttributeError):
        raise ValueError(f"Could not parse tenor string: {tenor_str}")
    raise ValueError(f"Could not parse tenor string: {tenor_str}")

def parse_tenor_to_years(tenor_str: str) -> float:
    """Parses a tenor string (e.g., '10YR', '6MO') into a float representing years."""
    try:
        tenor_str = str(tenor_str).strip().upper()
        if 'YR' in tenor_str: return float(tenor_str.replace('YR', ''))
        if 'MO' in tenor_str: return float(tenor_str.replace('MO', '')) / 12.0
    except (ValueError, AttributeError):
        return np.nan
    raise ValueError(f"Could not parse tenor string to years: {tenor_str}")

def create_ql_yield_curve(
    zero_curve_df: pd.DataFrame,
    eval_date: datetime.date
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

# Replace the old `calculate_swaption_vega` function with this one.
def calculate_swaption_vega(
    swaption_row: pd.Series,
    yield_curve_handle: ql.RelinkableYieldTermStructureHandle
) -> float:
    """
    Calculates the Bachelier (Normal) vega for a single swaption.
    """
    try:
        # Get the current evaluation date from the global QL settings
        ql_eval_date = ql.Settings.instance().evaluationDate

        swap_index = ql.Euribor6M(yield_curve_handle)
        expiry_period = parse_tenor(swaption_row['ExpiryStr'])
        tenor_period = parse_tenor(swaption_row['TenorStr'])

        # Create a temporary swap to determine the ATM forward rate
        dummy_swap = ql.MakeVanillaSwap(tenor_period, swap_index, 0.0, expiry_period)
        forward_rate = dummy_swap.fairRate()
        
        if forward_rate <= 0: return 0.0

        # Create the actual underlying swap for the swaption
        underlying_swap = ql.MakeVanillaSwap(tenor_period, swap_index, forward_rate, expiry_period)
        
        # --- BUG FIX STARTS HERE ---
        # The exercise date is the start date of the swap, not the maturity date.
        exercise_date = ql.TARGET().advance(ql_eval_date, expiry_period)
        exercise = ql.EuropeanExercise(exercise_date)
        # --- BUG FIX ENDS HERE ---

        swaption_obj = ql.Swaption(underlying_swap, exercise)
        
        # Set up the pricing engine using the market's normal volatility
        market_normal_vol = swaption_row['MarketVol_bps'] / 10000.0
        vol_handle = ql.QuoteHandle(ql.SimpleQuote(market_normal_vol))
        pricing_engine = ql.BachelierSwaptionEngine(yield_curve_handle, vol_handle)
        swaption_obj.setPricingEngine(pricing_engine)
        
        # --- VEGA UNIT CLARIFICATION ---
        # QuantLib's vega() is the change in NPV for a 1% (0.01) absolute change in volatility.
        # We are weighting by the price sensitivity (in currency units) to the error, which is measured in bps.
        # Since our error 'e' is in bps, we need a vega measured in currency units per bp.
        # Vega per 1 bp = vega() * (1bp as decimal) = vega() * 0.0001
        # This is equivalent to vega() / 10000.
        # However, for weighting purposes, the absolute scaling factor cancels out. Using vega() directly
        # is also valid as it maintains the correct relative weights between instruments.
        # Let's use vega() as is, for simplicity and robustness. It represents the sensitivity to a 1% change.
        vega_value = swaption_obj.vega()
        
        return vega_value if not np.isnan(vega_value) else 0.0

    except Exception:
        import traceback
        print(f'Error {traceback.format_exc()} \n --> return 0.0')
        return 0.0

# --- NEW ON-THE-FLY CALCULATION FUNCTION ---
def calculate_daily_rmse_metrics(df_swaption: pd.DataFrame) -> pd.DataFrame:
    """Calculates daily unweighted and vega-weighted RMSE from per-swaption data."""
    print("--- Calculating daily RMSE metrics (unweighted and vega-weighted) ---")
    daily_results = []
    
    print("Pre-loading yield curves...")
    unique_dates = pd.to_datetime(df_swaption['EvaluationDate']).unique()
    yield_curves = {}
    for dt in tqdm(unique_dates, desc="Loading Curves"):
        eval_date = dt.date()
        date_str = eval_date.strftime('%d.%m.%Y')
        zero_path = os.path.join(FOLDER_ZERO_CURVES, f"{date_str}.csv")
        if os.path.exists(zero_path):
            zero_df = pd.read_csv(zero_path, parse_dates=['Date'])
            yield_curves[eval_date] = create_ql_yield_curve(zero_df, eval_date)

    grouped = df_swaption.groupby('EvaluationDate')
    for date_str, group in tqdm(grouped, total=len(grouped), desc="Calculating Daily Metrics"):
        eval_date = pd.to_datetime(date_str).date()
        
        if eval_date not in yield_curves:
            continue
            
        yield_curve_handle = yield_curves[eval_date]
        ql.Settings.instance().evaluationDate = ql.Date(eval_date.day, eval_date.month, eval_date.year)
        
        vegas = group.apply(lambda row: calculate_swaption_vega(row, yield_curve_handle), axis=1)
        
        rmse_nn_unweighted = np.sqrt((group['NN_Error_bps']**2).mean())
        rmse_lm_unweighted = np.sqrt((group['LM_Error_bps']**2).mean())

        total_vega = vegas.sum()
        if total_vega > 0:
            mse_nn_weighted = ((group['NN_Error_bps']**2) * vegas).sum() / total_vega
            mse_lm_weighted = ((group['LM_Error_bps']**2) * vegas).sum() / total_vega
            rmse_nn_weighted = np.sqrt(mse_nn_weighted)
            rmse_lm_weighted = np.sqrt(mse_lm_weighted)
        else:
            rmse_nn_weighted, rmse_lm_weighted = np.nan, np.nan
            
        daily_results.append({
            'Date': eval_date,
            'RMSE_NN_OutOfSample': rmse_nn_unweighted,
            'RMSE_LM_OutOfSample': rmse_lm_unweighted,
            'RMSE_NN_OutOfSample_VegaWeighted': rmse_nn_weighted,
            'RMSE_LM_OutOfSample_VegaWeighted': rmse_lm_weighted
        })
        
    return pd.DataFrame(daily_results)

# --- PART 1: DAILY SUMMARY ANALYSIS (UPDATED FUNCTIONS) ---
def generate_summary_tables(df: pd.DataFrame, save_dir: str):
    """Generates a summary table for performance metrics."""
    print("="*80)
    print(" GENERATING SUMMARY TABLES FROM CALCULATED DAILY METRICS ".center(80, "="))
    print("="*80)
    os.makedirs(save_dir, exist_ok=True)

    rmse_cols = [
        'RMSE_NN_OutOfSample', 'RMSE_LM_OutOfSample',
        'RMSE_NN_OutOfSample_VegaWeighted', 'RMSE_LM_OutOfSample_VegaWeighted'
    ]
    rmse_stats = df[rmse_cols].agg(['mean', 'std', 'median', 'min', 'max']).T
    rmse_stats.columns = ['Mean', 'Std Dev', 'Median', 'Min', 'Max']
    index_map = {
        'RMSE_NN_OutOfSample': 'NN (Unweighted)',
        'RMSE_LM_OutOfSample': 'LM (Unweighted)',
        'RMSE_NN_OutOfSample_VegaWeighted': 'NN (Vega-Weighted)',
        'RMSE_LM_OutOfSample_VegaWeighted': 'LM (Vega-Weighted)'
    }
    rmse_stats.index = [index_map[col] for col in rmse_cols]

    print("\n--- Table 1: Aggregate Out-of-Sample Performance Metrics (RMSE in bps) ---")
    print(rmse_stats.to_string(float_format="%.4f"))
    
    rmse_stats.to_csv(os.path.join(save_dir, 'table1_performance_metrics.csv'))
    with open(os.path.join(save_dir, 'table1_performance_metrics.txt'), 'w') as f:
        f.write("Table 1: Aggregate Out-of-Sample Performance Metrics (RMSE in bps)\n")
        f.write(rmse_stats.to_string(float_format="%.4f"))
    print(f"\nSaved Table 1 to '{save_dir}'")
    print("\n" + "="*80 + "\n")

def plot_daily_rmse(df: pd.DataFrame, save_path: str):
    """Plots both unweighted and vega-weighted daily RMSE for both models."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    ax.plot(df['Date'], df['RMSE_NN_OutOfSample'], label='NN (Unweighted)', 
            color=NN_COLOR, linestyle='-', marker='o', markersize=4, zorder=10)
    ax.plot(df['Date'], df['RMSE_LM_OutOfSample'], label='LM (Unweighted)', 
            color=LM_COLOR, linestyle='-', marker='x', markersize=5, zorder=10)
    ax.plot(df['Date'], df['RMSE_NN_OutOfSample_VegaWeighted'], label='NN (Vega-Weighted)', 
            color=NN_COLOR, linestyle='--', marker=None)
    ax.plot(df['Date'], df['RMSE_LM_OutOfSample_VegaWeighted'], label='LM (Vega-Weighted)', 
            color=LM_COLOR, linestyle='--', marker=None)
            
    ax.set_title('Daily RMSE Over Time (Unweighted vs. Vega-Weighted)', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('RMSE (bps)', fontsize=12)
    ax.legend()
    ax.grid(True, which='both', linestyle='--')
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved Plot 1: {save_path}")

def plot_rmse_distribution(df: pd.DataFrame, save_path: str):
    """Plots violin plots comparing distributions of unweighted and vega-weighted RMSE."""
    value_vars = [
        'RMSE_NN_OutOfSample', 'RMSE_LM_OutOfSample',
        'RMSE_NN_OutOfSample_VegaWeighted', 'RMSE_LM_OutOfSample_VegaWeighted'
    ]
    df_melted = df.melt(id_vars=['Date'], value_vars=value_vars, var_name='Variable', value_name='Daily RMSE (bps)')
    df_melted['Model'] = np.where(df_melted['Variable'].str.contains('_NN_'), 'Neural Network', 'Levenberg-Marquardt')
    df_melted['Weighting'] = np.where(df_melted['Variable'].str.contains('_VegaWeighted'), 'Vega-Weighted', 'Unweighted')

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.violinplot(data=df_melted, x='Model', y='Daily RMSE (bps)', hue='Weighting', 
                   ax=ax, cut=0, split=True, palette={'Unweighted': '#a1c4fd', 'Vega-Weighted': '#4a89dc'})
    ax.set_title('Distribution of Daily RMSE (Unweighted vs. Vega-Weighted)', fontsize=16)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Daily RMSE (bps)', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved Plot 2: {save_path}")

def plot_parameter_evolution(df: pd.DataFrame, save_path: str):
    """Plots a comparison of the evolution of the model parameters over time."""
    nn_alpha_col = next((c for c in df.columns if c in ['NN_param_a', 'NN_a_1']), None)
    lm_alpha_col = next((c for c in df.columns if c in ['LM_param_a', 'LM_a_1']), None)
    nn_sigma_cols = sorted([c for c in df.columns if 'NN_sigma' in c])
    lm_sigma_cols = sorted([c for c in df.columns if 'LM_sigma' in c])

    if not nn_alpha_col or not lm_alpha_col or not nn_sigma_cols or not lm_sigma_cols:
        print("Could not generate Plot 3: Alpha or Sigma parameter columns not found.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    fig.suptitle('Comparison of Model Parameter Evolution', fontsize=18, y=0.95)

    axes[0].plot(df['Date'], df[nn_alpha_col], color=NN_COLOR, linestyle='-', label=nn_alpha_col)
    axes[0].plot(df['Date'], df[lm_alpha_col], color=LM_COLOR, linestyle='--', label=lm_alpha_col)
    axes[0].set_title('Evolution of Mean-Reversion Parameter (Alpha)', fontsize=14)
    axes[0].set_ylabel('Parameter Value')
    axes[0].legend()

    num_sigmas = len(nn_sigma_cols)
    nn_sigma_colors = cm.coolwarm(np.linspace(0, 0.4, num_sigmas))
    lm_sigma_colors = cm.coolwarm(np.linspace(0.6, 1.0, num_sigmas))
    for i, col in enumerate(nn_sigma_cols): axes[1].plot(df['Date'], df[col], color=nn_sigma_colors[i], linestyle='-', label=col)
    for i, col in enumerate(lm_sigma_cols): axes[1].plot(df['Date'], df[col], color=lm_sigma_colors[i], linestyle='--', label=col)
    axes[1].set_title('Evolution of Volatility Parameters (Sigmas)', fontsize=14)
    axes[1].set_ylabel('Parameter Value')
    axes[1].legend()

    for ax in axes: ax.grid(True, which='both', linestyle='--')
    axes[1].set_xlabel('Date', fontsize=12)
    fig.autofmt_xdate()
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved Plot 3: {save_path}")

# --- PART 2: PER-SWAPTION GRANULAR ANALYSIS (UNCHANGED FUNCTIONS) ---
def plot_volatility_surface(df: pd.DataFrame, save_path_prefix: str):
    """Plots the reconstructed volatility surfaces for the latest date."""
    if df.empty: return
    eval_date = pd.to_datetime(df['EvaluationDate']).max()
    date_str = eval_date.strftime('%Y-%m-%d')
    day_df = df[df['EvaluationDate'] == date_str].copy()
    if day_df.empty: return

    day_df['Expiry'] = day_df['ExpiryStr'].apply(parse_tenor_to_years)
    day_df['Tenor'] = day_df['TenorStr'].apply(parse_tenor_to_years)
    day_df.dropna(subset=['Expiry', 'Tenor'], inplace=True)

    fig = plt.figure(figsize=(24, 8))
    fig.suptitle(f'Volatility Surface Reconstruction on Hold-Out Set ({date_str})', fontsize=16)
    
    titles = ['Market Volatility (Ground Truth)', 'Neural Network Surface', 'Levenberg-Marquardt Surface']
    z_vars = ['MarketVol_bps', 'NN_ModelVol_bps', 'LM_ModelVol_bps']

    for i in range(3):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        ax.set_title(titles[i])
        ax.plot_trisurf(day_df['Expiry'], day_df['Tenor'], day_df[z_vars[i]], cmap=cm.viridis, antialiased=True)
        ax.set_xlabel('Expiry (Years)'); ax.set_ylabel('Tenor (Years)'); ax.set_zlabel('Volatility (bps)')
        ax.view_init(elev=30, azim=-120)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = f"{save_path_prefix}_{date_str}.png"
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved Plot 5: {save_path}")

def plot_error_heatmaps(df: pd.DataFrame, save_path: str):
    """Plots heatmaps of mean prediction errors across the volatility surface."""
    df_copy = df.copy()
    df_copy['Expiry'] = df_copy['ExpiryStr'].apply(parse_tenor_to_years)
    df_copy['Tenor'] = df_copy['TenorStr'].apply(parse_tenor_to_years)
    df_copy.dropna(subset=['Expiry', 'Tenor'], inplace=True)

    expiry_bins = pd.cut(df_copy['Expiry'], bins=np.arange(0, 31, 5), right=False)
    tenor_bins = pd.cut(df_copy['Tenor'], bins=np.arange(0, 31, 5), right=False)
    
    nn_pivot = df_copy.pivot_table(index=expiry_bins, columns=tenor_bins, values='NN_Error_bps', aggfunc='mean')
    lm_pivot = df_copy.pivot_table(index=expiry_bins, columns=tenor_bins, values='LM_Error_bps', aggfunc='mean')
    diff_pivot = lm_pivot - nn_pivot

    vmax = max(abs(nn_pivot.min().min()), abs(nn_pivot.max().max()), abs(lm_pivot.min().min()), abs(lm_pivot.max().max()))
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle('Mean Prediction Errors (Model - Market) in bps across Volatility Surface', fontsize=16)

    sns.heatmap(nn_pivot, ax=ax1, cmap='coolwarm', annot=True, fmt=".2f", vmin=-vmax, vmax=vmax)
    ax1.set_title('Neural Network Mean Error')
    
    sns.heatmap(lm_pivot, ax=ax2, cmap='coolwarm', annot=True, fmt=".2f", vmin=-vmax, vmax=vmax)
    ax2.set_title('Levenberg-Marquardt Mean Error')

    sns.heatmap(diff_pivot, ax=ax3, cmap='coolwarm', annot=True, fmt=".2f")
    ax3.set_title('Error Difference (LM - NN)')

    for ax in [ax1, ax2, ax3]: ax.set_xlabel('Tenor Bins'); ax.set_ylabel('Expiry Bins' if ax==ax1 else '')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved Plot 6: {save_path}")

def plot_scatter_comparison(df: pd.DataFrame, save_path: str):
    """Plots model vs. market volatility on a scatter plot for all hold-out swaptions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True, sharex=True)
    fig.suptitle('Model vs. Market Volatility on All Hold-Out Swaptions', fontsize=16)

    max_val = max(df['MarketVol_bps'].max(), df['NN_ModelVol_bps'].max(), df['LM_ModelVol_bps'].max()) * 1.05
    min_val = min(df['MarketVol_bps'].min(), df['NN_ModelVol_bps'].min(), df['LM_ModelVol_bps'].min()) * 0.95
    
    ax1.scatter(df['MarketVol_bps'], df['NN_ModelVol_bps'], alpha=0.3, color=NN_COLOR, s=10)
    ax1.set_title('Neural Network')
    
    ax2.scatter(df['MarketVol_bps'], df['LM_ModelVol_bps'], alpha=0.3, color=LM_COLOR, s=10)
    ax2.set_title('Levenberg-Marquardt')

    for ax in [ax1, ax2]:
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='y=x (Perfect Fit)')
        ax.set_xlabel('Market Volatility (bps)')
        ax.set_ylabel('Model Volatility (bps)' if ax==ax1 else '')
        ax.legend()
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved Plot 10: {save_path}")

def plot_error_by_bucket(df: pd.DataFrame, save_path: str, bucket_col: str):
    """Generic function to plot error distribution by a specified bucket column."""
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
        except ValueError:
            print(f"Could not generate plot for {bucket_col}: Not enough unique points for quantile binning.")
            return
    else: return
    
    if bucket_col != 'Volatility':
        bins = [0, 3, 7, 31]; labels = ['Short (0-3Y)', 'Medium (3-7Y)', 'Long (7Y+)']
        df_copy['Bucket'] = pd.cut(df_copy['Bucket'], bins=bins, labels=labels, right=False)

    df_melted = df_copy.melt(
        id_vars=['Bucket'], value_vars=['NN_Error_bps', 'LM_Error_bps'],
        var_name='Model', value_name='Error (bps)'
    )
    df_melted['Model'] = df_melted['Model'].map({'NN_Error_bps': 'Neural Network', 'LM_Error_bps': 'Levenberg-Marquardt'})

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.boxplot(data=df_melted, x='Bucket', y='Error (bps)', hue='Model', palette=MODEL_PALETTE, ax=ax)
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

        # --- On-the-fly calculation of daily summary metrics ---
        summary_df = calculate_daily_rmse_metrics(swaption_df)
        summary_df['Date'] = pd.to_datetime(summary_df['Date'])
        
        # --- Generate Part 1 Visuals ---
        generate_summary_tables(summary_df, COMPARISON_DIR)
        plot_daily_rmse(summary_df, os.path.join(COMPARISON_DIR, 'plot1_daily_rmse_combined.png'))
        plot_rmse_distribution(summary_df, os.path.join(COMPARISON_DIR, 'plot2_rmse_distribution_combined.png'))
        
        try:
            original_summary_df = pd.read_csv(ORIGINAL_SUMMARY_CSV, parse_dates=['Date'])
            plot_parameter_evolution(original_summary_df, os.path.join(COMPARISON_DIR, 'plot3_parameter_evolution.png'))
        except FileNotFoundError:
            print(f"INFO: Original summary file '{ORIGINAL_SUMMARY_CSV}' not found. Skipping parameter evolution plot.")
            
        # --- Generate Part 2 Visuals ---
        plot_volatility_surface(swaption_df, os.path.join(COMPARISON_DIR, 'plot5_volatility_surface'))
        plot_error_heatmaps(swaption_df, os.path.join(COMPARISON_DIR, 'plot6_error_heatmaps.png'))
        plot_scatter_comparison(swaption_df, os.path.join(COMPARISON_DIR, 'plot10_scatter_comparison.png'))
        plot_error_by_bucket(swaption_df, os.path.join(COMPARISON_DIR, 'plot7_error_by_expiry.png'), 'Expiry')
        plot_error_by_bucket(swaption_df, os.path.join(COMPARISON_DIR, 'plot8_error_by_tenor.png'), 'Tenor')
        plot_error_by_bucket(swaption_df, os.path.join(COMPARISON_DIR, 'plot9_error_by_volatility.png'), 'Volatility')
                
        print("\n--- All visualizations have been generated successfully. ---")

    except FileNotFoundError as e:
        print(f"\nERROR: Could not find a required data file: {e.filename}")
        print(f"Please ensure '{SWAPTION_CSV}' and the zero curve data folder '{FOLDER_ZERO_CURVES}' exist.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()