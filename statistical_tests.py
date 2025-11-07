"""
This script performs a comprehensive statistical analysis on the daily Root Mean Squared Error (RMSE)
results of various Hull-White model calibration strategies. It is designed to evaluate the statistical
properties of the error distributions and to quantify the significance of performance differences
between calibration methods.

The analysis includes:
1.  Normality Testing (Shapiro-Wilk): Assesses whether the daily RMSE series for each strategy
    follows a normal distribution. This informs the choice of parametric vs. non-parametric tests
    for subsequent comparisons.
2.  Stationarity Testing (Augmented Dickey-Fuller - ADF): Determines if the daily RMSE series
    are stationary, which is crucial for time-series analysis and understanding error propagation.
3.  Pairwise Comparisons:
    -   Difference in Means: Uses either an Independent Samples t-test (if both series are normal)
        or a Mann-Whitney U test (if one or both are non-normal) to compare the central tendency
        of RMSEs between pairs of strategies.
    -   Difference in Variances (Levene's Test): Compares the homogeneity of variances between
        pairs of strategies, indicating differences in the stability of their performance.

The script loads daily RMSE data from a specified CSV file, processes each strategy's error series,
and prints detailed statistical test results to the console, including test statistics, p-values,
and conclusions at various significance levels (alpha = 0.01, 0.05, 0.10).

Usage:
    - Ensure the input CSV file ('results/comparison/daily_summary_results.csv' by default)
      contains the RMSE columns for the defined strategies.
    - Run the script directly.

Dependencies:
    - pandas: For data loading and manipulation.
    - scipy: For statistical tests (Shapiro-Wilk, t-test, Mann-Whitney U, Levene's).
    - statsmodels: For time-series specific tests (Augmented Dickey-Fuller).
"""

import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller

def perform_statistical_analysis(file_path):
    """
    Perform a comprehensive statistical analysis on the daily Root Mean Squared Error (RMSE)
    results of various Hull-White model calibration strategies.

    Parameters:
    file_path (str): The path to the CSV file containing the daily RMSE results.

    Returns:
    None

    Notes:
    - The function assumes that the input CSV file contains columns named after the respective
      strategies (e.g., 'RMSE_NN', 'RMSE_LM_Static', etc.).
    - The function performs three types of statistical tests:
        1. Normality Test (Shapiro-Wilk): Assesses whether the daily RMSE series for each strategy
            follows a normal distribution.
        2. Stationarity Test (Augmented Dickey-Fuller - ADF): Determines if the daily RMSE series
            are stationary, which is crucial for time-series analysis and understanding error propagation.
        3. Pairwise Comparisons:
            -   Difference in Means: Uses either an Independent Samples t-test (if both series are normal)
                or a Mann-Whitney U test (if one or both are non-normal) to compare the central tendency
                of RMSEs between pairs of strategies.
            -   Difference in Variances (Levene's Test): Compares the homogeneity of variances between
                pairs of strategies, indicating differences in the stability of their performance.

    The function prints detailed statistical test results to the console, including test statistics,
    p-values, and conclusions at various significance levels (alpha = 0.01, 0.05, 0.10).
    """
    try:
        data = pd.read_csv(file_path).dropna() # Drop rows with NaNs to ensure tests run
    except FileNotFoundError:
        print(f"ERROR: Data file not found at '{file_path}'. Please check the path.")
        return

    # Define the strategies and their corresponding column names from the CSV
    STRATEGY_COLUMNS = {
        'Neural Network': 'RMSE_NN',
        'LM Static': 'RMSE_LM_Static',
        'LM Pure Rolling': 'RMSE_LM_Pure_Rolling',
        'LM Adaptive Anchor': 'RMSE_LM_Adaptive_Anchor'
    }

    # Verify that all required columns exist in the DataFrame
    for name, col in STRATEGY_COLUMNS.items():
        if col not in data.columns:
            print(f"ERROR: Column '{col}' for strategy '{name}' not found in the CSV file.")
            return

    alpha_levels = [0.01, 0.05, 0.10]
    normality_results = {}

    # --- Normality Test ---
    print("="*50)
    print("--- 1. Normality Test (Shapiro-Wilk) ---")
    print("Null Hypothesis (H0): The data is normally distributed.")
    print("="*50)
    for name, col in STRATEGY_COLUMNS.items():
        error_series = data[col]
        shapiro_test = stats.shapiro(error_series)
        normality_results[name] = shapiro_test.pvalue > 0.05 # Store result for later
        print(f"\nStrategy: {name}")
        print(f"  - Shapiro-Wilk test: Statistic={shapiro_test.statistic:.4f}, p-value={shapiro_test.pvalue:.4f}")
        for alpha in alpha_levels:
            if shapiro_test.pvalue > alpha:
                print(f"  - At alpha={alpha}, the null hypothesis cannot be rejected (data may be normal).")
            else:
                print(f"  - At alpha={alpha}, the null hypothesis is rejected (data is not normal).")

    # --- Stationarity Test ---
    print("\n" + "="*50)
    print("--- 2. Stationarity Test (Augmented Dickey-Fuller) ---")
    print("Null Hypothesis (H0): The time series is non-stationary (has a unit root).")
    print("="*50)
    for name, col in STRATEGY_COLUMNS.items():
        error_series = data[col]
        adf_test = adfuller(error_series)
        print(f"\nStrategy: {name}")
        print(f"  - ADF Statistic: {adf_test[0]:.4f}")
        print(f"  - p-value: {adf_test[1]:.4f}")
        for alpha in alpha_levels:
            if adf_test[1] < alpha:
                print(f"  - At alpha={alpha}, the null hypothesis is rejected (the series is stationary).")
            else:
                print(f"  - At alpha={alpha}, the null hypothesis cannot be rejected (the series is non-stationary).")

    # --- 3. Pairwise Comparisons for Means and Variances ---
    comparison_pairs = [
        ('Neural Network', 'LM Static'),
        ('Neural Network', 'LM Pure Rolling'),
        ('Neural Network', 'LM Adaptive Anchor'),
        ('LM Static', 'LM Adaptive Anchor') # Compare the best LM to the baseline
    ]

    for pair in comparison_pairs:
        strategy1_name, strategy2_name = pair
        error1 = data[STRATEGY_COLUMNS[strategy1_name]]
        error2 = data[STRATEGY_COLUMNS[strategy2_name]]

        print("\n" + "="*60)
        print(f"--- 3. Pairwise Comparison: '{strategy1_name}' vs. '{strategy2_name}' ---")
        print("="*60)

        # --- Test for Difference in Means ---
        print("\n--- Test for Difference in Means ---")
        is_s1_normal = normality_results[strategy1_name]
        is_s2_normal = normality_results[strategy2_name]

        if is_s1_normal and is_s2_normal:
            print("Performing Independent Samples t-test (as both distributions appear normal).")
            test_result = stats.ttest_ind(error1, error2)
            print(f"t-statistic: {test_result.statistic:.4f}, p-value: {test_result.pvalue:.4f}")
            h0_text = "the means are equal"
        else:
            print("Performing Mann-Whitney U test (as at least one distribution appears non-normal).")
            test_result = stats.mannwhitneyu(error1, error2)
            print(f"U-statistic: {test_result.statistic:.4f}, p-value: {test_result.pvalue:.4f}")
            h0_text = "the distributions are the same"
        
        for alpha in alpha_levels:
            if test_result.pvalue < alpha:
                print(f"  - At alpha={alpha}, the null hypothesis (that {h0_text}) is REJECTED. The difference is statistically significant.")
            else:
                print(f"  - At alpha={alpha}, the null hypothesis (that {h0_text}) CANNOT be rejected. The difference is not statistically significant.")

        # --- Test for Difference in Variances ---
        print("\n--- Test for Difference in Variances (Levene's Test) ---")
        levene_result = stats.levene(error1, error2)
        print(f"Statistic: {levene_result.statistic:.4f}, p-value: {levene_result.pvalue:.4f}")

        for alpha in alpha_levels:
            if levene_result.pvalue < alpha:
                print(f"  - At alpha={alpha}, the null hypothesis (that variances are equal) is REJECTED. The difference in variance is statistically significant.")
            else:
                print(f"  - At alpha={alpha}, the null hypothesis (that variances are equal) CANNOT be rejected. The difference in variance is not statistically significant.")

if __name__ == "__main__":
    file_path = 'results/comparison/daily_summary_results.csv'
    perform_statistical_analysis(file_path)