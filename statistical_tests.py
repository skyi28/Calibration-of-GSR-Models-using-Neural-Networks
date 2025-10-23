import pandas as pd
from scipy import stats

def perform_statistical_analysis(file_path):
    """
    Performs statistical tests to compare the errors of a neural network and the Levenberg-Marquardt algorithm.

    Args:
        file_path (str): The path to the data file.
    """

    # Load the data from the text file
    data = pd.read_csv(file_path)

    # --- 1. Normality Test ---
    print("--- Normality Test ---")
    alpha_levels = [0.01, 0.05, 0.10]
    nn_error = data['RMSE_NN_OutOfSample']
    lm_error = data['RMSE_LM_OutOfSample']

    shapiro_nn = stats.shapiro(nn_error)
    shapiro_lm = stats.shapiro(lm_error)

    print(f"Shapiro-Wilk test for Neural Network errors: Statistic={shapiro_nn.statistic:.4f}, p-value={shapiro_nn.pvalue:.4f}")
    print(f"Shapiro-Wilk test for Levenberg-Marquardt errors: Statistic={shapiro_lm.statistic:.4f}, p-value={shapiro_lm.pvalue:.4f}")

    is_nn_normal = shapiro_nn.pvalue > 0.05
    is_lm_normal = shapiro_lm.pvalue > 0.05

    for alpha in alpha_levels:
        print(f"\nSignificance Level (alpha) = {alpha}")
        if shapiro_nn.pvalue > alpha:
            print("  - RMSE_NN_OutOfSample: The null hypothesis (data is normally distributed) cannot be rejected.")
        else:
            print("  - RMSE_NN_OutOfSample: The null hypothesis (data is normally distributed) is rejected.")
        if shapiro_lm.pvalue > alpha:
            print("  - RMSE_LM_OutOfSample: The null hypothesis (data is normally distributed) cannot be rejected.")
        else:
            print("  - RMSE_LM_OutOfSample: The null hypothesis (data is normally distributed) is rejected.")

    # --- 2. Test for Difference in Means ---
    print("\n--- Test for Difference in Means ---")
    if is_nn_normal and is_lm_normal:
        print("Performing Independent Samples t-test (assuming normality).")
        ttest_ind_result = stats.ttest_ind(nn_error, lm_error)
        print(f"t-statistic: {ttest_ind_result.statistic:.4f}, p-value: {ttest_ind_result.pvalue:.4f}")

        for alpha in alpha_levels:
            print(f"\nSignificance Level (alpha) = {alpha}")
            if ttest_ind_result.pvalue < alpha:
                print("  - The null hypothesis (that the means are equal) is rejected. There is a statistically significant difference between the means of the two error distributions.")
            else:
                print("  - The null hypothesis (that the means are equal) cannot be rejected. There is no statistically significant difference between the means of the two error distributions.")
    else:
        print("Performing Mann-Whitney U test (as at least one distribution is not normal).")
        mannwhitneyu_result = stats.mannwhitneyu(nn_error, lm_error)
        print(f"U-statistic: {mannwhitneyu_result.statistic:.4f}, p-value: {mannwhitneyu_result.pvalue:.4f}")

        for alpha in alpha_levels:
            print(f"\nSignificance Level (alpha) = {alpha}")
            if mannwhitneyu_result.pvalue < alpha:
                print("  - The null hypothesis (that the distributions are the same) is rejected. There is a statistically significant difference between the two error distributions.")
            else:
                print("  - The null hypothesis (that the distributions are the same) cannot be rejected. There is no statistically significant difference between the two error distributions.")

    # --- 3. Test for Difference in Variances ---
    print("\n--- Test for Difference in Variances ---")
    # Levene's test is generally more robust to deviations from normality than the F-test.
    levene_result = stats.levene(nn_error, lm_error)
    print("Performing Levene's test for equality of variances.")
    print(f"Statistic: {levene_result.statistic:.4f}, p-value: {levene_result.pvalue:.4f}")

    for alpha in alpha_levels:
        print(f"\nSignificance Level (alpha) = {alpha}")
        if levene_result.pvalue < alpha:
            print("  - The null hypothesis (that the variances are equal) is rejected. There is a statistically significant difference in the variances of the two error distributions.")
        else:
            print("  - The null hypothesis (that the variances are equal) cannot be rejected. There is no statistically significant difference in the variances of the two error distributions.")

# Provide the path to your data file
file_path = 'results/comparison/daily_summary_results.csv'
perform_statistical_analysis(file_path)