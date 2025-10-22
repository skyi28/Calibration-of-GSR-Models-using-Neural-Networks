import os
import math
import numpy as np
import pandas as pd
import keras_tuner as kt
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
# Make sure these settings match your main script exactly
TUNER_SETTINGS = {
    "directory": "results/neural_network/hyperband_tuner",
    "project_name": "hull_white_calibration"
}

# --- New Configuration for Analysis Plots ---
PLOTS_DIR = "results/neural_network/hyperparameters/analysis_plots/"

# What percentage of the top models do you want to extract? (e.g., 0.05 for top 5%)
PERCENTAGE_TO_EXTRACT = 0.05

# --- Analysis and Visualization Function (Hybrid Version) ---
def analyze_and_visualize(analysis_df, plots_output_dir):
    """
    Analyzes and visualizes the hyperparameter and error data of top models.

    Args:
        analysis_df (pd.DataFrame): DataFrame containing hyperparameters and scores
                                    for the top models.
        plots_output_dir (str): The directory where analysis plots will be saved.
    """
    print("\n--- Starting Hyperparameter and Error Analysis ---")
    os.makedirs(plots_output_dir, exist_ok=True)

    if analysis_df.empty:
        print("Error: No data provided for analysis.")
        return

    print(f"Successfully loaded data for {len(analysis_df)} top models for analysis.")
    print("\n--- Data Overview ---")
    print(analysis_df.describe())
    print("\n------------------------------------")

    # --- Visualization ---
    print("Generating visualizations...")
    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Distribution of Categorical Hyperparameters (using plt.bar)
    categorical_features = ['activation', 'use_dropout']
    for feature in categorical_features:
        if feature in analysis_df.columns:
            counts = analysis_df[feature].value_counts()
            plt.figure(figsize=(10, 6))
            plt.bar(counts.index.astype(str), counts.values, width=0.5, color='#3b72ad')
            plt.title(f'Distribution of "{feature}" in Top Models', fontsize=16)
            plt.xlabel(feature.replace("_", " ").title(), fontsize=12)
            plt.ylabel("Count", fontsize=12)
            plt.tight_layout()
            
            plot_path = os.path.join(plots_output_dir, f'distribution_{feature}.png')
            plt.savefig(plot_path)
            plt.close()
            print(f"  - Saved plot: {plot_path}")

    # 2. Distribution of Numerical Hyperparameters (using plt.hist)
    numerical_features = [
        'num_layers', 'neurons_0', 'neurons_1', 'neurons_2', 'neurons_3',
        'learning_rate', 'underestimation_penalty', 'dropout_rate'
    ]
    for feature in numerical_features:
        if feature in analysis_df.columns:
            plt.figure(figsize=(10, 6))
            if analysis_df[feature].dtype == 'int64':
                min_val = analysis_df[feature].min()
                max_val = analysis_df[feature].max()
                bins = np.arange(min_val - 0.5, max_val + 1.5, 1)
                # plt.hist(analysis_df[feature], bins=bins, color="#3b72ad", rwidth=0.9)
                sns.histplot(analysis_df[feature], bins=bins, color="#3b72ad", discrete=True)
            else:
                sns.histplot(analysis_df[feature], bins='auto', color="#3b72ad")
                # plt.hist(analysis_df[feature], bins='auto', color="#d5585c")

            plt.title(f'Distribution of "{feature}" in Top Models', fontsize=16)
            plt.xlabel(feature.replace("_", " ").title(), fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            
            plot_path = os.path.join(plots_output_dir, f'distribution_{feature}.png')
            plt.savefig(plot_path)
            plt.close()
            print(f"  - Saved plot: {plot_path}")

    # 3. Correlation Heatmap for Numerical Hyperparameters (using sns.heatmap)
    existing_numerical_features = [f for f in numerical_features if f in analysis_df.columns]
    if existing_numerical_features:
        df_numeric_corr = analysis_df[existing_numerical_features].fillna(0)
        correlation_matrix = df_numeric_corr.corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Matrix of Numerical Hyperparameters", fontsize=16)
        plt.tight_layout()
        
        plot_path = os.path.join(plots_output_dir, 'hyperparameter_correlation_heatmap.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"  - Saved plot: {plot_path}")

    # 4. Histogram of Model Errors (val_rmse)
    if 'val_rmse' in analysis_df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(analysis_df['val_rmse'], kde=True, color='#3b72ad')
        plt.title('Distribution of Validation RMSE in Top Models', fontsize=16)
        plt.xlabel('Validation RMSE (Error)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        
        plot_path = os.path.join(plots_output_dir, 'error_distribution_histogram.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"  - Saved plot: {plot_path}")
        
    # 5. Heatmap of Correlation between Errors and Features (Single Row)
    df_corr = analysis_df.copy()
    if 'activation' in df_corr.columns:
        df_corr = pd.get_dummies(df_corr, columns=['activation'], prefix='activation')
    if 'use_dropout' in df_corr.columns:
        df_corr['use_dropout'] = df_corr['use_dropout'].astype(int)
    # Drop the tuner initial epoch, round, bracket, trial id and bracket column
    if 'tuner/initial_epoch' in df_corr.columns:
        df_corr.drop(columns=['tuner/initial_epoch'], inplace=True)
    if 'tuner/round' in df_corr.columns:
        df_corr.drop(columns=['tuner/round'], inplace=True)
    if 'tuner/bracket' in df_corr.columns:
        df_corr.drop(columns=['tuner/bracket'], inplace=True)
    if 'tuner/trial_id' in df_corr.columns:
        df_corr.drop(columns=['tuner/trial_id'], inplace=True)
    if 'tuner/bracket' in df_corr.columns:
        df_corr.drop(columns=['tuner/bracket'], inplace=True)
    # Order columns alphabetically for consistency
    df_corr = df_corr.reindex(sorted(df_corr.columns), axis=1)
    # Order columns to have 'val_rmse' at the end
    df_corr = df_corr[['val_rmse'] + [col for col in df_corr.columns if col != 'val_rmse']]

    if not df_corr.empty:
        df_corr.fillna(0, inplace=True)
        correlation_matrix_with_error = df_corr.corr()
        
        # Isolate the correlations with the error ('val_rmse') and transpose for a row-based view
        error_correlation = correlation_matrix_with_error[['val_rmse']].T
        error_correlation.drop(columns=['val_rmse'], inplace=True)

        plt.figure(figsize=(18, 4)) # Adjusted for better aspect ratio
        sns.heatmap(
            error_correlation, 
            annot=True, 
            cmap='coolwarm', 
            fmt=".2f",
            linewidths=.5,
            cbar_kws={'label': 'Correlation Coefficient'}
        )
        plt.title('Correlation of Hyperparameters with Model Error (val_rmse)', fontsize=16)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plot_path = os.path.join(plots_output_dir, 'error_feature_correlation_heatmap.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"  - Saved plot: {plot_path}")

    plt.style.use('default') # Reset style
    print(f"\nAnalysis complete. All plots saved to: {plots_output_dir}")


# --- Main Logic (Unchanged) ---
if __name__ == "__main__":
    print(f"Loading tuner state from project: {TUNER_SETTINGS['project_name']}")

    # Dummy class to allow Keras Tuner to reload without the full model definition
    class DummyHyperModel:
        def build(self, hp):
            return None

    tuner = kt.Hyperband(
        hypermodel=DummyHyperModel().build,
        objective=kt.Objective("val_rmse", direction="min"),
        **TUNER_SETTINGS
    )
    tuner.reload()

    print("Retrieving all trials...")
    all_trials = tuner.oracle.get_best_trials(num_trials=len(tuner.oracle.trials))

    completed_trials = [t for t in all_trials if t.status == "COMPLETED"]

    if not completed_trials:
        print("Error: No completed trials found. Cannot extract top performers.")
        exit()

    print(f"Found {len(completed_trials)} successfully completed trials.")

    completed_trials.sort(key=lambda t: t.score)

    num_to_extract = math.ceil(len(completed_trials) * PERCENTAGE_TO_EXTRACT)
    
    print(f"Extracting the top {num_to_extract} models ({PERCENTAGE_TO_EXTRACT:.0%})...")

    top_trials = completed_trials[:num_to_extract]

    top_trials_data = []
    for trial in top_trials:
        data_point = trial.hyperparameters.values
        data_point['val_rmse'] = trial.score
        top_trials_data.append(data_point)

    analysis_dataframe = pd.DataFrame(top_trials_data)
    
    analyze_and_visualize(analysis_dataframe, PLOTS_DIR)