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
def analyze_and_visualize(hyperparameter_data, plots_output_dir):
    """
    Analyzes and visualizes the hyperparameter data of top models using
    Matplotlib for distributions and Seaborn for the heatmap.

    Args:
        hyperparameter_data (list): A list of dictionaries, where each dictionary
                                    contains the hyperparameters of a top model.
        plots_output_dir (str): The directory where analysis plots will be saved.
    """
    print("\n--- Starting Hyperparameter Analysis (Hybrid: Matplotlib + Seaborn) ---")
    os.makedirs(plots_output_dir, exist_ok=True)

    if not hyperparameter_data:
        print("Error: No hyperparameter data provided for analysis.")
        return

    df = pd.DataFrame(hyperparameter_data)

    print(f"Successfully loaded {len(df)} hyperparameter sets for analysis.")
    print("\n--- Hyperparameter Data Overview ---")
    print(df.describe())
    print("\n------------------------------------")

    # --- Visualization ---
    print("Generating visualizations...")
    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Distribution of Categorical Hyperparameters (using plt.bar)
    categorical_features = ['activation', 'use_dropout']
    for feature in categorical_features:
        if feature in df.columns:
            counts = df[feature].value_counts()
            cmap = plt.cm.get_cmap('coolwarm', len(counts))

            plt.figure(figsize=(10, 6))
            plt.bar(counts.index.astype(str), counts.values, color='#3b72ad')
            plt.title(f'Distribution of "{feature}" in Top Models', fontsize=16)
            plt.xlabel(feature.replace("_", " ").title(), fontsize=12)
            plt.ylabel("Count", fontsize=12)
            
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
        if feature in df.columns:
            plt.figure(figsize=(10, 6))
            if df[feature].dtype == 'int64':
                min_val = df[feature].min()
                max_val = df[feature].max()
                bins = np.arange(min_val - 0.5, max_val + 1.5, 1)
                plt.hist(df[feature], bins=bins, color="#3b72ad", rwidth=0.9)
            else:
                plt.hist(df[feature], bins='auto', color="#d5585c")

            plt.title(f'Distribution of "{feature}" in Top Models', fontsize=16)
            plt.xlabel(feature.replace("_", " ").title(), fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            
            plot_path = os.path.join(plots_output_dir, f'distribution_{feature}.png')
            plt.savefig(plot_path)
            plt.close()
            print(f"  - Saved plot: {plot_path}")

    # 3. Correlation Heatmap for Numerical Hyperparameters (using sns.heatmap)
    existing_numerical_features = [f for f in numerical_features if f in df.columns]
    if existing_numerical_features:
        df_numeric_corr = df[existing_numerical_features].fillna(0)
        correlation_matrix = df_numeric_corr.corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Matrix of Numerical Hyperparameters", fontsize=16)
        plt.tight_layout() # Adjust layout
        
        plot_path = os.path.join(plots_output_dir, 'correlation_heatmap.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"  - Saved plot: {plot_path}")

    plt.style.use('default') # Reset style
    print(f"\nAnalysis complete. All plots saved to: {plots_output_dir}")


# --- Main Logic (No Changes Here) ---
if __name__ == "__main__":
    print(f"Loading tuner state from project: {TUNER_SETTINGS['project_name']}")

    tuner = kt.Hyperband(
        hypermodel=lambda: None,
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

    top_hyperparameters_list = []

    for i, trial in enumerate(top_trials):
        rank = i + 1
        trial_id = trial.trial_id
        score = trial.score
        hyperparameters = trial.hyperparameters.values
        
        top_hyperparameters_list.append(hyperparameters)

    analyze_and_visualize(top_hyperparameters_list, PLOTS_DIR)