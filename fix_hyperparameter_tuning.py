# salvage_top_models.py

import keras_tuner as kt
import json
import os
import math

# --- Configuration ---
# Make sure these settings match your main script exactly
TUNER_SETTINGS = {
    "directory": "results/neural_network/hyperband_tuner",
    "project_name": "hull_white_calibration"
}

# Where to save the output JSON files
OUTPUT_DIR = "results/neural_network/hyperparameters/top_performers"

# What percentage of the top models do you want to extract? (e.g., 0.33 for top third)
PERCENTAGE_TO_EXTRACT = 0.05

# --- Main Logic ---
if __name__ == "__main__":
    print(f"Loading tuner state from project: {TUNER_SETTINGS['project_name']}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Re-instantiate the tuner object to load its state.
    # We use a dummy hypermodel as it's not needed for retrieving results.
    tuner = kt.Hyperband(
        hypermodel=lambda: None,
        objective=kt.Objective("val_rmse", direction="min"),
        **TUNER_SETTINGS
    )
    tuner.reload()

    print("Retrieving all trials...")
    # Get all trials from the oracle, which stores the search history
    all_trials = tuner.oracle.get_best_trials(num_trials=len(tuner.oracle.trials))

    # Filter out trials that failed, were stopped, or are still running
    completed_trials = [t for t in all_trials if t.status == "COMPLETED"]

    if not completed_trials:
        print("Error: No completed trials found. Cannot extract top performers.")
        exit()

    print(f"Found {len(completed_trials)} successfully completed trials.")

    # Sort the completed trials by their validation score (val_rmse).
    # The objective is "min", so lower is better.
    completed_trials.sort(key=lambda t: t.score)

    # Calculate how many models to extract
    num_to_extract = math.ceil(len(completed_trials) * PERCENTAGE_TO_EXTRACT)
    
    print(f"Extracting the top {num_to_extract} models ({PERCENTAGE_TO_EXTRACT:.0%})...")

    # Get the slice of the best trials
    top_trials = completed_trials[:num_to_extract]

    for i, trial in enumerate(top_trials):
        rank = i + 1
        trial_id = trial.trial_id
        score = trial.score
        hyperparameters = trial.hyperparameters.values
        
        # Define a descriptive filename
        filename = f"rank_{rank:03d}_trial_{trial_id}_rmse_{score:.4f}.json"
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        # Save the hyperparameters to a JSON file
        with open(output_path, "w") as f:
            json.dump(hyperparameters, f, indent=4)
            
        print(f"  - Saved Rank #{rank}: Trial ID {trial_id} (RMSE: {score:.4f}) -> {filename}")

    print(f"\nSuccessfully saved {len(top_trials)} hyperparameter sets to: {OUTPUT_DIR}")