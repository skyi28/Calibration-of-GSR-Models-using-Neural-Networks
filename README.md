This project implements a sophisticated, end-to-end pipeline for calibrating a financial interest rate model (Hull-White) using a hybrid machine learning approach. It trains a TensorFlow Neural Network to act as a "surrogate model," learning the complex relationship between market conditions (the yield curve) and the corresponding optimal model parameters.

Instead of running a slow, iterative calibration process for each new market data point, this trained network can predict the optimal parameters almost instantly. The core innovation is the use of tf.custom_gradient to bridge the gap between TensorFlow's automatic differentiation and the external, non-differentiable QuantLib financial library.

The pipeline handles everything from raw data processing and financial bootstrapping to model training, validation, and results visualization.

## ⚠️ Data Licensing Disclaimer
The raw market data used to develop and test this project is proprietary and sourced from a commercial data provider (Bloomberg). Due to strict licensing agreements, this data cannot be published or shared in this repository.
To run this project, you will need to provide your own licensed data in the format described below.

## Input Data Specification
To use this script, you must place your own data into the data/ directory, following this structure:

data/
├── EUR SWAP CURVE/
│   └── raw/
│       ├── 31.12.2021.xlsx
│       ├── 03.01.2022.xlsx
│       └── ... (more daily files)
│
└── EUR BVOL CUBE/
    └── xlsx/
        ├── 31.12.2021.xlsx
        ├── 03.01.2022.xlsx
        └── ... (more daily files)

### Swap Curve Data
Location: data/EUR SWAP CURVE/raw/
File Naming: Files must be named with the date of the curve in DD.MM.YYYY.xlsx format.
File Format: Each .xlsx file should contain a table with the following structure. The script will calculate a mid-rate from the Bid and Ask columns.

Term	Unit	Bid	Ask
1	MO	0.5	    0.6
6	MO	0.55	0.65
1	YR	0.7	    0.8
2	YR	0.9	    1.0
...	...	...	...

### Volatility Cube Data
This data represents the normal volatilities in basis points for a grid of swaptions.
Location: data/EUR BVOL CUBE/xlsx/
File Naming: Files must be named with the date of the cube in DD.MM.YYYY.xlsx format, corresponding to the swap curve files.
File Format: Each .xlsx file should contain a table structured as a matrix of volatilities and strikes. The script is designed to parse this specific format where volatility and strike (in percent) information are interleaved.

Example Structure:
Expiry	Type	1YR	    2YR	    5YR	    10YR	...
1YR	    Vol	    85.5	90.1	92.3	95.7	...
	    Strike	-0.25	-0.15	0.10	0.50	...
2YR	    Vol	    88.2	91.5	94.0	96.2	...
	    Strike	-0.20	-0.10	0.15	0.55	...
...	    ...	    ...	    ...	    ...	    ...	    ...

## How It Works

### Data Preparation (Optional):
The script first preprocesses the raw swap curve Excel files into a clean CSV format.
It then uses a bootstrapping algorithm to derive fundamental zero-coupon yield curves from the swap curves, saving these as well.

## ML Feature Preparation:
The zero curves are used as the input features for the neural network.
A StandardScaler is fitted only on the training portion of the data to normalize the features, preventing data leakage.

## Hybrid Training:
A custom ResidualParameterModel is trained.
The loss function involves pricing a portfolio of swaptions in QuantLib and calculating the Root Mean Squared Error (RMSE) between market and model volatilities.
Since QuantLib is external, a tf.custom_gradient is used to manually compute the loss gradient via a parallelized finite-difference method.

## Validation and Output:
After training, the model is evaluated on the unseen test dataset.
For each test day, the script:
Predicts the Hull-White parameters (a, sigma) using the trained network.
Saves these parameters to a .json file in results/neural_network/parameters/.
Calculates the full volatility surface and saves the comparison (Market vs. Model) to a .csv file in results/neural_network/predictions/.
Generates a 3D plot of the results for the final test day.

## How to Run
Populate the data directory with your own licensed data, following the structure described above.
Configure the script: Open the main Python script and review the CONFIG section at the top.
Set PREPROCESS_CURVES = True and BOOTSTRAP_CURVES = True for the very first run to process your raw data. For all subsequent runs, set them to False to save time.
Review the TF_NN_CALIBRATION_SETTINGS dictionary to adjust model architecture, learning rates, epochs, etc.
Execute the script: Run the main script from your terminal.

## Future Work & Potential Improvements
This implementation provides a strong foundation. Here are some potential next steps:

Systematic Hyperparameter Tuning: Integrate a library like Optuna or Ray Tune to automatically find the optimal model and training parameters.
Enhanced Feature Engineering: Add lagged features from the previous day's volatility surface to provide more context to the model.
Weighted Loss Function: Prioritize fitting more liquid or at-the-money swaptions by assigning higher weights to their errors in the loss function.
Experiment Tracking: Use a tool like MLflow or Weights & Biases to log experiments, compare runs, and manage model artifacts.
Deploy as an API: Wrap the trained model in a FastAPI or Flask service to provide on-demand parameter predictions to other systems.

