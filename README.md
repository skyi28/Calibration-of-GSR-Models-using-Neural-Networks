This project implements a sophisticated, end-to-end pipeline for calibrating a financial interest rate model (Hull-White) using a hybrid machine learning approach. It trains a TensorFlow Neural Network to act as a "surrogate model," learning the complex relationship between market conditions (the yield curve) and the corresponding optimal model parameters.

Instead of running a slow, iterative calibration process for each new market data point, this trained network can predict the optimal parameters almost instantly. The core innovation is the use of `tf.custom_gradient` to bridge the gap between TensorFlow's automatic differentiation and the external, non-differentiable QuantLib financial library.

The pipeline handles everything from raw data processing and financial bootstrapping to model training, validation, and results visualization.

---

## ⚠️ Data Licensing Disclaimer

The raw market data used to develop and test this project is proprietary and sourced from a commercial data provider (Bloomberg). Due to strict licensing agreements, this data cannot be published or shared in this repository.

To run this project, you will need to provide your own licensed data in the format described below.

---

## 📂 Input Data Specification

To use this script, place your own data into the `data/` directory, following this structure:

```
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
```

---

### 📈 Swap Curve Data

* **Location:** `data/EUR SWAP CURVE/raw/`
* **File Naming:** Files must be named with the date of the curve in `DD.MM.YYYY.xlsx` format.
* **Format:** Each `.xlsx` file must contain a table like this:

```
Term    Unit    Bid     Ask
1       MO      0.5     0.6
6       MO      0.55    0.65
1       YR      0.7     0.8
2       YR      0.9     1.0
...     ...     ...     ...
```

> The script will calculate the mid-rate as `(Bid + Ask) / 2` for each row.

---

### 📊 Volatility Cube Data

This data represents **normal volatilities** (in basis points) for a grid of swaptions.

* **Location:** `data/EUR BVOL CUBE/xlsx/`
* **File Naming:** Must follow the same date-based naming as the swap curve files (`DD.MM.YYYY.xlsx`).
* **Format:** Each file contains interleaved rows of **Vol** (volatility) and **Strike** (in percent):

```
Expiry   Type     1YR    2YR    5YR    10YR   ...
1YR      Vol      85.5   90.1   92.3   95.7   ...
         Strike   -0.25  -0.15  0.10   0.50   ...
2YR      Vol      88.2   91.5   94.0   96.2   ...
         Strike   -0.20  -0.10  0.15   0.55   ...
...      ...      ...    ...    ...    ...    ...
```

---

## ⚙️ How It Works

### ✅ Data Preparation (Optional)

1. Preprocesses raw Excel files into a clean CSV format.
2. Bootstraps zero-coupon yield curves from the swap data.

### 🧠 ML Feature Preparation

* The bootstrapped zero curves become input features for the neural network.
* A `StandardScaler` is fit **only on the training set** to avoid data leakage.

### 🔄 Hybrid Training

* A custom `ResidualParameterModel` is trained.
* Loss = RMSE between QuantLib-modeled and actual market volatilities.
* Gradient is manually computed using parallelized **finite-difference approximation** via `tf.custom_gradient`.

### 🧪 Validation and Output

For each day in the test set:

* Predict Hull-White parameters (`a`, `sigma`) using the trained network.
* Save parameters to:

  ```
  results/neural_network/parameters/YYYY-MM-DD.json
  ```
* Save volatility surface comparison (market vs. model) to:

  ```
  results/neural_network/predictions/YYYY-MM-DD.csv
  ```
* Generate a 3D plot of the final test day's results.

---

## ▶️ How to Run

1. **Prepare data:**

   * Place your swap and vol data in the specified folders.
2. **Edit config:**

   * Open the main script and review the `CONFIG` section.
   * Set:

     ```python
     PREPROCESS_CURVES = True
     BOOTSTRAP_CURVES = True
     ```

     for first run. Set both to `False` on future runs to save time.
3. **Tune model settings:**
   Adjust the `TF_NN_CALIBRATION_SETTINGS` dictionary to change:

   * Neural network architecture
   * Learning rates
   * Number of epochs, etc.
4. **Run script:**
   Execute the main script from your terminal.

---

## 🚀 Future Work & Potential Improvements

* **Systematic Hyperparameter Tuning:** Use libraries like **Optuna** or **Ray Tune**.
* **Enhanced Feature Engineering:** Add lagged volatility surfaces as input features.
* **Weighted Loss Function:** Emphasize liquid or ATM swaptions.
* **Experiment Tracking:** Integrate **MLflow** or **Weights & Biases**.
* **Deployment as an API:** Use **FastAPI** or **Flask** to serve predictions.

---