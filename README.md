# Hybrid-Model-for-Multi-Step-Forecasting-of-Solar-Irradiance

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.12.3-blue.svg)](https://www.python.org/downloads/)

## Overview

This project develops a hybrid machine learning model for multi-step forecasting of solar irradiance, combining statistical time-series analysis, deep learning, and ensemble methods. The model integrates:

- **ARIMA** (Auto-Regressive Integrated Moving Average) for capturing linear trends and seasonality.
- **GRU** (Gated Recurrent Units) for modeling non-linear sequential dependencies.
- **XGBoost** for refining predictions using meteorological features.

The approach was evaluated on the HI-SEAS weather station dataset, which includes solar radiation, temperature, humidity, wind speed, pressure, and other variables from September to December 2016. Performance metrics such as Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R² were used to assess the models. The GRU component showed superior performance, making this framework suitable for solar energy applications like grid stability and energy storage optimization.

This project is implemented as a Jupyter Notebook (`Hybrid Model for Multi-Step Forecasting of Solar Irradiance.ipynb`) and was developed and tested on Google Colab with GPU support.

## Features

- **Data Preprocessing**: Handles timezone-aware datetime indexing, cyclical encodings, lagged features, rolling statistics, and missing value imputation.
- **Feature Engineering**: Includes time-based features (e.g., hour, day of week, sunrise/sunset indicators), lagged variables, and domain-specific features like `Is_Daytime`.
- **Model Integration**: A pipeline that uses ARIMA for baseline forecasts, GRU for residual corrections, and XGBoost for final refinement.
- **Evaluation**: Comprehensive metrics and visualizations for multi-step forecasts.
- **Visualization**: Plots comparing actual vs. predicted values and model performance.

## Dataset

The dataset is the **HI-SEAS Weather Station Dataset** from Kaggle, containing 32,786 data points collected between September and December 2016. Key features include:

- **Target**: Solar Radiation (W/m²)
- **Inputs**: Temperature (°F), Humidity (%), Barometric Pressure (Hg), Wind Speed (mph), Wind Direction (°), Local Time, Date.

Download the dataset from: [Kaggle - Predicting Solar Irradiance](https://www.kaggle.com/datasets/dipankarbiswas01/predicting-solar-irradiance?resource=download)

The notebook automatically downloads the dataset using `gdown` from a Google Drive link.

## Installation and Setup

This project requires Python 3.12.3 and specific package versions for compatibility. It was tested on Google Colab with a T4 GPU.

### Prerequisites

- Python 3.12.3
- Jupyter Notebook or Google Colab
- GPU support (optional but recommended for GRU training)

### Step-by-Step Setup

1. **Clone the Repository**:
   ```
   git clone https://github.com/yourusername/hybrid-solar-forecasting.git
   cd hybrid-solar-forecasting
   ```

2. **Install Dependencies**:
   The notebook includes an installation script that downgrades NumPy and Pandas to compatible versions and installs required packages. Run the first cell in the notebook to set this up:
   - NumPy: 1.26.4
   - Pandas: 2.2.1
   - Other packages: `xgboost`, `gdown`, `pmdarima`, `torch`, `sklearn`, `matplotlib`, `seaborn`, `joblib`

   **Important Note**: After running the installation cell, **restart the runtime/session and run all cells again** to ensure the downgraded versions take effect. This avoids compatibility issues.

3. **Environment Setup in Colab**:
   - Open the notebook in Google Colab.
   - Enable GPU: Runtime > Change runtime type > Hardware accelerator > T4 GPU.
   - Run the first cell to install packages and check for GPU availability.

### Required Libraries

- `numpy==1.26.4`
- `pandas==2.2.1`
- `xgboost`
- `pmdarima`
- `torch` (with CUDA for GPU)
- `sklearn`
- `matplotlib`
- `seaborn`
- `gdown`
- `joblib`
- `pytz`

Install via pip if needed:
```
pip install numpy==1.26.4 pandas==2.2.1 xgboost gdown pmdarima torch scikit-learn matplotlib seaborn joblib pytz
```

## Usage

1. **Run the Notebook**:
   - Open `Hybrid Model for Multi-Step Forecasting of Solar Irradiance.ipynb` in Jupyter or Colab.
   - Execute cells sequentially:
     - **Setup Cell**: Installs packages and imports libraries.
     - **Data Loading and Preprocessing**: Downloads and engineers the dataset (creates 49 features).
     - **Feature Selection**: Applies correlation analysis, variance thresholding, and XGBoost importance to select top 24 features.
     - **Model Training**:
       - ARIMA: Uses `auto_arima` for parameter selection.
       - GRU: PyTorch-based model with 2 layers, 64 hidden units, trained on residuals.
       - XGBoost: MultiOutputRegressor for final predictions.
     - **Evaluation**: Computes RMSE, MAE, R² and generates plots.
   - The notebook splits data 80/20 (train/test) while preserving temporal order.

2. **Customization**:
   - Adjust hyperparameters (e.g., GRU layers, XGBoost estimators) in the respective cells.
   - For multi-step forecasts, modify the forecast horizon in the evaluation section.
   - Save models using `joblib` for reuse.

3. **Expected Output**:
   - Preprocessed dataset with 32,786 rows and 49 columns.
   - Model performance tables and prediction plots (e.g., actual vs. predicted solar radiation).
   - Figures may not load in Colab; download the notebook for local viewing if needed.

## Models and Approach

The hybrid pipeline:

1. **ARIMA**: Generates baseline forecasts for linear trends (using `pmdarima.auto_arima`).
2. **GRU**: Models residuals from ARIMA to capture non-linear patterns (PyTorch implementation with Adam optimizer and MSE loss).
3. **XGBoost**: Integrates ARIMA+GRU outputs with meteorological features (100 estimators via `MultiOutputRegressor`).

Preprocessing includes scaling (MinMaxScaler [0,1]), feature engineering (lags, rolling stats), and selection (top 20-24 features).

## Limitations and Future Work

- **Limitations**:
  - Short dataset duration (4 months) limits generalization.
  - Hybrid model showed negative R², indicating issues with component integration.
  - Computational demands for GRU on larger datasets.
  - No inclusion of advanced features like cloud cover or weather APIs.

- **Future Enhancements**:
  - Use larger, multi-year datasets.
  - Experiment with Transformer models or advanced hybrids (e.g., better residual handling).
  - Incorporate real-time data feeds for operational use.
  - Deploy as a web app for solar energy forecasting.

## Contributing

Contributions are welcome! Please fork the repo, make changes, and submit a pull request. Ensure compatibility with the specified package versions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Project developed by [Your Name]. Last updated: September 07, 2025.*
