# TrendAccelerator Architecture

This document outlines the architecture of the `TrendAccelerator` library, a TensorFlow-based Python library for scalable, GPU-accelerated time-series forecasting.

## 1. Overview

TrendAccelerator is designed as a Python library, meaning its components are modules, classes, and functions that users will import and use within their own Python applications. The architecture prioritizes modularity, extensibility, and performance through GPU acceleration.

## 2. File and Folder Structure

Proposed file and folder structure for the `TrendAccelerator` library:

```
trendaccelerator/
├── trendaccelerator/
│   ├── init.py
│   ├── core/
│   │   ├── init.py
│   │   ├── base_model.py        # Abstract base class for forecasting models
│   │   ├── sarimax.py           # SARIMAX model implementation
│   │   ├── arimax.py            # ARIMAX model implementation (potentially combined with SARIMAX)
│   │   ├── lowess.py            # LOWESS smoothing implementation
│   │   ├── exponential_smoothing.py # Exponential Smoothing models
│   │   └── confidence_intervals.py # Logic for generating confidence intervals
│   ├── preprocessing/
│   │   ├── init.py
│   │   ├── differencing.py
│   │   ├── stationarity.py      # (e.g., ADF test, KPSS test)
│   │   ├── imputation.py
│   │   ├── lagging.py
│   │   └── transformers.py      # (e.g., Scalers, Box-Cox)
│   ├── detection/
│   │   ├── init.py
│   │   ├── anomalies.py         # (e.g., Z-score method)
│   │   ├── level_shifts.py      # (e.g., Step function detection)
│   │   ├── changepoints.py      # (e.g., CUSUM method)
│   ├── utils/
│   │   ├── init.py
│   │   ├── gpu_setup.py         # TensorFlow GPU configuration and checks
│   │   ├── model_selection.py   # Automated p,d,q/P,D,Q,m optimization
│   │   ├── metrics.py           # Forecasting accuracy metrics (e.g., MAE, RMSE, MAPE)
│   │   └── plotting.py          # Optional: Basic plotting utilities for forecasts/diagnostics
│   └── exceptions.py          # Custom exceptions for the library
├── tests/
│   ├── init.py
│   ├── core/
│   │   ├── test_sarimax.py
│   │   ├── test_lowess.py
│   │   └── test_exponential_smoothing.py
│   ├── preprocessing/
│   │   ├── test_differencing.py
│   │   └── test_stationarity.py
│   ├── detection/
│   │   └── test_anomalies.py
│   └── utils/
│       └── test_model_selection.py
├── examples/
│   ├── tutorial.ipynb
│   └── use_case_iot.py
│   └── use_case_economics.py
├── docs/
│   ├── index.md
│   ├── installation.md
│   ├── usage.md
│   └── api/
│       ├── core.md
│       ├── preprocessing.md
│       └── detection.md
├── pyproject.toml             # For build system and dependencies (e.g., using Poetry or Hatch)
├── README.md
├── LICENSE
└── .gitignore
```

## 3. What Each Part Does

### 3.1. `trendaccelerator/` (Main Library Code)

* **`trendaccelerator/__init__.py`**: Makes key classes and functions available when the library is imported (e.g., `from trendaccelerator import SARIMAX`). It can also define the library`s version.
* **`trendaccelerator/core/`**: Contains the core forecasting model implementations.
    * **`base_model.py`**: An abstract base class defining a common interface for all forecasting models (e.g., `fit()`, `predict()`, `summary()` methods). This promotes consistency.
    * **`sarimax.py`**: GPU-accelerated SARIMAX model. Includes logic for exogenous variables and transfer functions. TensorFlow will be heavily used here for mathematical operations and gradient computations.
    * **`arimax.py`**: GPU-accelerated ARIMAX model. This might be closely related to or integrated with `sarimax.py`.
    * **`lowess.py`**: GPU-accelerated LOWESS (Locally Weighted Scatterplot Smoothing) for trend extraction. TensorFlow will be used for the weighted linear regressions.
    * **`exponential_smoothing.py`**: Implementations of various Exponential Smoothing methods (e.g., Simple, Holt's, Holt-Winters'). GPU acceleration will be applied to the iterative calculations.
    * **`confidence_intervals.py`**: Logic to compute and provide confidence intervals for the forecasts generated by the models. This might involve analytical formulas or simulation-based approaches, accelerated with TensorFlow.
* **`trendaccelerator/preprocessing/`**: Tools for preparing time-series data.
    * **`differencing.py`**: Functions for applying differencing to make a series stationary.
    * **`stationarity.py`**: Stationarity tests (e.g., Augmented Dickey-Fuller, Kwiatkowski-Phillips-Schmidt-Shin).
    * **`imputation.py`**: Methods for handling missing values in time series (e.g., mean, median, forward fill, interpolation).
    * **`lagging.py`**: Utilities to create lagged features from time series, often used for model input.
    * **`transformers.py`**: Data transformation tools like scalers (MinMax, Standard) or power transforms (Box-Cox) that can be accelerated with TensorFlow where appropriate.
* **`trendaccelerator/detection/`**: Tools for identifying patterns or events in time series.
    * **`anomalies.py`**: Anomaly detection algorithms (e.g., Z-score method).
    * **`level_shifts.py`**: Methods to detect level shifts (sudden, persistent changes in the mean).
    * **`changepoints.py`**: Changepoint detection algorithms (e.g., CUSUM).
* **`trendaccelerator/utils/`**: Utility functions supporting the library.
    * **`gpu_setup.py`**: Functions to manage TensorFlow`s GPU visibility, memory growth configuration, and to check for available NVIDIA GPUs.
    * **`model_selection.py`**: Automated hyperparameter optimization for models like SARIMAX (e.g., finding optimal p,d,q, P,D,Q,m values using grid search, random search, or more sophisticated methods, potentially leveraging TensorFlow for parallel evaluation).
    * **`metrics.py`**: Common forecasting evaluation metrics (e.g., MSE, MAE, MAPE, SMAPE).
    * **`plotting.py`**: (Optional) Basic utilities built on libraries like Matplotlib or Plotly to quickly visualize forecasts, decomposition, or diagnostics. These are convenience functions, not core to the forecasting logic.
* **`trendaccelerator/exceptions.py`**: Custom exception classes for more specific error handling within the library (e.g., `ModelNotFittedError`, `StationarityError`).

### 3.2. `tests/`

Contains unit and integration tests for all library components. This is crucial for ensuring correctness and reliability. Each sub-module within `trendaccelerator/` should have a corresponding test sub-module. Tests will verify:
    * Correctness of mathematical computations in models.
    * Output shapes and types.
    * Handling of edge cases.
    * Performance of GPU-accelerated functions (though full benchmarking might be separate).
    * Preprocessing and detection logic.

### 3.3. `examples/`

Jupyter notebooks or Python scripts demonstrating how to use `TrendAccelerator` for various tasks and datasets. These serve as practical guides for users.

### 3.4. `docs/`

Source files for generating documentation (e.g., using Sphinx or MkDocs).
    * **`index.md`**: The main page of the documentation.
    * **`installation.md`**: Instructions on how to install the library.
    * **`usage.md`**: General guidelines and tutorials.
    * **`api/`**: Auto-generated or manually written API reference for all public modules, classes, and functions.

### 3.5. Root Directory Files

* **`pyproject.toml`**: Configuration file for the Python build system (e.g., Poetry, Hatch, or standard setuptools with PEP 517/518). Specifies project metadata, dependencies, and build instructions.
* **`README.md`**: The file you provided, giving an overview of the project.
* **`LICENSE`**: The software license for the library (e.g., MIT, Apache 2.0).
* **`.gitignore`**: Specifies intentionally untracked files that Git should ignore.

---

## 4. State Management and Service Connection

### 4.1. State Management

Being a library, `TrendAccelerator` primarily deals with **in-memory state** managed by the Python script or application.

* **Input Data**: Users provide their time-series data, typically as NumPy arrays or Pandas Series/DataFrames. Exogenous variables are also passed in this manner.
* **Model State**:
    * When a forecasting model (e.g., `SARIMAX`) is instantiated and its `fit()` method is called, the **model parameters** (coefficients, error variance, etc.) become the internal state of the model object. These parameters are learned from the data and stored within the object.
    * For TensorFlow-based models, these parameters will often be `tf.Variable` objects, managed by TensorFlow. The GPU`s memory will hold these variables and intermediate computations during training (`fit()`) and prediction (`predict()`).
* **Preprocessing State**: Some preprocessing steps might also involve state. For example, a scaler fitted on the training data will store its learned parameters (e.g., min/max values) to be applied consistently to new data. This state is typically held within the transformer object itself.
* **No Persistent State (by default)**: The library itself doesn`t manage persistent storage (like databases or file systems) for model states or data by default. Users are responsible for saving and loading model objects if they need persistence (e.g., using `joblib`, `pickle`, or TensorFlow`s own model saving utilities).

### 4.2. Service Connection (Module Interaction)

"Services" in this context refer to the different functional components (modules, classes) of the library. They connect primarily through **Python function calls and object instantiations**.

* **Data Flow**:
    1.  **User Input**: The user loads their time-series data.
    2.  **Preprocessing (Optional)**: The user might pass this data to functions in the `trendaccelerator.preprocessing` module (e.g., `differencing.apply_diff()`, `stationarity.adf_test()`). The output of these functions (e.g., a differenced series, test statistics) is returned to the user or can be directly fed into models.
    3.  **Detection (Optional)**: Users might use `trendaccelerator.detection` modules on their data before or after preprocessing to identify anomalies or changepoints.
    4.  **Model Initialization**: The user instantiates a model from `trendaccelerator.core` (e.g., `model = SARIMAX(order=(1,1,1))`).
    5.  **Model Training (`fit`)**: The user calls the `fit()` method of the model object, passing the (potentially preprocessed) time-series data and any exogenous variables.
        * Inside `fit()`, the model will heavily rely on **TensorFlow operations** for its computations.
        * It may use utilities from `trendaccelerator.utils.gpu_setup` to ensure proper GPU configuration.
        * If automated model selection is used, the `trendaccelerator.utils.model_selection` module will be invoked, which in turn might instantiate and fit multiple trial models.
    6.  **Prediction (`predict`)**: The user calls the `predict()` method, specifying the forecast horizon. This method uses the trained model parameters (internal state) and TensorFlow operations to generate forecasts.
    7.  **Confidence Intervals**: The `predict()` method or a separate utility in `trendaccelerator.core.confidence_intervals` can be called to generate confidence intervals around the forecasts.
    8.  **Evaluation**: Users can employ `trendaccelerator.utils.metrics` to evaluate forecast accuracy.

* **Internal Connections**:
    * Core forecasting models (`sarimax.py`, `lowess.py`, etc.) will all use TensorFlow as their backend for numerical computation and GPU acceleration.
    * The `base_model.py` provides a common structure, ensuring that different models can be used somewhat interchangeably by the user.
    * Utility functions (e.g., for GPU setup, model selection) are called as needed by other parts of the library or directly by the user.
    * Error handling will use custom exceptions defined in `trendaccelerator.exceptions.py`.

### 4.3. GPU Acceleration

* **TensorFlow Backend**: The primary mechanism for GPU acceleration is the use of TensorFlow for all computationally intensive tasks within the forecasting models, preprocessing steps (where applicable), and detection algorithms.
* **Data Transfer**: When GPU-accelerated functions are called, data (NumPy arrays, Pandas Series) will be implicitly or explicitly converted to TensorFlow tensors and moved to GPU memory. Results will be transferred back to CPU memory as needed.
* **`gpu_setup.py`**: This utility ensures that TensorFlow can find and utilize the NVIDIA GPU(s) correctly. It might handle settings like memory growth to prevent TensorFlow from allocating all GPU memory at once.

This architecture aims for a clean separation of concerns, making the library easier to maintain, test, and extend. The heavy lifting of GPU computation is delegated to TensorFlow, allowing the library's code to focus on the logic of the time-series algorithms.
