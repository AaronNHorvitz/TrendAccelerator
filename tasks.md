# TrendAccelerator MVP: Granular Development Plan

This document outlines a granular step-by-step plan to build the Minimum Viable Product (MVP) for `TrendAccelerator`. Each task is designed to be small, testable, with a clear start and end, and focused on a single concern.

---

## Phase I: Project Setup & Core Structure

1.  **Task:** Create the main project root directory `TrendAccelerator/` and the nested Python package directory `trendaccelerator/`.
    * **Start:** No project directory exists.
    * **End:** Directories `TrendAccelerator/` (root) and `TrendAccelerator/trendaccelerator/` (package) are created.
    * **Test:** Verify directory existence using OS commands (`ls` or `dir`).

2.  **Task:** Initialize `pyproject.toml` in the `TrendAccelerator/` root directory. Include basic project metadata (name: "trendaccelerator", version: "0.1.0") and `tensorflow` as a core dependency.
    * **Start:** No `pyproject.toml` file.
    * **End:** `TrendAccelerator/pyproject.toml` exists with initial metadata and `tensorflow` in dependencies (e.g., under `[project.dependencies]`).
    * **Test:** In a virtual environment, navigate to `TrendAccelerator/` and run `pip install .`. Verify `tensorflow` is installed or listed as a dependency. Manually inspect `pyproject.toml`.

3.  **Task:** Create `__init__.py` inside `TrendAccelerator/trendaccelerator/`. Initialize it with `__version__ = "0.1.0"`.
    * **Start:** `TrendAccelerator/trendaccelerator/__init__.py` does not exist.
    * **End:** File `TrendAccelerator/trendaccelerator/__init__.py` exists and contains `__version__ = "0.1.0"`.
    * **Test:** In a Python interpreter (after `pip install .` or setting `PYTHONPATH`), run `import trendaccelerator; print(trendaccelerator.__version__)`. Should output "0.1.0".

4.  **Task:** Create the basic internal directory structure: `core/`, `preprocessing/`, `utils/` within `TrendAccelerator/trendaccelerator/`. Add an empty `__init__.py` file to each of these new directories.
    * **Start:** `trendaccelerator/` contains only its `__init__.py`.
    * **End:** Directories `trendaccelerator/core/`, `trendaccelerator/preprocessing/`, `trendaccelerator/utils/` exist, each containing an empty `__init__.py`.
    * **Test:** Verify directory and file existence. In Python, check that `import trendaccelerator.core`, `import trendaccelerator.preprocessing`, `import trendaccelerator.utils` execute without error.

5.  **Task:** Create the `tests/` directory in the `TrendAccelerator/` root. Add an empty `__init__.py` to `tests/`. Also create `tests/core/`, `tests/preprocessing/`, `tests/utils/` and add empty `__init__.py` files to each.
    * **Start:** No `tests/` directory.
    * **End:** `TrendAccelerator/tests/` and its subdirectories `core/`, `preprocessing/`, `utils/` exist, each with an `__init__.py`.
    * **Test:** Verify directory and file existence.

6.  **Task:** Configure `pytest`. Add a placeholder test file `TrendAccelerator/tests/test_setup.py` with a single test: `def test_pytest_works(): assert True`.
    * **Start:** No test files or `pytest` configuration.
    * **End:** `tests/test_setup.py` exists.
    * **Test:** Navigate to `TrendAccelerator/` root in the terminal and run `pytest`. The test should pass.

---

## Phase II: Base Model Interface & Simplest Model (CPU-based TensorFlow)

7.  **Task:** Create `TrendAccelerator/trendaccelerator/core/base_model.py`. Define an abstract base class `BaseModel` using `abc.ABC` and `abc.abstractmethod`. It should have abstract methods: `fit(self, y, X=None)` and `predict(self, steps, X_future=None)`.
    * **Start:** No `base_model.py`.
    * **End:** `trendaccelerator/core/base_model.py` exists with the `BaseModel` definition.
    * **Test:** In Python, try `from trendaccelerator.core.base_model import BaseModel`. Attempting `BaseModel()` should raise a `TypeError` because it has abstract methods.

8.  **Task:** Create `TrendAccelerator/trendaccelerator/core/naive_model.py`. Implement a `NaiveModel` class that inherits from `BaseModel`.
    * `__init__(self)`: No specific initialization needed beyond superclass.
    * `fit(self, y, X=None)`: `y` is expected to be a 1D `tf.Tensor`. Store the last value of `y` as `self.last_value`.
    * `predict(self, steps, X_future=None)`: Return a `tf.Tensor` of shape `(steps,)` where each element is `self.last_value`.
    * **Start:** No `naive_model.py`.
    * **End:** `NaiveModel` class implemented.
    * **Test:** Create `TrendAccelerator/tests/core/test_naive_model.py`.
        * Test 1: Instantiate `NaiveModel`.
        * Test 2: Call `fit` with a `tf.constant([1.0, 2.0, 3.0])`. Check `model.last_value` is `3.0`.
        * Test 3: Call `predict(steps=3)`. Check output is `tf.constant([3.0, 3.0, 3.0])`.

9.  **Task:** Expose `NaiveModel` in `TrendAccelerator/trendaccelerator/core/__init__.py` by adding `from .naive_model import NaiveModel`.
    * **Start:** `NaiveModel` is not directly importable from `trendaccelerator.core`.
    * **End:** `from trendaccelerator.core import NaiveModel` works.
    * **Test:** Add to `tests/core/test_naive_model.py`: `from trendaccelerator.core import NaiveModel`. This import should succeed.

---

## Phase III: Simplified AR(I) Model (Core Logic, CPU-based TensorFlow)

*Focus: Implement an AR(I) model. MA, Seasonality, and Exogenous variables will be for future iterations beyond MVP.*

10. **Task:** Create `TrendAccelerator/trendaccelerator/core/arima.py` (naming it `arima.py` for now, can be `sarimax.py` later if merging). Implement an `ARIMAModel` class inheriting from `BaseModel`.
    * `__init__(self, order=(0,0,0))`: Store `p, d, q = order`. Initialize `self.ar_params = None`, `self.y_train_original = None`, `self.y_train_differenced = None`, `self.d = d`.
    * Implement empty `fit` and `predict` methods (e.g., `pass` or `raise NotImplementedError`).
    * **Start:** No `arima.py`.
    * **End:** `ARIMAModel` class skeleton exists.
    * **Test:** Create `TrendAccelerator/tests/core/test_arima_model.py`. Instantiate `ARIMAModel(order=(1,1,0))`. Check that `model.d` is 1.

11. **Task:** Implement a private helper method `_difference(self, data: tf.Tensor, d: int) -> tf.Tensor` in `ARIMAModel`. If `d=0`, return `data`. If `d > 0`, iteratively apply first-order differencing `d` times using TensorFlow operations.
    * **Start:** `_difference` method does not exist or is a placeholder.
    * **End:** `_difference` method correctly differences a `tf.Tensor`.
    * **Test:** In `test_arima_model.py`, add unit tests for `_difference`:
        * `_difference(tf.constant([1,2,3,4]), d=0)` -> `[1,2,3,4]`
        * `_difference(tf.constant([1,2,4,7]), d=1)` -> `[1,2,3]` (or handle shape appropriately)
        * `_difference(tf.constant([1,2,4,8,15]), d=2)` -> `[1,2]`

12. **Task:** Update `ARIMAModel.fit(self, y, X=None)`:
    * Validate `y` is a 1D `tf.Tensor`. If not, raise `ValueError`.
    * Store `y` as `self.y_train_original`.
    * Use `self._difference` to compute the differenced series based on `self.d` and store it as `self.y_train_differenced`.
    * (No parameter estimation yet).
    * **Start:** `fit` is a placeholder.
    * **End:** `fit` stores original `y`, calculates and stores differenced `y`.
    * **Test:** In `test_arima_model.py`:
        * Call `fit` with 2D tensor, check for `ValueError`.
        * Call `fit` with `order=(1,1,0)` and `y=tf.constant([1.0, 2.0, 4.0, 7.0])`. Verify `self.y_train_original` and `self.y_train_differenced` (should be `[1.0, 2.0, 3.0]`).

13. **Task:** Implement AR(p) design matrix creation in `ARIMAModel.fit`.
    * After differencing, if `p > 0` (from `order=(p,d,q)`), create lagged versions of `self.y_train_differenced` to serve as features for AR estimation. The target will be `self.y_train_differenced` shifted.
    * Handle edge cases for series length vs `p`. For now, store these features as `self.ar_features` and the target as `self.ar_target`.
    * Example: if `y_diff = [y1, y2, y3, y4]` and `p=1`, features could be `[[y1], [y2], [y3]]` and target `[y2, y3, y4]`.
    * **Start:** `fit` only does differencing.
    * **End:** `fit` also creates `self.ar_features` and `self.ar_target` `tf.Tensor`s.
    * **Test:** Call `fit` with `order=(1,0,0)` and `y=tf.constant([1.,2.,3.,4.])`. Verify shapes and values of `self.ar_features` and `self.ar_target`. Repeat for `p=2`.

14. **Task:** Implement AR(p) coefficient estimation in `ARIMAModel.fit` using `tf.linalg.lstsq`.
    * If `p > 0`, use `tf.linalg.lstsq(self.ar_features, self.ar_target)` to estimate `self.ar_params`. Store the result. If `p=0`, `self.ar_params` can be an empty tensor or None.
    * **Start:** `self.ar_features` and `self.ar_target` are created, but no estimation.
    * **End:** `self.ar_params` are computed and stored.
    * **Test:**
        * Generate a known AR(1) series (e.g., `y_t = 0.5 * y_{t-1} + noise`). Fit with `order=(1,0,0)`. Check if `self.ar_params[0]` is close to `0.5`.
        * Test with `p=0`: ensure `self.ar_params` is handled gracefully (e.g., empty or None).

15. **Task:** Implement initial `ARIMAModel.predict(self, steps, X_future=None)` for AR(p) forecasts on the *differenced scale*.
    * If `p > 0`: Iteratively generate `steps` forecasts. For each step, use `self.ar_params` and the last `p` available values (from `self.y_train_differenced` or previously generated forecasts) to predict the next value.
    * If `p = 0` (and `q=0` for now): Forecasts are 0 on the differenced scale (mean of differenced series, assuming it's zero for simplicity in MVP).
    * Return a `tf.Tensor` of these differenced forecasts.
    * **Start:** `predict` is a placeholder.
    * **End:** `predict` generates multi-step forecasts on the differenced scale.
    * **Test:**
        * Fit AR(1) model from previous test. Predict `steps=3`. Manually verify the first few forecast values on the differenced scale.
        * Fit with `order=(0,0,0)`. Predict `steps=3`. Output should be `[0., 0., 0.]`.

16. **Task:** Implement a private helper method `_inverse_difference(self, last_original_values: tf.Tensor, forecasts_diff: tf.Tensor, d: int) -> tf.Tensor` in `ARIMAModel`.
    * `last_original_values` should be the last `d` values of the original training series (or relevant segment before differencing).
    * Iteratively add back the differences `d` times to `forecasts_diff`.
    * **Start:** No `_inverse_difference` method.
    * **End:** `_inverse_difference` method correctly reconstructs values from differenced forecasts.
    * **Test:**
        * Take `original = tf.constant([1., 2., 4., 7.])`. `d=1`. `diffed = [1., 2., 3.]`. `last_original_values = tf.constant([1.])` (if `forecasts_diff` starts from first diff). `_inverse_difference([1.], [1.,2.,3.], 1)` should be `[2.,4.,7.]` (adjust `last_original_values` based on how `_difference` was defined).
        * Create a small test: difference a series, then inverse difference. Should get (close to) original values after the initial `d` values.

17. **Task:** Integrate inverse differencing into `ARIMAModel.predict`.
    * If `self.d > 0`, after generating `forecasts_diff`, use `self._inverse_difference` to convert them back to the original scale. You'll need the last `d` values from `self.y_train_original`.
    * Return forecasts on the original scale.
    * **Start:** `predict` returns forecasts on differenced scale.
    * **End:** `predict` returns forecasts on the original scale.
    * **Test:** Fit model with `order=(1,1,0)` on `y=tf.constant([1.,2.,4.,8.,15.])`. Predict `steps=2`. Verify forecasts are on the original data's scale and make intuitive sense.

18. **Task:** Expose `ARIMAModel` in `TrendAccelerator/trendaccelerator/core/__init__.py` by adding `from .arima import ARIMAModel`.
    * **Start:** `ARIMAModel` not directly importable from `trendaccelerator.core`.
    * **End:** `from trendaccelerator.core import ARIMAModel` works.
    * **Test:** Add to `tests/core/test_arima_model.py`: `from trendaccelerator.core import ARIMAModel`. This import should succeed.

---

## Phase IV: GPU Acceleration Integration

19. **Task:** Create `TrendAccelerator/trendaccelerator/utils/gpu_setup.py`. Implement a function `setup_tf_gpu(verbose=False)` that lists available GPUs and attempts to set memory growth for each physical GPU found.
    * `tf.config.experimental.list_physical_devices('GPU')`
    * `tf.config.experimental.set_memory_growth(gpu, True)`
    * Include try-except blocks for environments without GPUs or issues.
    * If `verbose`, print found GPUs and status.
    * **Start:** No `gpu_setup.py`.
    * **End:** `gpu_setup.py` exists with `setup_tf_gpu` function.
    * **Test:**
        * Create `TrendAccelerator/tests/utils/test_gpu_setup.py`.
        * Call `setup_tf_gpu()`. On a CPU-only system, it should run without error. On a GPU system, it should configure memory growth (verified by TensorFlow logs if verbose, or lack of full memory allocation on `nvidia-smi` if observed immediately after TF init).

20. **Task:** Ensure TensorFlow operations within `ARIMAModel` (`_difference`, `lstsq`, AR forecast loop, `_inverse_difference`) are GPU-compatible. This is mostly inherent if standard `tf` ops are used. Add a call to `setup_tf_gpu()` (e.g., in a test setup or example script) before model usage.
    * **Start:** Model runs on CPU or TF default.
    * **End:** Model operations can utilize GPU if available and `setup_tf_gpu()` has been called.
    * **Test:** On a machine with an NVIDIA GPU:
        * In `test_arima_model.py` or a new test, call `setup_tf_gpu()`.
        * Fit and predict with `ARIMAModel`. Monitor GPU usage (e.g., `nvidia-smi`) during execution to see if TensorFlow is using the GPU. (Note: for small MVP data, GPU usage might be brief).
        * A simple check: `print(tf.config.list_physical_devices('GPU'))` in the test to confirm TF sees the GPU.

---

## Phase V: Basic Standalone Preprocessing - Differencing

21. **Task:** Create `TrendAccelerator/trendaccelerator/preprocessing/differencing.py`. Implement a public function `apply_difference(series: tf.Tensor, d: int = 1) -> tf.Tensor`. This function should be similar to the private `_difference` in `ARIMAModel` but usable independently.
    * **Start:** No `preprocessing/differencing.py`.
    * **End:** `apply_difference` function implemented and returns a differenced `tf.Tensor`.
    * **Test:** Create `TrendAccelerator/tests/preprocessing/test_differencing.py`. Test `apply_difference` with `d=0, 1, 2` on sample `tf.Tensor` data. Verify outputs.

22. **Task:** Expose `apply_difference` in `TrendAccelerator/trendaccelerator/preprocessing/__init__.py` by adding `from .differencing import apply_difference`.
    * **Start:** `apply_difference` not importable from `trendaccelerator.preprocessing`.
    * **End:** `from trendaccelerator.preprocessing import apply_difference` works.
    * **Test:** Add to `tests/preprocessing/test_differencing.py`: `from trendaccelerator.preprocessing import apply_difference`. This import should succeed.

---

## Phase VI: Minimal Documentation & Example

23. **Task:** Update `TrendAccelerator/README.md` with:
    * Brief "Installation" section (e.g., "Clone repo and run `pip install .`").
    * A "Basic Usage" code snippet demonstrating:
        * Importing `ARIMAModel` and `apply_difference`.
        * Creating a sample `tf.Tensor` time series.
        * Optionally using `apply_difference`.
        * Instantiating `ARIMAModel`, calling `fit`, then `predict`.
        * Printing forecasts.
    * **Start:** `README.md` is the original from the prompt or non-existent.
    * **End:** `README.md` contains installation and basic usage example for the current AR(I) model.
    * **Test:** Manually review `README.md`. Copy-paste and run the usage example in a fresh Python environment where the library is installed. It should run without errors and print forecasts.

24. **Task:** Create a simple example script `TrendAccelerator/examples/basic_arima_usage.py`. This script should contain the "Basic Usage" code from the `README.md`.
    * **Start:** No `examples/` directory or script.
    * **End:** `TrendAccelerator/examples/basic_arima_usage.py` exists and is runnable. (Create `examples/__init__.py` if Python complains about imports relative to project root when running script directly, or advise running as module if necessary).
    * **Test:** Navigate to `TrendAccelerator/examples/` (or project root) and run `python basic_arima_usage.py` (or `python -m examples.basic_arima_usage`). The script should execute and print forecast output.

---