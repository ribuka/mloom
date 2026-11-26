"""Feature selection and hyperparameter optimization of a user-chosen model.

The public API is re-exported here; see "Python API" in ``docs/spec.md``.
"""

from importlib.metadata import version

from mloom.config import Config
from mloom.errors import InputError, MloomError, OptimizationError
from mloom.optimize import optimize_model
from mloom.prediction import predict
from mloom.result import Result

__version__ = version("mloom")

__all__ = [
    "Config",
    "InputError",
    "MloomError",
    "OptimizationError",
    "Result",
    "__version__",
    "optimize_model",
    "predict",
]
