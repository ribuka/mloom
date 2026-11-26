"""Feature selection and hyperparameter optimization of a user-chosen model.

The public API is re-exported here; see "Python API" in ``docs/spec.md``.
"""

from importlib.metadata import version

from mloom.base import Base, load_base
from mloom.config import Config
from mloom.errors import InputError, MloomError, OptimizationError
from mloom.model_file import read_run
from mloom.optimize import optimize_model
from mloom.prediction import predict
from mloom.refit import refit_model
from mloom.result import Result

__version__ = version("mloom")

__all__ = [
    "Base",
    "Config",
    "InputError",
    "MloomError",
    "OptimizationError",
    "Result",
    "__version__",
    "load_base",
    "optimize_model",
    "predict",
    "read_run",
    "refit_model",
]
