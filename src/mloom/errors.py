"""Exceptions raised by mloom.

The CLI maps ``InputError`` to exit code 2 and every other exception to
exit code 1 (see "CLI" in ``docs/spec.md``).
"""


class MloomError(Exception):
    """Base class of the errors mloom raises on purpose."""


class InputError(MloomError, ValueError):
    """The config, the data, or ``unavailable-sensors.json`` is invalid.

    Raised before any model is fitted.
    """


class OptimizationError(MloomError, RuntimeError):
    """No trial of the hyperparameter search finished with a score."""
