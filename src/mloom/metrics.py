"""Evaluation metrics: which ones each task accepts and how to score them.

Scores are computed with scikit-learn scorers, which are always "greater is
better". A metric that is naturally minimized (``rmse``, ``mae``,
``log_loss``) uses the negated scorer, and ``report_value`` turns the score
back into the natural value written to ``run.json``.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Metric:
    """An evaluation metric.

    Parameters
    ----------
    name : str
        Name used in the config and in ``run.json``.
    scorer : str
        Name of the scikit-learn scorer.
    negated : bool
        Whether the scorer returns the negated metric.
    """

    name: str
    scorer: str
    negated: bool

    def report_value(self, score: float) -> float:
        """Convert a scorer value into the natural metric value.

        Parameters
        ----------
        score : float
            Value returned by the scorer.

        Returns
        -------
        float
            The metric value (for example a positive RMSE).
        """
        return -score if self.negated else score

    @property
    def direction(self) -> Literal["minimize", "maximize"]:
        """Whether a smaller or a larger natural value is better.

        Returns
        -------
        Literal["minimize", "maximize"]
            ``"minimize"`` or ``"maximize"``.
        """
        return "minimize" if self.negated else "maximize"


METRICS: dict[str, dict[str, Metric]] = {
    "regression": {
        "rmse": Metric("rmse", "neg_root_mean_squared_error", negated=True),
        "mae": Metric("mae", "neg_mean_absolute_error", negated=True),
        "r2": Metric("r2", "r2", negated=False),
    },
    "classification": {
        "log_loss": Metric("log_loss", "neg_log_loss", negated=True),
        "accuracy": Metric("accuracy", "accuracy", negated=False),
        "f1_macro": Metric("f1_macro", "f1_macro", negated=False),
        "roc_auc": Metric("roc_auc", "roc_auc", negated=False),
    },
}

DEFAULT_METRICS: dict[str, str] = {"regression": "rmse", "classification": "log_loss"}

BINARY_ONLY_METRICS: frozenset[str] = frozenset({"roc_auc"})
