"""Registry of the models a user can choose: estimator, preprocessing, search space.

Each model fixes its hyperparameter search space here, so that neither the
user nor the Skill has to write one (see "モデル" in ``docs/spec.md``).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import optuna

Params = dict[str, Any]


@dataclass(frozen=True)
class ModelSpec:
    """A model of the registry.

    Parameters
    ----------
    name : str
        Name used in the config.
    scale : bool
        Whether the pipeline standardizes the features before the model.
    handles_missing : bool
        Whether the estimator fits and predicts with missing values, so that
        the features may be left unimputed (``[impute] method = "none"``).
    default_impute : str
        The imputation ``[impute] method = "auto"`` stands for.
    suggest : Callable[[optuna.Trial, int], Params]
        Samples the hyperparameters of a trial, given the number of candidate
        columns.
    builders : dict[str, Callable[[Params, int], Any]]
        Builds the unfitted estimator from the hyperparameters and the seed,
        keyed by task.
    """

    name: str
    scale: bool
    handles_missing: bool
    default_impute: str
    suggest: Callable[[optuna.Trial, int], Params]
    builders: dict[str, Callable[[Params, int], Any]]

    def build(self, task: str, params: Params, seed: int) -> Any:  # noqa: ANN401 - any scikit-learn estimator
        """Build the unfitted estimator.

        Parameters
        ----------
        task : str
            ``"regression"`` or ``"classification"``.
        params : Params
            Hyperparameters from ``suggest`` (with ``n_components`` already
            limited for PLS).
        seed : int
            Random seed passed to the estimator.

        Returns
        -------
        Any
            A scikit-learn compatible estimator.
        """
        return self.builders[task](params, seed)


def _suggest_ridge(trial: optuna.Trial, n_candidates: int) -> Params:  # noqa: ARG001 - signature of ModelSpec.suggest
    return {"alpha": trial.suggest_float("alpha", 1e-4, 1e3, log=True)}


def _build_ridge_regressor(params: Params, seed: int) -> Any:  # noqa: ANN401, ARG001 - estimator; signature of a builder
    from sklearn.linear_model import Ridge

    return Ridge(alpha=params["alpha"])


def _build_ridge_classifier(params: Params, seed: int) -> Any:  # noqa: ANN401 - estimator
    from sklearn.linear_model import LogisticRegression

    # The inverse of alpha, so that one search space serves both tasks.
    return LogisticRegression(C=1.0 / params["alpha"], max_iter=1000, random_state=seed)


def _suggest_pls(trial: optuna.Trial, n_candidates: int) -> Params:
    return {"n_components": trial.suggest_int("n_components", 1, min(20, n_candidates))}


def _build_pls_regressor(params: Params, seed: int) -> Any:  # noqa: ANN401, ARG001 - estimator; signature of a builder
    from sklearn.cross_decomposition import PLSRegression

    # The pipeline already standardizes the features.
    return PLSRegression(n_components=params["n_components"], scale=False)


def _suggest_random_forest(trial: optuna.Trial, n_candidates: int) -> Params:  # noqa: ARG001 - signature of ModelSpec.suggest
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 32, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20, log=True),
        "max_features": trial.suggest_float("max_features", 0.1, 1.0),
    }


def _build_random_forest_regressor(params: Params, seed: int) -> Any:  # noqa: ANN401 - estimator
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(**params, random_state=seed, n_jobs=1)


def _build_random_forest_classifier(params: Params, seed: int) -> Any:  # noqa: ANN401 - estimator
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(**params, random_state=seed, n_jobs=1)


def _suggest_hist_gradient_boosting(trial: optuna.Trial, n_candidates: int) -> Params:  # noqa: ARG001 - signature of ModelSpec.suggest
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_iter": trial.suggest_int("max_iter", 50, 1000, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 8, 256, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 100, log=True),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True),
        "max_features": trial.suggest_float("max_features", 0.5, 1.0),
    }


def _build_hist_gradient_boosting_regressor(params: Params, seed: int) -> Any:  # noqa: ANN401 - estimator
    from sklearn.ensemble import HistGradientBoostingRegressor

    # early_stopping="auto" would hold out 10% of the rows above 10000 rows.
    return HistGradientBoostingRegressor(**params, early_stopping=False, random_state=seed)


def _build_hist_gradient_boosting_classifier(params: Params, seed: int) -> Any:  # noqa: ANN401 - estimator
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(**params, early_stopping=False, random_state=seed)


MODELS: dict[str, ModelSpec] = {
    "ridge": ModelSpec(
        name="ridge",
        scale=True,
        handles_missing=False,
        default_impute="median",
        suggest=_suggest_ridge,
        builders={
            "regression": _build_ridge_regressor,
            "classification": _build_ridge_classifier,
        },
    ),
    "pls": ModelSpec(
        name="pls",
        scale=True,
        handles_missing=False,
        default_impute="median",
        suggest=_suggest_pls,
        builders={"regression": _build_pls_regressor},
    ),
    "random_forest": ModelSpec(
        name="random_forest",
        scale=False,
        handles_missing=True,
        default_impute="none",
        suggest=_suggest_random_forest,
        builders={
            "regression": _build_random_forest_regressor,
            "classification": _build_random_forest_classifier,
        },
    ),
    "hist_gradient_boosting": ModelSpec(
        name="hist_gradient_boosting",
        scale=False,
        handles_missing=True,
        default_impute="none",
        suggest=_suggest_hist_gradient_boosting,
        builders={
            "regression": _build_hist_gradient_boosting_regressor,
            "classification": _build_hist_gradient_boosting_classifier,
        },
    ),
}
