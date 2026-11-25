"""Registry of the models a user can choose: estimator, scaling, search space.

Each model fixes its hyperparameter search space here, so that neither the
user nor the Skill has to write one (see "モデル" in ``docs/spec.md``).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import optuna

from mloom.errors import InputError

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
    suggest : Callable[[optuna.Trial, int], Params]
        Samples the hyperparameters of a trial, given the number of candidate
        columns.
    builders : dict[str, Callable[[Params, int], Any]]
        Builds the unfitted estimator from the hyperparameters and the seed,
        keyed by task.
    """

    name: str
    scale: bool
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


def _suggest_lightgbm(trial: optuna.Trial, n_candidates: int) -> Params:  # noqa: ARG001 - signature of ModelSpec.suggest
    return {
        "num_leaves": trial.suggest_int("num_leaves", 8, 256, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def _import_lightgbm() -> Any:  # noqa: ANN401 - the lightgbm module
    try:
        import lightgbm
    except ImportError as error:
        raise InputError(
            "model 'lightgbm' needs the lightgbm package: install mloom with the "
            "'lightgbm' extra (e.g. `uv add 'mloom[lightgbm]'`)"
        ) from error
    return lightgbm


def _lightgbm_kwargs(params: Params, seed: int) -> Params:
    return {**params, "subsample_freq": 1, "random_state": seed, "n_jobs": 1, "verbose": -1}


def _build_lightgbm_regressor(params: Params, seed: int) -> Any:  # noqa: ANN401 - estimator
    return _import_lightgbm().LGBMRegressor(**_lightgbm_kwargs(params, seed))


def _build_lightgbm_classifier(params: Params, seed: int) -> Any:  # noqa: ANN401 - estimator
    return _import_lightgbm().LGBMClassifier(**_lightgbm_kwargs(params, seed))


MODELS: dict[str, ModelSpec] = {
    "ridge": ModelSpec(
        name="ridge",
        scale=True,
        suggest=_suggest_ridge,
        builders={
            "regression": _build_ridge_regressor,
            "classification": _build_ridge_classifier,
        },
    ),
    "pls": ModelSpec(
        name="pls",
        scale=True,
        suggest=_suggest_pls,
        builders={"regression": _build_pls_regressor},
    ),
    "random_forest": ModelSpec(
        name="random_forest",
        scale=False,
        suggest=_suggest_random_forest,
        builders={
            "regression": _build_random_forest_regressor,
            "classification": _build_random_forest_classifier,
        },
    ),
    "lightgbm": ModelSpec(
        name="lightgbm",
        scale=False,
        suggest=_suggest_lightgbm,
        builders={
            "regression": _build_lightgbm_regressor,
            "classification": _build_lightgbm_classifier,
        },
    ),
}
