"""Registry of the models a user can choose: estimator, preprocessing, search space.

Each model fixes its hyperparameter search space here, so that neither the
user nor the Skill has to write one (see "モデル" in ``docs/spec.md``).
"""

import sys
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from optuna.distributions import IntDistribution
from optuna.trial import BaseTrial, FixedTrial

from mloom.errors import InputError

Params = dict[str, Any]
# The parameter of the search (and of ``best_params``) that is the number of selected columns.
K_PARAM = "k"
# The prefix of the parameters of the search (and of ``best_params``) that tell whether a categorical column is used.
USE_PREFIX = "use:"
# The parameter of LightGBM searched only when a categorical column is a candidate.
CAT_SMOOTH = "cat_smooth"


def use_param(column: str) -> str:
    """Name the parameter that tells whether a categorical column is used.

    Parameters
    ----------
    column : str
        Name of a categorical candidate column.

    Returns
    -------
    str
        ``use:<column>``. No parameter of a model has ``:`` in its name.
    """
    return f"{USE_PREFIX}{column}"


def model_params(params: Params) -> Params:
    """Leave out the parameters of the feature selection, keeping those of the model.

    Parameters
    ----------
    params : Params
        Parameters of a trial or of ``best_params``.

    Returns
    -------
    Params
        The hyperparameters of the model, without ``k`` and ``use:<column>``.
    """
    return {name: value for name, value in params.items() if name != K_PARAM and not name.startswith(USE_PREFIX)}


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
    handles_categorical : bool
        Whether the estimator takes categorical columns (encoded as integers),
        so that String, Categorical and Enum columns are feature candidates.
    suggest : Callable[[BaseTrial, int, bool], Params]
        Samples the hyperparameters of a trial, given the number of candidate
        columns and whether a categorical column is a candidate.
    builders : dict[str, Callable[[Params, int], Any]]
        Builds the unfitted estimator from the hyperparameters and the seed,
        keyed by task.
    """

    name: str
    scale: bool
    handles_missing: bool
    default_impute: str
    handles_categorical: bool
    suggest: Callable[[BaseTrial, int, bool], Params]
    builders: dict[str, Callable[[Params, int], Any]]

    def build(self, task: str, params: Params, seed: int, categorical: Sequence[int] = ()) -> Any:  # noqa: ANN401 - any scikit-learn estimator
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
        categorical : Sequence[int]
            Positions, in the input of the estimator, of the categorical
            columns; only for a model that ``handles_categorical``.

        Returns
        -------
        Any
            A scikit-learn compatible estimator.
        """
        estimator = self.builders[task](params, seed)
        if categorical:
            # categorical_column is LightGBM's alias of categorical_feature: LightGBM 4.7.0 warns of
            # categorical_feature among the parameters of the estimator (though it uses them), not of the alias.
            estimator.set_params(categorical_column=[int(i) for i in categorical])
        return estimator

    def check_params(self, params: Params) -> None:
        """Check that hyperparameters are those the search space of the model could suggest.

        Every parameter must be there, with a value of its type in its range,
        and no other parameter may be. A bound that depends on the number of
        candidate columns (``n_components`` of PLS) is not checked:
        ``build_pipeline`` limits it. ``cat_smooth`` is searched only with a
        categorical candidate, so parameters that have it are checked as
        with one.

        Parameters
        ----------
        params : Params
            Hyperparameters, without ``k`` and ``use:<column>``.

        Raises
        ------
        InputError
            When a parameter is missing, unknown, or of a wrong type or out of
            its range.
        """
        trial = FixedTrial(params)
        try:
            with warnings.catch_warnings():
                # FixedTrial only warns of a value out of the range of its distribution.
                warnings.simplefilter("error")
                self.suggest(trial, sys.maxsize, CAT_SMOOTH in params)
        except (ValueError, TypeError, UserWarning) as error:
            raise InputError(f"parameters of model {self.name!r}: {error}") from error
        unknown = sorted(set(params) - set(trial.distributions))
        if unknown:
            raise InputError(f"parameters of model {self.name!r}: unknown {', '.join(unknown)}")
        for name, distribution in trial.distributions.items():
            value = params[name]
            kinds = (int,) if isinstance(distribution, IntDistribution) else (int, float)
            if isinstance(value, bool) or not isinstance(value, kinds):
                raise InputError(
                    f"parameters of model {self.name!r}: {name} must be {kinds[0].__name__}; got {value!r}"
                )


def _suggest_ridge(trial: BaseTrial, n_candidates: int, categorical: bool) -> Params:  # noqa: ARG001 - signature of ModelSpec.suggest
    return {"alpha": trial.suggest_float("alpha", 1e-4, 1e3, log=True)}


def _build_ridge_regressor(params: Params, seed: int) -> Any:  # noqa: ANN401, ARG001 - estimator; signature of a builder
    from sklearn.linear_model import Ridge

    return Ridge(alpha=params["alpha"])


def _build_ridge_classifier(params: Params, seed: int) -> Any:  # noqa: ANN401 - estimator
    from sklearn.linear_model import LogisticRegression

    # The inverse of alpha, so that one search space serves both tasks.
    return LogisticRegression(C=1.0 / params["alpha"], max_iter=1000, random_state=seed)


def _suggest_pls(trial: BaseTrial, n_candidates: int, categorical: bool) -> Params:  # noqa: ARG001 - signature of ModelSpec.suggest
    return {"n_components": trial.suggest_int("n_components", 1, min(20, n_candidates))}


def _build_pls_regressor(params: Params, seed: int) -> Any:  # noqa: ANN401, ARG001 - estimator; signature of a builder
    from sklearn.cross_decomposition import PLSRegression

    # The pipeline already standardizes the features.
    return PLSRegression(n_components=params["n_components"], scale=False)


def _suggest_random_forest(trial: BaseTrial, n_candidates: int, categorical: bool) -> Params:  # noqa: ARG001 - signature of ModelSpec.suggest
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


def _suggest_lightgbm(trial: BaseTrial, n_candidates: int, categorical: bool) -> Params:  # noqa: ARG001 - signature of ModelSpec.suggest
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 8, 256, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    if categorical:
        # The default 10 lowered the scores much when a category has few rows (see "HPO" in docs/spec.md).
        params[CAT_SMOOTH] = trial.suggest_float(CAT_SMOOTH, 0.1, 100.0, log=True)
    return params


def _lightgbm_kwargs(params: Params, seed: int) -> Params:
    # subsample_freq=1 makes subsample take effect (LightGBM's default 0 samples no rows);
    # verbose=-1 silences the log of every fit.
    return {**params, "n_jobs": 1, "random_state": seed, "subsample_freq": 1, "verbose": -1}


def _build_lightgbm_regressor(params: Params, seed: int) -> Any:  # noqa: ANN401 - estimator
    from lightgbm import LGBMRegressor

    return LGBMRegressor(**_lightgbm_kwargs(params, seed))


def _build_lightgbm_classifier(params: Params, seed: int) -> Any:  # noqa: ANN401 - estimator
    from lightgbm import LGBMClassifier

    return LGBMClassifier(**_lightgbm_kwargs(params, seed))


MODELS: dict[str, ModelSpec] = {
    "ridge": ModelSpec(
        name="ridge",
        handles_categorical=False,
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
        handles_categorical=False,
        scale=True,
        handles_missing=False,
        default_impute="median",
        suggest=_suggest_pls,
        builders={"regression": _build_pls_regressor},
    ),
    "random_forest": ModelSpec(
        name="random_forest",
        handles_categorical=False,
        scale=False,
        handles_missing=True,
        default_impute="none",
        suggest=_suggest_random_forest,
        builders={
            "regression": _build_random_forest_regressor,
            "classification": _build_random_forest_classifier,
        },
    ),
    "lightgbm": ModelSpec(
        name="lightgbm",
        handles_categorical=True,
        scale=False,
        handles_missing=True,
        default_impute="none",
        suggest=_suggest_lightgbm,
        builders={
            "regression": _build_lightgbm_regressor,
            "classification": _build_lightgbm_classifier,
        },
    ),
}
