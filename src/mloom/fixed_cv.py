"""CV scores of a model whose columns and hyperparameters are fixed, on the folds of a run.

``mloom refit`` scores the model of its base this way, and ``mloom optimize
--base`` scores how the base was made on the folds of its search (see
"ベースのモデル" in ``docs/spec.md``). Unlike a trial of the search, a fold
that fails to fit raises.
"""

from sklearn.metrics import get_scorer

from mloom.config import Config
from mloom.interpretation import FoldModels, fit_fold
from mloom.models import Params
from mloom.pipeline import Preprocessing, fit_fold_preprocess
from mloom.training import TrainingData


def fixed_fold_models(
    config: Config, params: Params, preprocessing: Preprocessing, keep: list[int], training: TrainingData
) -> FoldModels:
    """Fit the preprocessing of each CV fold, with the columns and the hyperparameters fixed.

    Parameters
    ----------
    config : Config
        The config of the model (its model, task, seed and metric).
    params : Params
        Hyperparameters of the model, without ``k`` and ``use:<column>``.
    preprocessing : Preprocessing
        The preprocessing of the model.
    keep : list[int]
        Positions, among the candidate columns of ``training``, of the
        columns the model uses on every fold.
    training : TrainingData
        The data and the CV folds of the run.

    Returns
    -------
    FoldModels
        The model on each fold; its preprocessing is fitted on the same rows
        as in the search of ``optimize_model``.
    """
    cv_plan = training.cv_plan
    n_numeric = len(training.columns.numeric)
    return FoldModels(
        config,
        params,
        cv_plan.folds,
        [
            (fit_fold_preprocess(preprocessing, training.fold_statistics(train, unlabeled), n_numeric), keep)
            for (train, _), unlabeled in zip(cv_plan.folds, cv_plan.unlabeled_folds, strict=True)
        ],
    )


def fixed_fold_scores(models: FoldModels, training: TrainingData) -> list[float]:
    """Fit the model of each fold on its training side and score it on its test side.

    Parameters
    ----------
    models : FoldModels
        The models on the folds (``fixed_fold_models``).
    training : TrainingData
        The data of the run.

    Returns
    -------
    list[float]
        The metric of each fold (``Metric.report_value``).
    """
    metric = models.config.metric_spec
    scorer = get_scorer(metric.scorer)
    scores = []
    for fold, (_, test) in enumerate(models.folds):
        pipeline = fit_fold(models, fold, training.x, training.y)
        scores.append(metric.report_value(float(scorer(pipeline, training.x[test], training.y[test]))))
    return scores
