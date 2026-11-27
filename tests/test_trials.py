"""Tests of the trials of the search in the record: left out of the JSON, or written as CSV."""

import copy

import pytest

from mloom.errors import InputError
from mloom.trials import trials_csv, without_trials

RUN = {
    "target": "y",
    "hpo": {
        "n_trials": 2,
        "best_trial": 0,
        "trials": [
            {
                "number": 0,
                "state": "complete",
                "value": 0.5,
                "folds": [0.4, 0.6],
                "params": {"alpha": 0.1, "k": 2},
                "seconds": 1.5,
                "errors": [],
            },
            {
                "number": 1,
                "state": "fail",
                "value": None,
                "folds": [None, 0.7],
                "params": {"alpha": 3.0, "k": 1, "use:grade": True},
                "seconds": 0.25,
                "errors": [{"fold": 0, "error": "ValueError('a, b')"}],
            },
        ],
    },
}


def test_without_trials_leaves_only_the_trials_out() -> None:
    """The record keeps everything but ``hpo.trials``, and the given record is not changed."""
    before = copy.deepcopy(RUN)
    shown = without_trials(RUN)
    assert shown == {"target": "y", "hpo": {"n_trials": 2, "best_trial": 0}}
    assert before == RUN


@pytest.mark.parametrize("run", [{"hpo": None}, {"hpo": {"n_trials": 1}}])
def test_without_trials_keeps_a_record_without_them(run: dict) -> None:
    """A record of a refit (``hpo`` null), or without trials, is kept as it is."""
    assert without_trials(run) == run


def test_trials_csv_has_a_row_for_each_trial() -> None:
    """Folds and parameters get a column each; a null or a missing parameter is empty; the errors are joined."""
    assert trials_csv(RUN).splitlines() == [
        "number,state,value,seconds,fold_0,fold_1,params.alpha,params.k,params.use:grade,errors",
        "0,complete,0.5,1.5,0.4,0.6,0.1,2,,",
        "1,fail,,0.25,,0.7,3.0,1,True,\"fold 0: ValueError('a, b')\"",
    ]


@pytest.mark.parametrize("run", [{"hpo": None}, {"hpo": {"n_trials": 1}}])
def test_trials_csv_of_a_record_without_trials_raises(run: dict) -> None:
    """A record with no trials of a search, such as that of a refit, raises ``InputError``."""
    with pytest.raises(InputError, match="no trials of a search"):
        trials_csv(run)
