"""The trials of the search in the record of the run: left out of the JSON that ``mloom show`` prints, or as a CSV table.

See "記録の表示" under "CLI" in ``docs/spec.md``. Like ``mloom.model_file``,
this module imports neither scikit-learn nor numpy.
"""

import csv
import io
from typing import Any

from mloom.errors import InputError

TRIALS = "trials"


def without_trials(run: dict[str, Any]) -> dict[str, Any]:
    """Leave ``hpo.trials`` out of the record of the run.

    Parameters
    ----------
    run : dict[str, Any]
        The record of the run (``mloom.read_run``). It is not changed.

    Returns
    -------
    dict[str, Any]
        The record without ``hpo.trials``; the same record when it has none.
    """
    hpo = run.get("hpo")
    if not isinstance(hpo, dict) or TRIALS not in hpo:
        return run
    return {**run, "hpo": {key: value for key, value in hpo.items() if key != TRIALS}}


def trials_csv(run: dict[str, Any]) -> str:
    """Write the trials of the search as a CSV table.

    Parameters
    ----------
    run : dict[str, Any]
        The record of the run (``mloom.read_run``).

    Returns
    -------
    str
        One row for each trial: ``number``, ``state``, ``value``,
        ``seconds``, ``fold_<i>``, ``params.<name>`` and ``errors``. A null,
        and a parameter the trial does not have, are empty.

    Raises
    ------
    InputError
        When the record has no trials of a search, such as that of a refit.
    """
    hpo = run.get("hpo")
    if not isinstance(hpo, dict) or TRIALS not in hpo:
        raise InputError("the record has no trials of a search (a model.pkl made by refit, for example)")
    trials = hpo[TRIALS]
    n_folds = max((len(trial["folds"]) for trial in trials), default=0)
    # dict keeps the order in which the names first appear.
    names = list(dict.fromkeys(name for trial in trials for name in trial["params"]))
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        ["number", "state", "value", "seconds", *(f"fold_{i}" for i in range(n_folds)), *(f"params.{n}" for n in names), "errors"]
    )
    for trial in trials:
        folds = trial["folds"] + [None] * (n_folds - len(trial["folds"]))
        writer.writerow(
            [
                trial["number"],
                trial["state"],
                trial["value"],
                trial["seconds"],
                *folds,
                *(trial["params"].get(name) for name in names),
                "; ".join(f"fold {error['fold']}: {error['error']}" for error in trial["errors"]),
            ]
        )
    return output.getvalue()
