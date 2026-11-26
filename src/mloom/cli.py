"""The command line: turns arguments into calls of the API, and saves the results.

It holds no processing of its own (see "CLI" in ``docs/spec.md``). Exit
codes: 0 on success, 2 for invalid input (including invalid arguments), 1
for any other error. The last line of stdout of ``optimize``, ``refit`` and
``predict`` is the path of the saved file (``model.pkl``, the predictions);
``show`` prints the record of the run as JSON. Progress, warnings (as
``warning:`` lines) and errors go to stderr.
"""

import argparse
import json
import sys
import traceback
import warnings
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import mloom
from mloom.base import Base, check_output, load_base
from mloom.config import (
    CV_METHODS,
    DEFAULT_SETTINGS,
    DEFAULT_UNAVAILABLE_SENSORS,
    TASKS,
    Config,
)
from mloom.errors import InputError
from mloom.evaluation import POSITIVE_SIDES
from mloom.filters import load_filters
from mloom.metrics import METRICS
from mloom.model_file import read_run
from mloom.models import MODELS
from mloom.optimize import optimize_model
from mloom.prediction import PASSED_FILTER, predict
from mloom.refit import refit_model
from mloom.result import Result

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_INPUT = 2
# Arguments of optimize that --base can do without: those of the base are used.
_REQUIRED_WITHOUT_BASE = ("target", "task", "model", "cv")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser of the ``mloom`` command.
    """
    parser = argparse.ArgumentParser(prog="mloom", description=mloom.__doc__)
    parser.add_argument("--version", action="version", version=mloom.__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    optimize = commands.add_parser("optimize", help="select features, tune a model, and save model.pkl")
    optimize.add_argument("data", type=Path, help="Parquet file")
    optimize.add_argument(
        "--base",
        type=Path,
        help="model.pkl to start from: its arguments are used unless given, and it is scored on the same CV folds",
    )
    optimize.add_argument("--target", help="target column (required without --base)")
    optimize.add_argument("--task", choices=TASKS, help="required without --base")
    optimize.add_argument("--model", choices=tuple(MODELS), help="required without --base")
    optimize.add_argument("--cv", choices=CV_METHODS, help="required without --base")
    optimize.add_argument("--output", required=True, type=Path, help="directory for model.pkl; not that of --base")
    optimize.add_argument("--n-splits", type=int, help="default: 5")
    optimize.add_argument("--time-column", help="required with --cv timeseries")
    optimize.add_argument("--group-column", help="required with --cv group")
    metrics = sorted({name for task_metrics in METRICS.values() for name in task_metrics})
    optimize.add_argument("--metric", choices=metrics, help="default: rmse (regression), log_loss (classification)")
    optimize.add_argument(
        "--exclude-columns", nargs="*", metavar="COLUMN", help="with --base, replaces the columns the base excluded"
    )
    _add_sensors_and_filters(optimize)
    optimize.add_argument(
        "--settings",
        type=Path,
        help=(
            f"project settings such as max_missing_rate (default: {DEFAULT_SETTINGS}, "
            "or with --base the settings recorded in it); keys it does not set come from the defaults of mloom"
        ),
    )
    optimize.add_argument("--n-trials", type=int, help="default: 100")
    optimize.add_argument("--timeout", type=float, help="seconds")
    optimize.add_argument("--seed", type=int, help="default: 0")
    refit = commands.add_parser(
        "refit", help="fit the columns and parameters of a base model.pkl on new data, without a search"
    )
    refit.add_argument("data", type=Path, help="Parquet file")
    refit.add_argument("--base", required=True, type=Path, help="model.pkl whose record tells how to make the model")
    refit.add_argument("--output", required=True, type=Path, help="directory for model.pkl; not that of --base")
    _add_sensors_and_filters(refit)
    refit.add_argument(
        "--settings",
        type=Path,
        help="project settings, read as optimize reads them (default: the settings recorded in --base)",
    )
    predict_parser = commands.add_parser("predict", help="predict every row of a Parquet file with a model.pkl")
    predict_parser.add_argument("model", type=Path, help="model.pkl saved by optimize or refit")
    predict_parser.add_argument("data", type=Path, help="Parquet file")
    predict_parser.add_argument("--output", required=True, type=Path, help="Parquet file of the predictions")
    predict_parser.add_argument("--keep-columns", nargs="*", default=[], metavar="COL", help="input columns to copy")
    predict_parser.add_argument("--shap", action="store_true", help="add the SHAP contributions of the selected columns")
    predict_parser.add_argument("--cm-actual-threshold", type=float, help="threshold of the target (regression)")
    predict_parser.add_argument(
        "--cm-predicted-threshold",
        type=float,
        help="threshold of the prediction (regression) or of the probability of --positive-class (default: 0.5)",
    )
    predict_parser.add_argument("--cm-positive", choices=POSITIVE_SIDES, help="side of the thresholds that is positive")
    predict_parser.add_argument("--positive-class", help="positive class of the confusion matrix (classification)")
    show = commands.add_parser("show", help="print the record of the run in a model.pkl as JSON")
    show.add_argument("model", type=Path, help="model.pkl saved by optimize or refit")
    return parser


def _add_sensors_and_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--unavailable-sensors",
        type=Path,
        default=DEFAULT_UNAVAILABLE_SENSORS,
        help=f"JSON of sensors not to use (default: {DEFAULT_UNAVAILABLE_SENSORS})",
    )
    parser.add_argument(
        "--filters",
        type=Path,
        help='JSON {"filters": [...], "train_filters": [...]} of the conditions on the rows',
    )


def to_config(args: argparse.Namespace, base: Base | None = None) -> Config:
    """Turn the parsed arguments of ``optimize`` into a config.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments; an argument not given is ``None``.
    base : Base | None
        The base of ``--base``. When given, the config starts from its config
        (``dataclasses.replace(base.config, ...)``) and the arguments given
        replace its values; ``--filters`` replaces both ``filters`` and
        ``train_filters``. Otherwise the defaults of ``Config`` are used.

    Returns
    -------
    Config
        The config.

    Raises
    ------
    InputError
        When ``--target``, ``--task``, ``--model`` or ``--cv`` is missing
        without a base, the arguments do not make a valid config, or the
        filters file is invalid.
    """
    given: dict[str, Any] = {
        "target": args.target,
        "task": args.task,
        "model": args.model,
        "cv": args.cv,
        "n_splits": args.n_splits,
        "time_column": args.time_column,
        "group_column": args.group_column,
        "metric": args.metric,
        "exclude_columns": None if args.exclude_columns is None else tuple(args.exclude_columns),
        "unavailable_sensors": args.unavailable_sensors,
        "settings": args.settings,
        "n_trials": args.n_trials,
        "timeout": args.timeout,
        "seed": args.seed,
    }
    if args.filters is not None:
        given["filters"], given["train_filters"] = load_filters(args.filters)
    values = {name: value for name, value in given.items() if value is not None}
    if base is not None:
        return replace(base.config, **values)
    missing = [f"--{name}" for name in _REQUIRED_WITHOUT_BASE if name not in values]
    if missing:
        raise InputError(f"optimize needs {', '.join(missing)} without --base")
    return Config(**values)


def main(argv: list[str] | None = None) -> int:
    """Run the ``mloom`` command.

    Parameters
    ----------
    argv : list[str] | None
        Arguments without the program name. ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        Exit code.
    """
    args = build_parser().parse_args(argv)
    run = {"optimize": _optimize, "refit": _refit, "predict": _predict, "show": _show}[args.command]
    try:
        output = run(args)
    except InputError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_INPUT
    except Exception:  # noqa: BLE001 - any other failure is reported as exit code 1
        traceback.print_exc()
        return EXIT_ERROR
    print(output)
    return EXIT_OK


def _optimize(args: argparse.Namespace) -> Path:
    base = None
    if args.base is not None:
        base = load_base(args.base)
        check_output(args.output, args.base)
    config = to_config(args, base)
    with _warning_lines():
        result = optimize_model(args.data, config, base=base)
    model_path = result.save(args.output)
    _print_summary(result)
    return model_path


def _refit(args: argparse.Namespace) -> Path:
    base = load_base(args.base)
    check_output(args.output, args.base)
    filters, train_filters = load_filters(args.filters) if args.filters is not None else (None, None)
    with _warning_lines():
        result = refit_model(
            args.data,
            base,
            filters=filters,
            train_filters=train_filters,
            unavailable_sensors=args.unavailable_sensors,
            settings=args.settings,
        )
    model_path = result.save(args.output)
    _print_summary(result)
    return model_path


@contextmanager
def _warning_lines() -> Generator[None]:
    # Each warning of the run is printed to stderr as one "warning:" line, also when the run fails.
    with warnings.catch_warnings(record=True) as caught:
        try:
            yield
        finally:
            for warning in caught:
                print(f"warning: {warning.message}", file=sys.stderr)


def _predict(args: argparse.Namespace) -> Path:
    predictions = predict(
        args.model,
        args.data,
        keep_columns=tuple(args.keep_columns),
        shap=args.shap,
        cm_actual_threshold=args.cm_actual_threshold,
        cm_predicted_threshold=args.cm_predicted_threshold,
        cm_positive=args.cm_positive,
        positive_class=args.positive_class,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.write_parquet(args.output)
    print(f"{predictions.height} rows predicted; {predictions[PASSED_FILTER].sum()} passed the filters", file=sys.stderr)
    return args.output


def _show(args: argparse.Namespace) -> str:
    return json.dumps(read_run(args.model), ensure_ascii=False, indent=2)


def _print_summary(result: Result) -> None:
    scores = result.cv_scores
    print(
        f"{result.config.metric_name}: {scores['mean']:.4g} +/- {scores['std']:.4g} "
        f"({len(scores['folds'])} folds); "
        f"{len(result.selected_columns)} of {len(result.columns.candidates)} candidate columns selected; "
        f"{len(result.columns.excluded)} columns excluded",
        file=sys.stderr,
    )
