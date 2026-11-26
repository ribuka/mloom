"""The command line: turns arguments into calls of the API, and saves the results.

It holds no processing of its own (see "CLI" in ``docs/spec.md``). Exit
codes: 0 on success, 2 for invalid input (including invalid arguments), 1
for any other error. The last line of stdout is the path of the saved file
(``run.json`` of ``optimize``, the predictions of ``predict``); progress and
errors go to stderr.
"""

import argparse
import sys
import traceback
from pathlib import Path

import mloom
from mloom.config import (
    CV_METHODS,
    DEFAULT_SETTINGS,
    DEFAULT_UNAVAILABLE_SENSORS,
    TASKS,
    Config,
)
from mloom.errors import InputError
from mloom.filters import load_filters
from mloom.metrics import METRICS
from mloom.models import MODELS
from mloom.optimize import optimize_model
from mloom.prediction import PASSED_FILTER, predict
from mloom.result import Result

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_INPUT = 2


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
    optimize = commands.add_parser("optimize", help="select features, tune a model, and save model.pkl and run.json")
    optimize.add_argument("data", type=Path, help="Parquet file")
    optimize.add_argument("--target", required=True, help="target column")
    optimize.add_argument("--task", required=True, choices=TASKS)
    optimize.add_argument("--model", required=True, choices=tuple(MODELS))
    optimize.add_argument("--cv", required=True, choices=CV_METHODS)
    optimize.add_argument("--output", required=True, type=Path, help="directory for model.pkl and run.json")
    optimize.add_argument("--n-splits", type=int, default=5)
    optimize.add_argument("--time-column", help="required with --cv timeseries")
    optimize.add_argument("--group-column", help="required with --cv group")
    metrics = sorted({name for task_metrics in METRICS.values() for name in task_metrics})
    optimize.add_argument("--metric", choices=metrics, help="default: rmse (regression), log_loss (classification)")
    optimize.add_argument("--exclude-columns", nargs="*", default=[], metavar="COLUMN")
    optimize.add_argument(
        "--unavailable-sensors",
        type=Path,
        default=DEFAULT_UNAVAILABLE_SENSORS,
        help=f"JSON of sensors not to use (default: {DEFAULT_UNAVAILABLE_SENSORS})",
    )
    optimize.add_argument(
        "--settings",
        type=Path,
        default=DEFAULT_SETTINGS,
        help=(
            f"project settings such as max_missing_rate (default: {DEFAULT_SETTINGS}); "
            "keys it does not set come from the defaults of mloom"
        ),
    )
    optimize.add_argument(
        "--filters",
        type=Path,
        help='JSON {"filters": [...], "train_filters": [...]} of the conditions on the rows',
    )
    optimize.add_argument("--n-trials", type=int, default=100)
    optimize.add_argument("--timeout", type=float, help="seconds")
    optimize.add_argument("--seed", type=int, default=0)
    predict_parser = commands.add_parser("predict", help="predict every row of a Parquet file with a model.pkl")
    predict_parser.add_argument("model", type=Path, help="model.pkl saved by optimize")
    predict_parser.add_argument("data", type=Path, help="Parquet file")
    predict_parser.add_argument("--output", required=True, type=Path, help="Parquet file of the predictions")
    predict_parser.add_argument("--keep-columns", nargs="*", default=[], metavar="COL", help="input columns to copy")
    return parser


def to_config(args: argparse.Namespace) -> Config:
    """Turn the parsed arguments of ``optimize`` into a config.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    Config
        The config.

    Raises
    ------
    InputError
        When the arguments do not make a valid config, or the filters file
        is invalid.
    """
    filters, train_filters = load_filters(args.filters) if args.filters is not None else ((), ())
    return Config(
        target=args.target,
        task=args.task,
        model=args.model,
        cv=args.cv,
        n_splits=args.n_splits,
        time_column=args.time_column,
        group_column=args.group_column,
        metric=args.metric,
        exclude_columns=tuple(args.exclude_columns),
        filters=filters,
        train_filters=train_filters,
        unavailable_sensors=args.unavailable_sensors,
        settings=args.settings,
        n_trials=args.n_trials,
        timeout=args.timeout,
        seed=args.seed,
    )


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
    run = _optimize if args.command == "optimize" else _predict
    try:
        saved = run(args)
    except InputError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_INPUT
    except Exception:  # noqa: BLE001 - any other failure is reported as exit code 1
        traceback.print_exc()
        return EXIT_ERROR
    print(saved)
    return EXIT_OK


def _optimize(args: argparse.Namespace) -> Path:
    result = optimize_model(args.data, to_config(args))
    run_path = result.save(args.output)
    _print_summary(result)
    return run_path


def _predict(args: argparse.Namespace) -> Path:
    predictions = predict(args.model, args.data, keep_columns=tuple(args.keep_columns))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.write_parquet(args.output)
    print(f"{predictions.height} rows predicted; {predictions[PASSED_FILTER].sum()} passed the filters", file=sys.stderr)
    return args.output


def _print_summary(result: Result) -> None:
    scores = result.cv_scores
    print(
        f"{result.config.metric_name}: {scores['mean']:.4g} +/- {scores['std']:.4g} "
        f"({len(scores['folds'])} folds); "
        f"{len(result.selected_columns)} of {len(result.columns.candidates)} candidate columns selected; "
        f"{len(result.columns.excluded)} columns excluded",
        file=sys.stderr,
    )
