---
name: model-optimization
description: Optimize a machine-learning model on a Parquet file with mloom - exclude unavailable sensors, select features, tune hyperparameters with CV, and save the fitted pipeline as model.pkl with run.json - then explain the result. Use when the user asks to build, tune, or optimize a regression or classification model (ridge, pls, random_forest, lightgbm) on Parquet data, e.g. "この Parquet で品質を予測するモデルを最適化して", "lightgbm で特徴量選択とハイパラ探索をして".
---

# Model optimization

Turn the user's request into one `mloom optimize` command, run it, and
explain `run.json`. mloom decides which columns are excluded, how models are
evaluated, the search spaces, and what is saved; this skill never decides
them.

## 1. Collect the five inputs

| Input | CLI argument | Choices |
| --- | --- | --- |
| Data | positional | a Parquet file |
| Target (one column) | `--target` | a column of the data |
| Regression or classification | `--task` | `regression`, `classification` |
| Model | `--model` | `ridge`, `pls` (regression only), `random_forest`, `lightgbm` |
| CV method | `--cv` | `timeseries` (needs `--time-column`), `group` (needs `--group-column`), `kfold` |

- Ask the user for any input they did not give. Ask all missing ones in one message.
- NEVER choose the CV method yourself. The right split depends on how the model will be used (future data → `timeseries`, unseen lots → `group`), which the data does not tell.
  - When you already know the columns (from the conversation or the user's code), use them to make the question concrete, such as "`timestamp` があります。将来のデータを予測する用途なら `timeseries` にしますか？".
- Optional arguments, only when the user asks for them: `--metric`, `--n-splits` (default 5), `--exclude-columns`, `--n-trials` (default 100), `--timeout` (seconds), `--seed` (default 0), `--unavailable-sensors`.
- Choose `--output` as `outputs/<target>-<model>-<cv>` unless the user names a directory. Tell the user that an existing `model.pkl` and `run.json` there are overwritten.

## 2. Run

```powershell
uv run mloom optimize <data.parquet> --target <column> --task <task> --model <model> --cv <cv> [--time-column <column> | --group-column <column>] --output <directory>
```

- Run it from the root of the user's repository: mloom reads `config/unavailable-sensors.json` there by default. Do not create or edit that file unless the user asks.
- The search can take minutes. Run it in the background if your environment supports that, and tell the user it is running.
- Exit codes:
  - `0`: success. The last line of stdout is the path of `run.json`.
  - `2`: invalid input. Show the `error:` line from stderr, and ask the user how to fix it. Do not work around it by changing the data.
  - `1`: unexpected error. Show the end of the traceback and stop.

## 3. Explain run.json

Read `run.json` and report, in the user's language:

- The score: `cv_scores.mean` ± `cv_scores.std` of `metric` (`metric_direction` says whether smaller or larger is better), the CV method and the number of folds.
- The selected columns (`selected_columns`), out of `candidate_columns`.
- The excluded columns grouped by `reason`. Always state the `unavailable_sensor` ones with their `detail`, and list `unavailable_sensors.not_in_data` (sensors in the file that are not in the data; possibly a typo).
- `best_params`, and `hpo.n_trials_failed` when it is not 0.
- Where `model.pkl` is, and that it is loaded with `pickle.load` and predicts from a table that has the candidate columns. Warn that a pickle must only be loaded from a trusted source.

Do not claim that a score is good or bad without the user's baseline; compare only when the user gives one or asks for several runs.
