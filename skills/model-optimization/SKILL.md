---
name: model-optimization
description: Optimize a machine-learning model on a Parquet file with mloom - exclude unavailable sensors, select features, tune hyperparameters with CV, and save the fitted pipeline with the record of the run as model.pkl - or refit a saved model.pkl on new data the same way it was made, without a search - or optimize again from a saved model.pkl, keeping its arguments and comparing with it on the same CV folds - or predict a Parquet file with a saved model.pkl (scores, confusion matrix, SHAP contributions), then explain the result - or check the columns for leaks. Use when the user asks to build, tune, or optimize a regression or classification model (ridge, pls, random_forest, lightgbm) on Parquet data, or to refit or predict with a model mloom saved, e.g. "この Parquet で品質を予測するモデルを最適化して", "lightgbm で特徴量選択とハイパラ探索をして", "データが増えたので、このモデルを再学習して", "前のモデルをもとに、新しいデータで作り直して", "この model.pkl で新しいデータを予測して", "リークしてそうな列がないか見て".
---

# Model optimization

Turn the user's request into one `mloom optimize` (with or without `--base`), `mloom refit` or `mloom predict` command,
run it, and explain the result. mloom decides which columns are excluded, how
models are evaluated, the search spaces, the preprocessing, and what is saved;
this skill never decides them. Nor does it decide whether to exclude a column:
it suggests columns to doubt, with the reasons (section 4), and passes only the
columns the user chooses.

## 1. Find the mloom clone

mloom is cloned once on the PC, apart from the user's repositories, and this
skill directory is a junction to `<clone>\skills\model-optimization`. Find
`<clone>` before running anything:

```powershell
$item = Get-Item -LiteralPath '<skill directory>'
$target = @($item.Target)[0]
$realDir = if ($target) { $target } else { $item.FullName }
$clone = Split-Path -Parent (Split-Path -Parent $realDir.TrimEnd('\'))
$clone
Test-Path -LiteralPath (Join-Path $clone 'pyproject.toml'), (Join-Path $clone 'config\mloom.toml')
```

- `<skill directory>` is the directory of this `SKILL.md` as your environment shows it (Claude Code: "Base directory for this skill"; Codex: the directory of the `SKILL.md` path in the list of skills).
- `.Target` is the real path when the directory is a junction, and empty otherwise. Do not walk `..\..` from the junction's own path: that reaches the parent of the place the junction is in, not the clone.
- If either `Test-Path` prints `False`, do not run mloom. Tell the user that the junction is set up wrong (it must point to `<clone>\skills\model-optimization`; see "導入" in the mloom README), with the `<clone>` you found.
- Run every command from the root of the user's repository, with `uv run --project "<clone>"`. Pass the data, `config/` files and outputs as paths relative to the user's repository. Never install mloom into the user's repository.

## 2. Collect the inputs of `mloom optimize`

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
- Optional arguments, only when the user asks for them: `--metric`, `--n-splits` (default 5), `--exclude-columns`, `--n-trials` (default 100), `--timeout` (seconds), `--seed` (default 0), `--unavailable-sensors`, `--settings`, `--filters` (section 3).
- `config/mloom.toml` of the user's repository holds `[columns] max_missing_rate` (threshold of column exclusion), `[preprocessing] use_unlabeled_rows` (whether rows without a target join the statistics of preprocessing), `[clip]` (clipping of outliers), `[impute]` (imputation of missing values), `[selection] max_correlation` (columns whose |Spearman correlation| is above it are clustered, and only one column of each cluster can be selected; 1 clusters nothing) and `[importance] n_repeats` (how many times the permutation importance permutes each group; 0 does not compute it). Keys it does not set come from `config/mloom.local.toml` and then `config/mloom.toml` of `<clone>`.
  - Do not edit any of these files unless the user asks. When the user asks, change only the keys the user names, to the values the user gives; the keys and their choices are in `<clone>\config\mloom.toml`.
- `config/unavailable-sensors.json` of the user's repository is read by default. Do not create or edit it unless the user asks.
  - When the user asks, write only the sensors the user names, as `{"sensors": [{"name": ..., "match": ..., "reason": ...}]}`. `match` is `exact` (the default), `starts_with`, `ends_with` or `contains`; the characters are compared as they are, case-sensitive, with no regular expression or wildcard. `reason` is optional.
  - For a sensor with several columns (`TI-101_mean`, `TI-101_max`), use `starts_with` up to the separator (`TI-101_`): `TI-101` alone also matches `TI-1010`.
- Choose `--output` as `outputs/<target>-<model>-<cv>` unless the user names a directory. Tell the user that an existing `model.pkl` there is overwritten.

## 3. Row filters (only when the user gives row conditions)

When the user limits the rows (a period, an operating condition, whether a text column contains a word, and so on), write the conditions to `<output>/filters.json` and pass `--filters <output>/filters.json`. Create the `--output` directory if it does not exist. Without such conditions from the user, do not pass `--filters`.

```json
{
  "filters": [{"column": "FI-101", "op": "gt", "value": 0}],
  "train_filters": [{"column": "timestamp", "op": "lt", "value": "2026-10-01T00:00:00"}]
}
```

- `filters` limits the rows of both training and prediction (rows outside get no prediction). `train_filters` limits only the training rows; prediction still predicts those rows.
  - If the user did not say whether a condition should also apply to prediction, ask. Do not choose between `filters` and `train_filters` yourself: the predictions change.
- Write only the conditions the user gave. NEVER add a condition, or fill in a missing value (a threshold, a date, a column) by guessing; ask instead.
- Conditions are combined with AND (OR between conditions is not possible; "contains any of these words" is `str_contains_any`). `op` is one of:

| `op` | Meaning | `value` |
| --- | --- | --- |
| `eq`, `ne`, `lt`, `le`, `gt`, `ge` | compare with the value | a scalar (not null) |
| `is_in` | one of the values | a non-empty array |
| `is_between` | within the range, both ends included | `[lower, upper]` |
| `is_null`, `is_not_null` | missing or not | none |
| `str_contains_all`, `str_contains_any` | the text contains all / any of the strings, as plain text (not a regular expression) | a non-empty array of strings |

- Values for date and datetime columns are ISO 8601 strings. mloom stops with exit code 2 when the offset does not match the column (an offset for a column without a time zone, or no offset for a column with one); show the error and ask the user which they mean.

## 4. What only the agent adds

mloom decides from the numbers of the data; it cannot read what a column means. You can. Use the meaning of the columns to suggest columns to doubt, and to compare the interpretation with domain intuition. You only suggest: the user decides, and you pass only what the user chose.

### 4.1 Columns to doubt

- Leaks. Statistical selection tends to pick them and the CV score looks good, so the record alone does not reveal them:
  - the same quantity as the target, or a value measured after it (for example, the same property measured in a later process);
  - an analysis value typed in by hand;
  - a column computed from the target;
  - a control output that moves in response to the target.
- Columns not available when the model predicts: typed in by hand, recorded late, or often stopped in operation.

### 4.2 When to raise them

- Before the run: only when the user's repository has a column description file, `config/columns.csv`, `config/columns.tsv`, `config/columns.md` or `config/columns.txt` (any format, such as column name, meaning, unit and when it is measured; a pasted tag list is fine). Read every one that exists. List the columns you doubt with the reason (section 4.3), and ask the user which to exclude. "Exclude none" is a valid answer.
  - Without such a file, do not read the column names of the Parquet file before the run for this check.
  - You may tell the user once that writing such a file makes the grounds of the suggestions clear. Never create or edit it; mloom does not read it.
- After a run (`mloom optimize`, with or without `--base`, or `mloom refit`): read `candidate_columns`, `selected_columns`, `selection.clusters` and `interpretation` of the record (section 6), and list the columns you doubt with the reason, selected or important ones first.
  - When the user chooses columns to exclude, ask whether to run again with only those columns added to `--exclude-columns` (section 4.4). `mloom refit` has no `--exclude-columns`: after a refit, that run is `mloom optimize --base` (section 8).
- When the user asks for this check itself (such as "リークしてそうな列がないか見て"), use the column description file and the record of the run the user names. With neither, say that the check needs one of them.

### 4.3 Grounds

- With a column description file, quote what it says about the column.
- Without one, say clearly that you are guessing from the column name.
- When the name does not tell the meaning (only a tag such as `FI-101`), do not guess: say that you cannot tell from the name.

### 4.4 Passing the columns

- Pass only the columns the user chose, with `--exclude-columns`. NEVER add a column yourself, and never drop one in another way (filters, editing the data or a config file).
- When running again, keep the columns already in `--exclude-columns` and add the chosen ones. With `--base`, `--exclude-columns` replaces the excluded columns of the base (section 8). Tell the user that the same `--output` overwrites `model.pkl` unless the user names another directory.
- `config/unavailable-sensors.json`: create or edit it only when the user asks (section 2).
- The CV score usually gets worse after a leak is excluded. That is expected; it is not a reason to put the column back.

### 4.5 No tuning on the CV score

- NEVER change the columns, the settings or the number of trials and run again on your own to raise the CV score. Choosing them by looking at the CV score tunes them to the CV data, and the score becomes optimistic.
- When the user asks to run again to raise the score, run as asked. Then say that repeating this makes the CV score optimistic, and recommend checking the model with `mloom predict` on data after its training period (section 9; only the rows after the training period are scored, section 10).

### 4.6 Compare the interpretation with domain intuition

- Read `interpretation` of the record (section 6) and, when the user asked for them, the SHAP contributions (`mloom predict --shap`, section 10), and compare them with what the domain leads one to expect (from the user's words or the column description file; when you have no ground for an expectation, ask instead of asserting).
- Point out:
  - a sign opposite to the expected one;
  - one column, or one group (a cluster of `selection.clusters`, or a group of `permutation_importance.groups`), standing far above the others: it may also be a leak (section 4.1);
  - an effect that is physically impossible.
- Add that these show a correlation in the data, not a cause. Quote the values mloom wrote; never compute them again.

## 5. Run `mloom optimize`

```powershell
uv run --project "<clone>" mloom optimize <data.parquet> --target <column> --task <task> --model <model> --cv <cv> [--time-column <column> | --group-column <column>] [--filters <output>/filters.json] --output <directory>
```

- The search can take minutes. Run it in the background if your environment supports that, and tell the user it is running.
- Exit codes (the same for `mloom predict`):
  - `0`: success. The last line of stdout is the path of `model.pkl` (`mloom predict`: of the predictions).
  - `2`: invalid input. Show the `error:` line from stderr, and ask the user how to fix it. Do not work around it by changing the data.
  - `1`: unexpected error. Show the end of the traceback and stop.

## 6. Explain the run record

`model.pkl` holds the record of the run. Print it as JSON (the same exit codes as above):

```powershell
uv run --project "<clone>" mloom show <output>/model.pkl
```

Read that output and report, in the user's language:

- The score: `cv_scores.mean` ± `cv_scores.std` of `metric` (`metric_direction` says whether smaller or larger is better), the CV method and the number of folds.
- The selected columns (`selected_columns`), out of `candidate_columns`. With `lightgbm`, string columns (String, Categorical, Enum) are also candidates, as categorical features of LightGBM, after the numeric ones; with the other models they are excluded as `non_numeric`.
- The categorical columns, from `preprocessing.categorical` (empty for no categorical candidate): each column with its number of categories (`n_categories`). Add that a value not seen in training, and null, are predicted as missing, without an error, and that a column made from the target (such as `grade` of the sample data) is a leak: raise it as in section 4.2.
- The clusters of similar columns, from `selection`: `n_clusters` (the number of clusters, single columns included; `best_params.k` is at most this), and for each entry of `selection.clusters` (clusters of two or more columns) its `columns`, its `representative` in the final model, and whether that representative was selected (`selected`). Add that a column that is not the representative is left out only because a similar column stands for it, not because it carries no information. `selection.max_correlation` is the threshold used.
- The interpretation, from `interpretation`. Quote its values; never compute them again. Always add that every one of them shows a correlation in the data, not a cause.
  - Importance (`permutation_importance`, `null` when `n_repeats` is 0): the top groups of `groups` as `mean` ± `std` in the unit of `metric` (for example, "permuting it worsens RMSE by 0.42"). When a group has two or more `columns`, name them all and say that the importance belongs to that group of similar columns, not to `column` alone. When `folds` mixes values near 0 with large ones, say that the folds used the group differently, so it is not stable. Say that the values come from the models of the CV folds on their test rows, not from the final model itself.
  - Coefficients (`coefficients`, for `ridge` and `pls`): read `per_unit` as the change of the output (`output`: the prediction, or the log odds of `class`, which is `null` in regression) per 1 of the column in its original unit. Use the unit from the user's description of the columns when there is one; otherwise say only "per unit of the column". Compare columns with `per_std` (per 1 standard deviation), not `per_unit`. Add that a coefficient holds with the other columns fixed and within the clipping bounds.
  - VIP (`vip`, for `pls`): columns above 1 are commonly regarded as important.
- The excluded columns grouped by `reason`. Always state the `unavailable_sensor` ones with their `detail`. For each entry of `unavailable_sensors.sensors`, its `name` and `match` with the `columns` it matched (a `starts_with` or `contains` entry may match more columns than the user meant), and list `unavailable_sensors.not_in_data` (entries that matched no column; possibly a typo). `unavailable_sensors.path` and `found` tell which file was read.
- The rows, from `data`:
  - `n_rows` (rows of the input), `n_rows_filtered_out` (rows outside `filters`), and `n_rows_train_filtered_out` (rows that passed `filters` but are outside `train_filters`).
  - Counted on the rows left after both filters: `n_rows_used` for training, `n_unlabeled_rows_used` (rows without a target used only for the statistics of preprocessing), `n_rows_without_target`, and `n_rows_with_infinite_target` (rows dropped because the target is +inf or -inf).
  - `infinite_by_column` when it is not empty: infinite values of the inputs are treated as missing, in training and in prediction. A column with many of them may be worth excluding.
- `filters` and `train_filters` as they were passed, when they are not empty.
- The settings: `settings.columns`, `settings.preprocessing`, `settings.clip`, `settings.impute`, `settings.selection` and `settings.importance` with `settings.sources` (the file each value came from) and `settings.files` (the files read, with `found`). For clipping, also `preprocessing.clip`: the method, the lower and upper bounds of each column (`bounds`) and the columns not clipped (`unclipped_columns`). For imputation, also `preprocessing.impute.method`: the method the final model used, with `auto` resolved for the model (`settings.impute.method` keeps the value as set, `auto` included).
- `best_params`, and `hpo.n_trials_failed` when it is not 0. `k` is the number of selected numeric columns; `use:<column>` tells whether the search used the categorical column (`true` for a selected one); `cat_smooth` is the smoothing of LightGBM for the categories, searched only with categorical candidates. Without a numeric candidate, there is no `k` and no `use:`, and every categorical column is used.
- The trials of the search are left out of the output of `mloom show`. Print them as CSV with `uv run --project "<clone>" mloom show <output>/model.pkl --trials` (one row per trial: `number`, `state`, `value`, `seconds`, `fold_<i>`, `params.<name>`, `errors`; exit code 2 for a refit, which has no search), only in these cases:
  - `hpo.n_trials_failed` is not 0: read `errors` of the `fail` rows and report the cause of the failed folds.
  - The user asks how the search went (whether it levelled off, which ranges of the parameters scored well): answer from `value` by `number` and the `params.*` columns.
  - Never change the number of trials or the settings and run again on your own because of the table (section 4.5).
- Where `model.pkl` is, and that `mloom predict` predicts with it (section 9). Warn that a pickle must only be loaded from a trusted source.

Do not claim that a score is good or bad without the user's baseline; compare only when the user gives one or asks for several runs.

Then raise the columns to doubt (section 4.2), and compare the interpretation with domain intuition (section 4.6).

## 7. Run `mloom refit` (when the user asks to refit an existing model)

Use it when the user asks to train an existing `model.pkl` again on new or more data, the same way it was made (the same selected columns, parameters, settings, filters and CV method). It searches nothing, so it has no `--target`, `--model`, `--cv` or other arguments of `mloom optimize`: they come from the record of the base.

```powershell
uv run --project "<clone>" mloom refit <data.parquet> --base <base model.pkl> --output <directory> [--filters <output>/filters.json] [--unavailable-sensors <file>] [--settings <file>]
```

- Read the record of the base first (`mloom show <base model.pkl>`). When it has `train_filters`, show them and ask whether the new data should be trained with the same conditions (a training period often needs to grow with the data). Write only the conditions the user gives to `<output>/filters.json` (section 3) and pass `--filters`. `--filters` replaces both `filters` and `train_filters` of the base (a key missing from the file is empty), so when the base has `filters` too, ask whether to keep them, and copy them into the file only when the user says so. Without `--filters`, the conditions of the base are used.
- Choose `--output` as `<directory of the base model.pkl>-refit` unless the user names a directory. Tell the user that an existing `model.pkl` there is overwritten. The directory of the base itself is refused (exit code 2).
- `--unavailable-sensors` and `--settings`: only when the user asks, as for `mloom optimize`. Without `--settings`, the settings recorded in the base are used.
- `warning:` lines on stderr name the columns of the base that could not be used. Exit codes are those of section 5.
- Explain the record as in section 6, and also:
  - `base.dropped_columns`, whenever it is not empty: the selected columns of the base that were dropped, with their `reason`. ALWAYS tell the user, since the model now behaves differently from the base.
  - `hpo` and `selection` are `null`: nothing was searched.
  - Do not compare `cv_scores` with `base.cv_scores` as they are: they come from different data (and the base selected the columns again on each fold).
  - `cv_scores` of the refit can be optimistic: its columns and parameters were chosen on the data of the base, which may overlap the test rows of the new folds.
  - Recommend checking the model on data after its training period with `mloom predict` (section 9).

## 8. Run `mloom optimize --base` (when the user asks to remake or improve an existing model)

Use it when the user asks to make an existing `model.pkl` again, or better, on new or more data, choosing the columns and parameters again. Keeping the way the base was made (the same columns and parameters) is `mloom refit` (section 7); choosing them again is `mloom optimize --base`. If the user's words do not tell which, ask.

```powershell
uv run --project "<clone>" mloom optimize <data.parquet> --base <base model.pkl> --output <directory> [<only the arguments the user wants to change>]
```

- Every argument of section 2 (`--target`, `--task`, `--model`, `--cv`, `--n-splits`, `--time-column`, `--group-column`, `--metric`, `--exclude-columns`, `--filters`, `--n-trials`, `--timeout`, `--seed`) comes from the record of the base unless given. Pass only the arguments the user says to change; never pass the others again.
  - `--exclude-columns` replaces the excluded columns of the base, and `--filters` replaces both `filters` and `train_filters` of the base: copy the values of the base the user wants to keep into them.
  - When the user changes `--target`, `--time-column` or `--group-column`, tell the user that the column the base used for it may now become a candidate feature, unless it is given in `--exclude-columns`.
- Read the record of the base first (`mloom show <base model.pkl>`). When it has `train_filters`, show them and ask whether to keep them, as for `mloom refit` (section 7).
- `--unavailable-sensors` and `--settings`: only when the user asks. Without `--settings`, the settings recorded in the base are used.
- Choose `--output` as `<directory of the base model.pkl>-new` unless the user names a directory. Tell the user that an existing `model.pkl` there is overwritten. The directory of the base itself is refused (exit code 2).
- `warning:` lines on stderr name the columns of the base that could not be used, and parameters of the base that were not tried first. Exit codes are those of section 5.
- Explain the record as in section 6, and also, from `base`:
  - `comparison.mean` and `comparison.std` (the difference of each fold between this run and how the base was made, on the same folds; positive means this run is better, whatever the metric). When the absolute value of `mean` is smaller than `std`, say that the runs cannot be told apart, and that refitting the base (section 7) is also a choice.
  - `cv_scores_on_this_data`: how the base was made, scored on the folds of this run. Say that it can favor the base: its columns and parameters were chosen on the data of the base, which may overlap the test rows of these folds.
  - `not_compared`, when it is not empty: why the base was not scored (`target`, `task`, `metric` differ from the base, or `columns`: no column of the base is left).
  - `base.dropped_columns`, whenever it is not empty: the selected columns of the base that could not be used in its score, with their `reason`.
  - `enqueued`: whether the parameters of the base were the first trial of the search.
  - The saved model is always the one of this search, even when the base scored better; mloom never picks the base instead.
  - Recommend checking the model on data after its training period with `mloom predict` (section 9).

## 9. Run `mloom predict` (when the user asks for predictions)

```powershell
uv run --project "<clone>" mloom predict <model.pkl> <data.parquet> --output <predictions.parquet> [--keep-columns <column> ...] [--shap] [--cm-actual-threshold <x> --cm-predicted-threshold <y> --cm-positive <above|below> | --positive-class <class> [--cm-predicted-threshold <p>]]
```

- Choose `--output` as `<directory of model.pkl>/predictions.parquet` unless the user names a file. Tell the user that an existing file there is overwritten.
- `--keep-columns`: only the input columns the user wants next to the predictions (such as `timestamp` or `lot`). The output has the same rows in the same order as the input.
- `--shap`: only when the user asks for the contributions of the columns. It can be slow with `random_forest` (about a minute for 1000 rows).
- Confusion matrix, only when the user gives its settings. Never choose a threshold, the positive side, or the positive class yourself; if the user gave only some of them, ask for the rest.
  - Regression: all three of `--cm-actual-threshold` (of the target), `--cm-predicted-threshold` (of the prediction), and `--cm-positive` (`above`: larger than the threshold is positive; `below`: smaller. A value equal to the threshold is negative).
  - Classification: `--positive-class`, and `--cm-predicted-threshold` (of the probability of that class; default 0.5) only when the user gives it.
- The input needs the candidate columns (`candidate_columns` of the run record, from `mloom show`) and the columns of `filters`. When it has the target column, it also needs `cv.time_column` and `cv.group_column` when they are set. mloom stops with exit code 2 otherwise.

## 10. Explain the predictions

Read the output Parquet (for example with `uv run --project "<clone>" python -c "import polars as pl; ..."`) and explain its columns:

| Column | Meaning |
| --- | --- |
| `--keep-columns` columns | copied from the input |
| `prediction` | the prediction. NaN (null in classification) on rows where `passed_filter` is false |
| `passed_filter` | whether the row meets `filters`. Rows outside `filters` are not predicted |
| `used_for_training` | true only for the rows the final model was trained on. null (cannot be told) when the input has no target column, or lacks a column of `train_filters` or its type does not match |
| `proba_<class>` | the probability of each class (classification) |
| `error` | prediction − actual, only on the scored rows; null elsewhere (regression) |
| `rmse`, `r2` | RMSE and R² over all the scored rows; the same value on every row (regression) |
| `cm` | `tp`, `fp`, `fn` or `tn` of the scored rows; null elsewhere (with the confusion-matrix arguments) |
| `accuracy`, `precision`, `recall` | computed from the `cm` counts of the scored rows; the same value on every row (with the confusion-matrix arguments) |
| `shap_base`, `shap_<column>` | the base value and the contribution of each selected column; in multiclass classification `shap_base_<class>` and `shap_<class>_<column>`. NaN on rows outside `filters` (with `--shap`) |

- The scored rows are only the rows whose `time_column` is later than the latest time of the rows the final model was trained on, whose target is neither missing nor infinite, and that were predicted (`passed_filter` true). Rows in the training period are not scored, because they would make the model look better than it is. Tell the user how many rows were scored (the rows where `error` or `cm` is not null).
- Why `rmse` and the other scores can be null:
  - The model was trained without `time_column` (`kfold` or `group` CV), so "after the training period" is undefined: all the score and `cm` columns are null.
  - The input has no target column: all of them are null.
  - No row is scored: `rmse` and `r2` are null. With one scored row, `r2` is null (R² needs two rows or more).
  - The denominator is 0: `precision` (no predicted positive) or `recall` (no actual positive) is null.
- The SHAP contributions are for the input of the estimator (after preprocessing and the column selection; standardized values for `ridge` and `pls`). The base value plus the contributions equals the prediction in regression, the log odds for `lightgbm` and `ridge` (logistic regression) in classification, and the probability for `random_forest`. Binary classification shows the contributions for `classes_[1]`.
- NEVER compute a score or a metric yourself (from `prediction`, `error`, or the training rows): report the values mloom wrote. Counting the values of `cm` or `passed_filter` to describe them is fine.
- With `--shap`, compare the contributions with domain intuition (section 4.6).
