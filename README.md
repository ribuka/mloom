# mloom

Parquet のデータと、ユーザーが選んだモデルを渡すと、使えないセンサの除外、特徴量選択、ハイパーパラメータ探索、最終学習までを自動で行い、学習済みの Pipeline を `model.pkl` に、実行の記録を `run.json` に保存する Python ライブラリです。

- Python API が本体です。エージェントは同梱の Skill（[skills/model-optimization/SKILL.md](skills/model-optimization/SKILL.md)）から薄い CLI を通して使います。
- 仕様と未決事項は [docs/spec.md](docs/spec.md) にあります。現在はたたき台（v0.1.0）です。

## 試し方

```powershell
uv sync
uv run examples/make_sample_data.py
uv run mloom optimize data/sample.parquet --target quality --task regression --model lightgbm --cv timeseries --time-column timestamp --exclude-columns grade --unavailable-sensors examples/unavailable-sensors.json --n-trials 20 --output outputs/quality-lightgbm
```

- `outputs/quality-lightgbm/` に `model.pkl` と `run.json` ができます。
- `--unavailable-sensors` を省くと、カレントディレクトリの `config/unavailable-sensors.json` を読みます（無ければ除外なし）。
- 同じく `--settings` を省くと `config/mloom.toml` を読みます。欠損率のしきい値 `max_missing_rate` と、目的変数が欠損した行を統計量に使うか `use_unlabeled_rows` を変えられます。
- `unavailable-sensors.json` と `mloom.toml` は、mloom を使う側のリポジトリの `config/` に置きます。

## 設定の既定値

使う側の `config/mloom.toml` に無いキーは、mloom リポジトリの次のファイルから読みます（上ほど優先）。

1. `config/mloom.local.toml`: その PC だけの既定値。git 対象外です。mloom を clone から実行したときだけ読みます。
2. [config/mloom.toml](config/mloom.toml): すべてのキーの既定値と説明。wheel にも同梱されます。
- 目的変数が欠損している行は学習と評価から外します。ただし既定では、欠損率と、補完の中央値・標準化の平均と標準偏差の計算には使います。
- 無限大（+inf・-inf）がある行は外し、行数を `run.json` の `data` に残します。
- 分類は `--target grade --task classification --exclude-columns quality`、Group の CV は `--cv group --group-column lot` です。

## 入力

| 入力 | CLI | 選べるもの |
| --- | --- | --- |
| データ | 位置引数 | Parquet ファイル（API では polars の DataFrame も可） |
| 目的変数 | `--target` | 列名（1 個） |
| 回帰か分類か | `--task` | `regression`、`classification` |
| モデル | `--model` | `ridge`、`pls`（回帰のみ）、`random_forest`、`lightgbm` |
| CV の方法 | `--cv` | `timeseries`（`--time-column` が必要）、`group`（`--group-column` が必要）、`kfold` |

`lightgbm` を使うには追加の依存 `mloom[lightgbm]` が要ります。その他の引数は `uv run mloom optimize --help` を見てください。

## Python API

```python
import mloom

result = mloom.optimize_model(
    "data/sample.parquet",  # または polars.DataFrame
    mloom.Config(
        target="quality",
        task="regression",
        model="ridge",
        cv="timeseries",
        time_column="timestamp",
        exclude_columns=("grade",),
        n_trials=20,
    ),
)
print(result.cv_scores, result.selected_columns)
result.save("outputs/quality-ridge")
```

## 行のフィルタ

`--filters` に JSON ファイルを渡すと、学習と予測に使う行を絞れます（API では `Config(filters=..., train_filters=...)`）。

```json
{
  "filters": [{"column": "FI-101", "op": "gt", "value": 0}],
  "train_filters": [{"column": "timestamp", "op": "lt", "value": "2026-10-01T00:00:00"}]
}
```

- `filters` は学習と予測の両方に効きます。`train_filters` は学習だけに効き、予測では行を絞りません。
- 条件どうしは AND です。`op` は `eq`・`ne`・`lt`・`le`・`gt`・`ge`・`is_in`・`is_between`・`is_null`・`is_not_null`・`str_contains_all`・`str_contains_any` です。意味とエラーになる場合は [docs/spec.md](docs/spec.md) の「行のフィルタ」を見てください。
- 外した行数は `run.json` の `data`（`n_rows_filtered_out`・`n_rows_train_filtered_out`）に残ります。

## 予測

```powershell
uv run mloom predict outputs/quality-ridge/model.pkl data/new.parquet --output outputs/quality-ridge/predictions.parquet --keep-columns timestamp
```

```python
predictions = mloom.predict("outputs/quality-ridge/model.pkl", "data/new.parquet", keep_columns=("timestamp",))
```

- 入力と同じ行数・同じ順で、`--keep-columns` の列、`prediction`、`passed_filter`、`used_for_training`、分類では `proba_<クラス>` を返します。
- `filters` の条件を満たさない行は予測せず、`prediction` を NaN（分類では null）にします。
- `used_for_training` は、最終モデルの学習に使った行だけが true です。入力に目的変数の列か `train_filters` の列が無いときは null です。
- 入力には、候補列（`run.json` の `candidate_columns`）と、`filters` に使う列が要ります。目的変数の列があるときは、`time_column`・`group_column` の列も要ります（学習に使った行の照合のため）。無ければ終了コード 2 で止まります。列の順は問わず、余分な列は無視されます。

### mloom を使わずに予測する

`model.pkl` は、mloom が無い環境でも読めます（学習時と同じ版の scikit-learn・numpy・scipy は要ります。版は `run.json` の `versions` にあります）。中身は `pipeline`、`filters`、`train_filters`、`training_rows` の辞書です。

```python
import pickle

import numpy as np
import polars as pl

with open("outputs/quality-ridge/model.pkl", "rb") as file:
    bundle = pickle.load(file)
data = pl.read_parquet("data/new.parquet")
# filters を AND で評価する（eq・lt などの比較だけの例）。欠損の行は条件を満たさない
conditions = [getattr(pl.col(f["column"]), f["op"])(f["value"]).fill_null(False) for f in bundle["filters"]]
passed = data.with_columns(passed=pl.all_horizontal(pl.lit(True), *conditions))["passed"]
prediction = np.full(data.height, np.nan)  # 条件を満たさない行は NaN
if passed.any():  # 0 行では predict を呼ばない
    prediction[passed.to_numpy()] = bundle["pipeline"].predict(data.filter(passed))
```

- `pipeline` の `predict` に渡す表には、候補列があれば足ります（列の順は問わず、余分な列は無視されます）。`filters` を評価するには、その列も要ります。
- 例は、比較の演算子（`eq`・`lt` など）の `filters` だけを扱います。ほかの演算子、日時の値（ISO 8601 の文字列を列の型に変換して比べる）、浮動小数の列の NaN（`mloom predict` では欠損として条件を満たさない）は、[docs/spec.md](docs/spec.md) の「行のフィルタ」に合わせて書き足してください。
- 分類では、`prediction` を `np.full(data.height, None, dtype=object)` で用意します。
- `model.pkl` は pickle です。pickle は読むときに任意のコードを実行できるため、信頼できないファイルは読まないでください。

## 開発

```powershell
uv sync
uv run -m ruff check .
uv run -m ty check
uv run -m pytest --tb=short -q
```
