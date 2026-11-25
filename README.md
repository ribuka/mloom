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
- 同じく `--settings` を省くと `config/settings.toml` を読みます（無ければ既定値）。欠損率のしきい値 `max_missing_rate` などを変えられます。例は [examples/settings.toml](examples/settings.toml) です。
- 目的変数が欠損している行と、無限大（+inf・-inf）がある行は学習から外し、行数を `run.json` の `data` に残します。
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

## 予測

```python
import pickle

import polars as pl

with open("outputs/quality-ridge/model.pkl", "rb") as file:
    pipeline = pickle.load(file)
pipeline.predict(pl.read_parquet("data/new.parquet"))
```

- 表には `run.json` の `candidate_columns` が含まれていれば足ります。列の順は問わず、余分な列は無視されます。
- 読む環境にも mloom が要ります（Pipeline の先頭が mloom の `ColumnSelector` のため）。
- `model.pkl` は pickle です。pickle は読むときに任意のコードを実行できるため、信頼できないファイルは読まないでください。

## 開発

```powershell
uv sync
uv run -m ruff check .
uv run -m ty check
uv run -m pytest --tb=short -q
```
