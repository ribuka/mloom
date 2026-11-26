# mloom

Parquet のデータと、ユーザーが選んだモデルを渡すと、使えないセンサの除外、特徴量選択、ハイパーパラメータ探索、最終学習までを自動で行い、学習済みの Pipeline を `model.pkl` に、実行の記録を `run.json` に保存する Python ライブラリです。

- Python API が本体です。エージェントは同梱の Skill（[skills/model-optimization/SKILL.md](skills/model-optimization/SKILL.md)）から薄い CLI を通して使います。
- 仕様と未決事項は [docs/spec.md](docs/spec.md) にあります。現在はたたき台（v0.1.0）です。

## 導入

mloom は PC に 1 つ clone し、Skill はジャンクションで各エージェントに見せます。利用側リポジトリには mloom を入れません。

1. 利用側リポジトリとは別の場所に clone します（例: `C:\Users\<you>\repos\mloom`。以下 `<clone>`）。
2. clone で `uv sync` を実行します。
3. 各エージェントの既定の Skill の置き場所に、`<clone>\skills\model-optimization` を指すジャンクションを作ります。

   ```powershell
   # Claude Code（~/.claude/skills/<名前>/SKILL.md を読む）
   New-Item -ItemType Directory -Force "$HOME\.claude\skills" | Out-Null
   New-Item -ItemType Junction -Path "$HOME\.claude\skills\model-optimization" -Target "<clone>\skills\model-optimization"
   # Codex（~/.agents/skills の下の SKILL.md を読む）
   New-Item -ItemType Directory -Force "$HOME\.agents\skills" | Out-Null
   New-Item -ItemType Junction -Path "$HOME\.agents\skills\model-optimization" -Target "<clone>\skills\model-optimization"
   ```

   - 置き場所は、[Claude Code のドキュメント](https://code.claude.com/docs/en/skills)と [Codex のドキュメント](https://developers.openai.com/codex/skills)（2026-11-26 時点で https://learn.chatgpt.com/docs/build-skills に転送）、それにこの手順で作ったジャンクションが読まれることで確かめました（[docs/spec.md](docs/spec.md) の「配布と実行」）。
   - Skill は、ジャンクションの実際のパスから `<clone>` を求め、利用側リポジトリの直下で `uv run --project "<clone>" mloom ...` を実行します。
4. 更新は、clone で `git pull` と `uv sync` を実行するだけです。ジャンクションの先が変わるので、Skill も mloom と同じ版になります。
5. 利用側リポジトリには mloom を入れません（`uv add mloom` などをしない）。`config/mloom.toml`・`config/unavailable-sensors.json`・データ・出力先は、利用側リポジトリに置きます。

ジャンクションを消すときは、`(Get-Item "$HOME\.claude\skills\model-optimization").Delete()` のように、ジャンクションだけを消します（`Remove-Item -Recurse` は先のファイルまで消すことがあります）。

## 試し方

利用側リポジトリの直下から、clone の mloom で実行します（データ・`config/`・出力先は利用側リポジトリからの相対パス）。

```powershell
uv run --project "<clone>" "<clone>\examples\make_sample_data.py"
uv run --project "<clone>" mloom optimize data/sample.parquet --target quality --task regression --model hist_gradient_boosting --cv timeseries --time-column timestamp --exclude-columns grade --unavailable-sensors "<clone>\examples\unavailable-sensors.json" --n-trials 20 --output outputs/quality-hgb
```

mloom の clone の中で試すだけなら、`--project` は要りません。

```powershell
uv sync
uv run examples/make_sample_data.py
uv run mloom optimize data/sample.parquet --target quality --task regression --model hist_gradient_boosting --cv timeseries --time-column timestamp --exclude-columns grade --unavailable-sensors examples/unavailable-sensors.json --n-trials 20 --output outputs/quality-hgb
```

- `outputs/quality-hgb/` に `model.pkl` と `run.json` ができます。
- `--unavailable-sensors` を省くと、カレントディレクトリの `config/unavailable-sensors.json` を読みます（無ければ除外なし）。
- 同じく `--settings` を省くと `config/mloom.toml` を読みます。欠損率のしきい値（`[columns] max_missing_rate`）、目的変数が欠損した行を統計量に使うか（`[preprocessing] use_unlabeled_rows`）、クリップ（`[clip]`）と補完（`[impute]`）の方式を変えられます（下の「前処理」）。
- `unavailable-sensors.json` と `mloom.toml` は、mloom を使う側のリポジトリの `config/` に置きます。

## 設定の既定値

使う側の `config/mloom.toml` に無いキーは、mloom リポジトリの次のファイルから読みます（上ほど優先）。

1. `config/mloom.local.toml`: その PC だけの既定値。git 対象外です。mloom を clone から実行したときだけ読みます。
2. [config/mloom.toml](config/mloom.toml): すべてのキーの既定値と説明。wheel にも同梱されます。
- 目的変数が欠損している行は学習と評価から外します。ただし既定では、欠損率と、前処理の統計量（クリップの上下限、補完の値、標準化の平均と標準偏差）の計算には使います。
- 目的変数が無限大（+inf・-inf）の行は外し、行数を `run.json` の `data.n_rows_with_infinite_target` に残します。説明変数の無限大は、学習でも予測でも欠損として扱います（列ごとの行数は `data.infinite_by_column`）。
- 分類は `--target grade --task classification --exclude-columns quality`、Group の CV は `--cv group --group-column lot` です。

## 入力

| 入力 | CLI | 選べるもの |
| --- | --- | --- |
| データ | 位置引数 | Parquet ファイル（API では polars の DataFrame も可） |
| 目的変数 | `--target` | 列名（1 個） |
| 回帰か分類か | `--task` | `regression`、`classification` |
| モデル | `--model` | `ridge`、`pls`（回帰のみ）、`random_forest`、`hist_gradient_boosting` |
| CV の方法 | `--cv` | `timeseries`（`--time-column` が必要）、`group`（`--group-column` が必要）、`kfold` |

その他の引数は `uv run mloom optimize --help` を見てください。

## 前処理

説明変数は、Pipeline の中で、無限大を欠損にする → クリップ → 補完 → 標準化（`ridge`・`pls` だけ）の順に処理します。方式は `config/mloom.toml` で選びます（キーと既定値は [config/mloom.toml](config/mloom.toml)）。

| 表 | `method` | 選べるもの |
| --- | --- | --- |
| `[clip]` | 既定 `none`（切らない） | `quantile`、`iqr`、`mad`、`sigma`。上下限は欠損と無限大を除いて計算し、ばらつきが 0 の列は切りません |
| `[impute]` | 既定 `auto` | `none`、`median`、`mean`、`most_frequent`、`constant`、`knn`、`iterative`。`auto` は `ridge`・`pls` では `median`、`hist_gradient_boosting`・`random_forest` では `none`（欠損のまま学習）です。`none` を `ridge`・`pls` で選ぶと終了コード 2 で止まります |

- 使った値は `run.json` の `settings`、最終モデルのクリップの上下限とクリップしなかった列は `preprocessing.clip` に残ります。

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
- 入力には、候補列（`run.json` の `candidate_columns`）と、`filters` に使う列が要ります。目的変数の列があるときは、`time_column`・`group_column` の列も要ります（学習に使った行の照合と、スコアのため）。無ければ終了コード 2 で止まります。列の順は問わず、余分な列は無視されます。

### スコア・混同行列・SHAP の寄与

```powershell
# 回帰: 実測が 0 より大きい、予測が 1 より大きいを陽性とする混同行列と、SHAP の寄与
uv run mloom predict outputs/quality-ridge/model.pkl data/new.parquet --output outputs/quality-ridge/predictions.parquet --cm-actual-threshold 0 --cm-predicted-threshold 1 --cm-positive above --shap
# 分類: クラス bad の確率が 0.3 より大きいを陽性とする混同行列
uv run mloom predict outputs/defect-rf/model.pkl data/new.parquet --output outputs/defect-rf/predictions.parquet --positive-class bad --cm-predicted-threshold 0.3
```

```python
predictions = mloom.predict(
    "outputs/quality-ridge/model.pkl",
    "data/new.parquet",
    shap=True,
    cm_actual_threshold=0,
    cm_predicted_threshold=1,
    cm_positive="above",
)
```

上の列の後ろに、次の順で列が続きます。

| 列 | 内容 | 出す条件 |
| --- | --- | --- |
| `error` | 予測 − 実測。スコアの対象の行だけ。ほかの行は null | 回帰のとき |
| `rmse`、`r2` | スコアの対象の行全体での RMSE と R²。全行に同じ値。対象が 0 行なら null（`r2` は 2 行未満で null） | 回帰のとき |
| `cm` | スコアの対象の行が、混同行列の `tp`・`fp`・`fn`・`tn` のどれか。ほかの行は null | 閾値（分類では `--positive-class`）を与えたとき |
| `accuracy`、`precision`、`recall` | スコアの対象の行の `cm` の数から計算した値。全行に同じ値。分母が 0 なら null | 同上 |
| `shap_base`、`shap_<列名>` | 基準値と、採用した列（`run.json` の `selected_columns`）ごとの SHAP の寄与。多値分類では、クラスごとに `shap_base_<クラス>`、`shap_<クラス>_<列名>`。フィルタの条件を満たさない行は NaN | `--shap` を付けたとき |

- スコアの対象の行は、時刻（`time_column`）が、最終モデルの学習に使った行の最も新しい時刻より後で、目的変数が欠損でも無限大でもなく、予測した（`passed_filter` が true の）行です。学習期間の中の行で測ると、成績が良く見えすぎるためです。
- `time_column` を指定せずに学習したモデル（`kfold`・`group` の CV）と、入力に目的変数の列が無いときは、スコアと混同行列の列はすべて null です。
- 回帰の混同行列は、`--cm-actual-threshold`・`--cm-predicted-threshold`・`--cm-positive`（`above` か `below`）を 3 つそろえて与えます。`above` は閾値より大きい値、`below` は閾値より小さい値を陽性とし、閾値ちょうどの値は陰性です。実測は実測の閾値で、予測は予測の閾値で判定します。
- 分類の混同行列は、`--positive-class` で陽性のクラスを与えます。予測は、そのクラスの確率が `--cm-predicted-threshold`（既定 0.5）より大きいときに陽性です。多値分類では「そのクラスか、それ以外か」で判定します。
- 引数がモデルに合わないとき（回帰で閾値を一部だけ与えた、回帰で `--positive-class` を与えた、学習時に無いクラスを与えた、など）は、終了コード 2 で止まります。
- SHAP の寄与は、推定器への入力（前処理と、採用した列の選択の後の値）に対する値です。基準値と寄与の合計は、回帰では予測値、分類では `hist_gradient_boosting` と `ridge`（ロジスティック回帰）は対数オッズ、`random_forest` は確率です。2 値分類は `classes_[1]` に対する値です。`random_forest` の寄与の計算は遅いことがあります（1000 行で約 1 分）。

### mloom を使わずに予測する

`model.pkl` は、mloom が無い環境でも読めます（学習時と同じ版の scikit-learn・numpy・scipy は要ります。版は `run.json` の `versions` にあります）。中身は `pipeline`、`filters`、`train_filters`、`training_rows`、`shap_background` の辞書です。

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
