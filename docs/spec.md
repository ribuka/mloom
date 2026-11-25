# mloom 仕様

Parquet のデータと、ユーザーが選んだモデルを渡すと、**列の除外 → 特徴量選択 → ハイパーパラメータ探索（HPO）→ 最終学習 → pickle 出力**までを自動で行う Python ライブラリ。エージェントからは、同じリポジトリに置く Skill（`SKILL.md`）から薄い CLI を通して使う。

- 本書はたたき台（2026-11-26 作成）。出典は会話記録「モデル最適化 Skill 化案」の抜粋と、その会話の最初のユーザーの入力。
- 「決定」は会話で合意した事項、「案」は本書で提案する未確定の事項。案は「未決事項」で確認してから決定に移す。

## 要件

### 決定事項

| 項目 | 内容 | 理由 |
| --- | --- | --- |
| 目的 | 少ない入力で、特徴量選択と HPO を自動化する | ユーザーが手を動かす量を減らすため |
| ユーザーの入力 | 次の 5 つ: 使うモデル（lgbm、pls など）、回帰か分類か、CV の方法（時系列、Group など）、データフレーム（Parquet）、目的変数 | 会話の最初にユーザーが示した入力のイメージ |
| タスク | 回帰と分類の両方を扱う。どちらかはユーザーが指定する | 同上 |
| CV の方法 | ユーザーが指定する。mloom がデータから推測しない | 同上 |
| 目的変数 | まずは 1 個だけ。複数の目的変数は後で検討する | 同上 |
| 派生列 | 工程知識で作る派生列（特徴量エンジニアリング）は扱わない。ユーザーに派生列の定義を求める仕組みも入れない | 列の計算は自動化できるが、どの列を作るべきかは入力だけでは決められない。定義をユーザーに求めると「少ない入力で済む」という目的と相反する |
| 列の選定 | 入力データにある列から、使う列を自動で選ぶ | 同上 |
| 使えないセンサ | 利用側リポジトリの `config/unavailable-sensors.json` にセンサが定義されていれば、そのセンサを使わない。特徴量選択の前に除外する | 故障や廃止などで使えないセンサを、モデルに入れないため |
| リポジトリ | 既存の `agent-skills` とは別のリポジトリ（本リポジトリ）にする | 決定論的な Python コードとその依存を持つため、汎用 Skill 集とは分ける |
| ライブラリ | Optuna ＋ scikit-learn | ユーザーがモデルを指定する前提なので、特徴量選択・CV・HPO を mloom 側でつなぐ。AutoML ライブラリにモデル選択まで任せる必要はない |
| モデル | 実行ごとにユーザーが選ぶ（例: `ridge`、`pls`、`lightgbm`、`random_forest`）。LightGBM に固定しない | 対象データごとに適したモデルが違うため |
| 出力 | 学習済みの Pipeline 全体を `model.pkl` に、実行の記録を `run.json` に保存する | モデル本体だけを保存すると、予測する側で列の選択や欠損補完を再現する必要がある。Pipeline ごと保存すれば同じ処理を通せる |
| 本体の形 | 処理の本体は決定論的な Python コードにする。Skill は「ユーザーの指定を CLI の引数に変換し、結果を説明する」だけにする | どの列を除外するか、どう評価するか、何を保存するかを、実行のたびに変わらないようにするため |
| 入口 | **Python API が本体**。Skill 向けに薄い CLI を用意する。CLI は Python API を呼ぶだけにする | 人は Notebook や既存コードから API で使い、エージェントは設定を渡して 1 回実行し、終了コードと `run.json` で結果を確かめる。センサ除外・特徴量選択・HPO の実装は 1 か所に置く |
| API の形 | `optimize_model(data, config)` が学習済み Pipeline と評価結果を返し、`result.save(output_dir)` で `model.pkl` と `run.json` を保存する | 同上 |
| データの渡し方 | API には Parquet のパス、または Polars の DataFrame を渡せる。CLI には Parquet のパスを渡す | 同上 |

### スコープ外

- 派生列の生成（ラグ、移動平均、差分、比など）と、その定義をユーザーに求める仕組み。
- モデルの種類の自動選択（モデルはユーザーが選ぶ）。
- 深層学習、GPU 前提のモデル。
- 推論サーバーやデプロイ。
- GUI。

## 利用の形

```text
人（Notebook / Python コード）──────────────┐
                                             ▼
エージェント ─ SKILL.md ─ CLI（mloom optimize）─▶ Python API（optimize_model）
                                             │
                                             ▼
                                   output_dir/model.pkl, run.json
```

| 使う場面 | 入口 | 使い方 |
| --- | --- | --- |
| Notebook や既存の Python コード | Python API | Parquet のパスか手元の Polars DataFrame を渡し、学習済み Pipeline とスコアをその場で確かめる |
| Skill やシェルから実行するエージェント | CLI | 引数を渡して 1 回実行し、終了コードと `run.json` で結果を確かめる。Python スクリプトを毎回生成しなくてよい |

## 入力

### データ

- 1 行が 1 サンプル、列がセンサ値などの説明変数と目的変数の、横持ちの表（案）。
- 形式: Parquet ファイル（API と CLI）、`polars.DataFrame`（API のみ）。
- 説明変数の候補は数値の列だけ（案）。文字列やカテゴリの列は除外して、`run.json` に理由を残す。

### 設定（案）

| 項目 | 必須 | 既定（案） | 内容 |
| --- | --- | --- | --- |
| `target` | ○ | なし | 目的変数の列名（1 個） |
| `task` | ○ | なし | `regression` または `classification` |
| `model` | ○ | なし | モデル名。「モデル」の表から選ぶ |
| `cv` | ○ | なし | CV の方法。下の「CV」の表から選ぶ |
| `n_splits` | | 5 | CV の分割数 |
| `time_column` | `cv` が `timeseries` のとき ○ | なし | 時刻の列。説明変数からは外す |
| `group_column` | `cv` が `group` のとき ○ | なし | グループの列（ロット、バッチなど）。説明変数からは外す |
| `metric` | | `task` ごとの既定 | 評価指標。回帰は `rmse`（既定）、`mae`、`r2`。分類は `log_loss`（既定）、`accuracy`、`f1_macro`、`roc_auc`（2 値のみ） |
| `exclude_columns` | | `[]` | 説明変数から外す列（ID 列など） |
| `unavailable_sensors` | | `config/unavailable-sensors.json` | 使えないセンサの定義ファイル。カレントディレクトリからの相対パス |
| `n_trials` | | 100 | HPO の試行数 |
| `timeout` | | なし | HPO の打ち切り時間（秒） |
| `seed` | | 0 | 乱数の種。CV の分割、Optuna の sampler、モデルに渡す |

### 使えないセンサの定義（`config/unavailable-sensors.json`）

決定事項: 利用側リポジトリの `config/unavailable-sensors.json` にあるセンサは使わない。特徴量選択の前に除外する。

形式（案）:

```json
{
  "sensors": [
    {"name": "TI-101", "reason": "2026-10 から故障中"},
    {"name": "FI-205"}
  ]
}
```

| 状況 | 扱い（案） |
| --- | --- |
| ファイルが無い | 除外なしで続ける。`run.json` に「ファイルなし」と残す |
| JSON として読めない、形式が違う | エラーで止める（終了コード 2） |
| 列名と一致する | その列を説明変数から外し、`run.json` に理由 `unavailable_sensor` で残す |
| 定義にあるがデータに無い | 続ける。`run.json` に警告として残す |
| 目的変数が定義にある | エラーで止める（終了コード 2） |

- 照合は列名の完全一致（案）。1 つのセンサから複数の列がある場合（`TI-101_mean` など）の照合は「未決事項」。

## 処理の流れ

1. データを読む（Polars）。
2. 列を除外する（下の「列の除外」）。除外した列と理由をすべて記録する。
3. CV の分割を決める（下の「CV」）。
4. Optuna で、特徴量選択とモデルのハイパーパラメータを同時に探索する（下の「特徴量選択」「HPO」）。各試行の評価は CV の平均スコア。
5. 最良の設定で、全データを使って Pipeline を学習し直す。
6. `model.pkl` と `run.json` を保存する。

学習データの情報が評価に漏れないように、教師ありの処理（特徴量選択、欠損の補完、標準化）はすべて Pipeline の中に置き、CV の分割ごとに学習データだけで fit する。

### 列の除外

特徴量選択の前に、目的変数を使わずに決められる列を外す。外した列は理由とともに `run.json` の `excluded_columns` に残す。

| 理由 | 対象 |
| --- | --- |
| `target` | 目的変数 |
| `role` | `time_column`、`group_column` |
| `user_excluded` | `exclude_columns` で指定した列 |
| `unavailable_sensor` | 使えないセンサ |
| `non_numeric` | 数値でない列（案） |
| `all_null` | すべて欠損の列 |
| `constant` | 欠損を除いて値が 1 種類の列 |
| `high_missing` | 欠損率がしきい値（案: 0.5）を超える列 |

- 目的変数が欠損している行は、学習と評価から外す（案）。外した行数を `run.json` に残す。

### CV

決定事項: CV の方法はユーザーが `cv` で指定する。mloom はデータから推測しない。

選べる方法（案）:

| `cv` | 分割 | 必要な列 |
| --- | --- | --- |
| `timeseries` | `time_column` の順に並べて `TimeSeriesSplit` | `time_column` |
| `group` | `GroupKFold`。分類では `StratifiedGroupKFold` | `group_column` |
| `kfold` | `KFold`（shuffle、`seed`）。分類では `StratifiedKFold` | なし |

- `timeseries` で行の並び順に頼らず `time_column` を必須にするのは、Parquet の行の順が時刻順とは限らないため（案）。
- 必要な列が無い、またはデータに無いときは、学習を始める前にエラーで止める（終了コード 2）。

### 特徴量選択（案）

- Pipeline の中で、各列を目的変数との関係でランク付けし（回帰は `f_regression`、分類は `f_classif`）、上位 `k` 列を使う（`SelectKBest`）。
- `k` は HPO で探索するハイパーパラメータの 1 つにする（1 〜 候補列数、対数スケール）。
- 最終学習で選ばれた列を「採用した列」として `run.json` に残す。
- ランク付けの方法（`mutual_info_regression`、モデルの重要度、相関の高い列の間引きなど）は「未決事項」。

### HPO（案）

- Optuna の `TPESampler(seed=seed)`、試行数 `n_trials`、打ち切り時間 `timeout`。
- 目的関数は CV の平均スコア。`rmse`、`mae`、`log_loss` は最小化、`r2`、`accuracy`、`f1_macro`、`roc_auc` は最大化。
- 探索空間はモデルごとに mloom 側で固定する（下の「モデル」）。Skill やユーザーには探索空間を書かせない。

### モデル（案）

モデル名と `task` から、推定器・前処理・探索空間を引くレジストリを mloom の中に持つ。

| `model` | 回帰の推定器 | 分類の推定器 | 前処理 | 探索するパラメータ（案） |
| --- | --- | --- | --- | --- |
| `ridge` | `Ridge` | `LogisticRegression`（L2） | 中央値で補完、標準化 | 回帰は `alpha`、分類は `C`: いずれも 1e-4 〜 1e3（対数） |
| `pls` | `PLSRegression` | 未決（PLS-DA） | 中央値で補完、標準化 | `n_components`: 1 〜 min(20, `k`) |
| `random_forest` | `RandomForestRegressor` | `RandomForestClassifier` | 中央値で補完 | `n_estimators`、`max_depth`、`min_samples_leaf`、`max_features` |
| `lightgbm` | `LGBMRegressor` | `LGBMClassifier` | 中央値で補完 | `num_leaves`、`learning_rate`、`n_estimators`、`min_child_samples`、`subsample`、`colsample_bytree`、`reg_lambda` |

- Pipeline の構成: 入力列の選択 → 欠損の補完 → （標準化）→ 特徴量選択 → 推定器。
- 分類の `ridge` に `RidgeClassifier` ではなく `LogisticRegression` を当てるのは、`log_loss` と `roc_auc` に要る確率（`predict_proba`）を出せるため（案）。
- `lightgbm` は追加の依存（`mloom[lightgbm]`）にする（案）。入っていない環境で `model: lightgbm` を指定したら、インストール方法を示してエラーで止める。
- `task` に対応しないモデルを指定したら、学習を始める前にエラーで止める（終了コード 2）。

## 出力

`output_dir` に次を保存する。

| ファイル | 内容 |
| --- | --- |
| `model.pkl` | 学習済みの Pipeline 全体（前処理・特徴量選択・モデル）。予測時も同じ処理を通す |
| `run.json` | 採用した列、除外した列と理由、最良のパラメータ、CV の分割、スコア、seed、ライブラリの版など |

### `model.pkl`

- 標準ライブラリの `pickle` で保存する。
- 読むときは `pickle.load` の結果の `predict` に、学習時の入力列を含む表を渡す。列の選択は Pipeline が行う。
- pickle は保存したときと同じ版の scikit-learn などで読む前提。版は `run.json` に残す。
- 信頼できない pickle を読まない（pickle は読むときに任意のコードを実行できる）。README に書く。

### `run.json`（案）

```json
{
  "mloom_version": "0.1.0",
  "created_at": "2026-11-26T10:00:00+09:00",
  "data": {"path": "data/train.parquet", "n_rows": 12000, "n_rows_used": 11950, "n_columns": 320},
  "target": "quality",
  "task": "regression",
  "model": "ridge",
  "metric": "rmse",
  "seed": 0,
  "unavailable_sensors": {
    "path": "config/unavailable-sensors.json",
    "found": true,
    "not_in_data": ["FI-205"]
  },
  "excluded_columns": [
    {"name": "TI-101", "reason": "unavailable_sensor"},
    {"name": "lot_id", "reason": "role"}
  ],
  "candidate_columns": ["..."],
  "selected_columns": ["..."],
  "cv": {"method": "timeseries", "splitter": "TimeSeriesSplit", "n_splits": 5, "time_column": "timestamp"},
  "hpo": {"sampler": "TPESampler", "n_trials": 100, "n_trials_completed": 100, "timeout": null},
  "best_params": {"k": 42, "alpha": 0.31},
  "cv_scores": {"folds": [0.81, 0.79, 0.84, 0.80, 0.82], "mean": 0.812, "std": 0.017},
  "versions": {"python": "3.13.x", "scikit-learn": "...", "optuna": "...", "polars": "..."}
}
```

## Python API（案）

```python
import mloom

result = mloom.optimize_model(
    "data/train.parquet",          # または polars.DataFrame
    mloom.Config(
        target="quality",
        task="regression",
        model="ridge",
        cv="timeseries",
        time_column="timestamp",
    ),
)
result.pipeline      # 学習済み Pipeline
result.cv_scores     # CV のスコア
result.save("outputs/ridge")  # model.pkl と run.json
```

- 設定は `mloom.Config`（dataclass）で渡す（案）。CLI の引数と 1 対 1 に対応させる。
- 不正な入力（目的変数が無い、モデル名が無い、候補列が 0 列など）は、学習を始める前に例外を投げる。

## CLI（案）

```powershell
uv run mloom optimize data/train.parquet --target quality --task regression --model ridge --cv timeseries --time-column timestamp --output outputs/ridge
```

- CLI は引数を `Config` に変換し、`optimize_model` と `result.save` を呼ぶだけにする。
- 標準出力の最後に `run.json` のパスを出す。進捗やログは標準エラーに出す。

| 終了コード | 意味 |
| --- | --- |
| 0 | 成功。`model.pkl` と `run.json` を保存した |
| 1 | 想定外のエラー |
| 2 | 入力の誤り（引数、データ、`unavailable-sensors.json`） |

## Skill（案）

- 本リポジトリの `skills/model-optimization/SKILL.md` に置く。
- 役割は 2 つだけ: ユーザーの指定を CLI の引数に変換して実行する、`run.json` を読んで結果を説明する（採用した列、除外した列と理由、スコア、最良のパラメータ）。
- どの列を除外するか、どう評価するか、何を保存するかを Skill で決めない。それらは Python 側で固定する。

## 非機能要件（案）

| 項目 | 内容 |
| --- | --- |
| 再現性 | 同じデータ・設定・`seed`・ライブラリの版なら、同じ列と同じパラメータになる（並列実行時の扱いは「未決事項」） |
| 対象 OS | Windows 11（PowerShell）。CI は Linux |
| 計算資源 | CPU のみ。GPU は使わない |
| 実行時間 | `n_trials` と `timeout` で上限を決められる |

## 技術スタック（案）

| 用途 | ライブラリ |
| --- | --- |
| データの読み込みと列の除外 | polars |
| Pipeline、CV、特徴量選択、モデル | scikit-learn |
| HPO | optuna |
| LightGBM（追加の依存） | lightgbm |
| パッケージ管理 | uv |
| lint、型チェック、テスト | ruff、ty、pytest |

## パッケージ構成（案）

| モジュール | 責務 |
| --- | --- |
| `mloom` | 公開 API の再エクスポート（`optimize_model`、`Config`、`Result`） |
| `mloom.config` | `Config` の定義と検証 |
| `mloom.data` | データの読み込み |
| `mloom.sensors` | `unavailable-sensors.json` の読み込みと照合 |
| `mloom.columns` | 列の除外と理由の記録 |
| `mloom.models` | モデルのレジストリ（推定器、前処理、探索空間） |
| `mloom.cv` | CV の分割の決定 |
| `mloom.optimize` | `optimize_model`：HPO と最終学習 |
| `mloom.result` | `Result` と保存（`model.pkl`、`run.json`） |
| `mloom.cli` / `mloom.__main__` | CLI |

## 未決事項

| # | 論点 | 案 |
| --- | --- | --- |
| 1 | 分類の `pls` | PLS-DA（目的変数を one-hot にして `PLSRegression`、最大の列をクラスとする）を mloom で実装するか、分類では `pls` を選べないことにするか |
| 2 | 分類の範囲 | 2 値と多値の両方。クラスの不均衡への対応（`class_weight` を探索するか）は後回し |
| 3 | CV で選べる方法 | `timeseries`、`group`、`kfold` の 3 つ。時系列かつグループ（ロットごとに時刻順）の分割が要るか |
| 4 | `timeseries` の時刻の列 | `time_column` を必須にする。行の並び順を時刻順とみなす指定も許すか |
| 5 | 特徴量選択の方法 | `SelectKBest` の `k` を HPO で探索。相関の高い列の間引きを足すか |
| 6 | `unavailable-sensors.json` の形式と照合 | `{"sensors": [{"name", "reason"}]}`、列名の完全一致。接頭辞やパターンでの照合が要るか |
| 7 | 数値でない列の扱い | 除外する（エンコードしない） |
| 8 | 欠損の扱い | 全モデルで中央値の補完。LightGBM などは欠損をそのまま渡すか |
| 9 | 選べるモデル | 表の 4 つで始める。API から任意の推定器と探索空間を渡せるようにするか |
| 10 | 予測の入口 | `pickle.load` だけで足りるか、`mloom predict` の CLI も用意するか |
| 11 | HPO の試行の記録 | `run.json` は最良の試行だけ。全試行を `trials.csv` に出すか |
| 12 | Skill の配布方法 | 利用側への Skill の入れ方（コピー、プラグイン、`agent-skills` からの参照） |
| 13 | 利用環境への導入 | オフラインか、PyPI のミラーがあるか。git の依存で入れられるか |
| 14 | 並列実行 | Optuna の `n_jobs` やモデルの `n_jobs` を使うか（再現性とのトレードオフ） |
| 15 | 複数の目的変数 | 「まずは 1 個」の次の段階で、目的変数ごとに別の Pipeline を作るか、1 つのモデルで同時に予測するか |

## 完了条件（v0.1、案）

- Parquet と 5 つの入力（データ、`target`、`task`、`model`、`cv`）を渡すと、API と CLI のどちらからでも `model.pkl` と `run.json` が保存される。
- 回帰と分類、`cv` の 3 つの方法のそれぞれで動く。
- `config/unavailable-sensors.json` にあるセンサの列が、採用した列に入らず、`run.json` に理由つきで残る。
- `model.pkl` を `pickle.load` して、学習時と同じ列を持つ表に `predict` できる。
- 同じ入力と `seed` で 2 回実行すると、採用した列と最良のパラメータが一致する。
- 表の 4 つのモデルそれぞれで、上の 4 つがテストで確かめられている。
