# mloom 仕様

Parquet のデータと、ユーザーが選んだモデルを渡すと、**列の除外 → 特徴量選択 → ハイパーパラメータ探索（HPO）→ 最終学習 → pickle 出力**までを自動で行う Python ライブラリ。エージェントからは、同じリポジトリに置く Skill（`SKILL.md`）から薄い CLI を通して使う。

- 本書はたたき台（2026-11-26 作成）。出典は会話記録「モデル最適化 Skill 化案」の抜粋と、その会話の最初のユーザーの入力。
- 「決定」は会話で合意した事項、「案」は本書で提案する未確定の事項。案は「未決事項」で確認してから決定に移す。
- 案のまま、試しに動かせるたたき台を実装した（2026-11-26、v0.1.0）。実装で値を決めた箇所は、その旨を書いている。
- たたき台の実装の後に、未決 8・10・12・13・16・18 を決め、行のフィルタ、クリップ、補完の方式の選択、`hist_gradient_boosting` を足すことを決めた（2026-11-26 ユーザー判断）。まだ実装していない箇所には「（未実装）」と書き、実装は後続の PR で行う。
- そのうち、行のフィルタ、mloom に依存しない `model.pkl`、`mloom predict` の基本（スコア・混同行列・SHAP を除く）を実装した（2026-11-26、Issue #4）。
- `mloom predict` のスコア・混同行列・SHAP の寄与を実装した（2026-11-26、Issue #6）。

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
| モデル | 実行ごとにユーザーが選ぶ（例: `ridge`、`pls`、`hist_gradient_boosting`、`random_forest`）。1 つのモデルに固定しない | 対象データごとに適したモデルが違うため |
| 勾配ブースティング | LightGBM をやめ、scikit-learn の `HistGradientBoostingRegressor`・`HistGradientBoostingClassifier`（`hist_gradient_boosting`）に置き換える（2026-11-26 ユーザー判断。2026-11-26 に実装） | `model.pkl` を読む環境に lightgbm を要らなくするため。同じヒストグラム型の勾配ブースティングで、欠損をそのまま扱える。結果は LightGBM と一致しない（サンプルデータの時系列 5 分割で、既定値の RMSE は LightGBM 3.450、HistGradientBoosting 3.487）。行のサブサンプリング（LightGBM の `subsample`）は無い |
| 出力 | 学習済みの Pipeline 全体と行のフィルタの定義を `model.pkl` に、実行の記録を `run.json` に保存する | モデル本体だけを保存すると、予測する側で列の選択や欠損補完を再現する必要がある。Pipeline ごと保存すれば同じ処理を通せる |
| `model.pkl` の依存 | `model.pkl` は mloom が無い環境でも読めるようにする。Pipeline には scikit-learn・numpy などの部品と関数だけを入れ、mloom のクラスや関数を入れない（2026-11-26 ユーザー判断。未決 16） | 予測する環境に mloom を入れなくて済むようにするため |
| 予測の入口 | `pickle.load` に加えて、`mloom predict` の CLI と、同じ処理の Python API を用意する（2026-11-26 ユーザー判断。未決 10。スコア・混同行列・SHAP は 2026-11-26 に Issue #6 で実装） | フィルタの適用と、フィルタの外の行を NaN にする処理を 1 か所にまとめ、Skill から予測まで実行できるようにするため |
| 行のフィルタ | 列と値の条件で、学習と予測に使う行を絞れる。条件は AND だけで、定義を `model.pkl` と `run.json` に残す（2026-11-26 ユーザー判断。「行のフィルタ」） | 運転条件などで、モデルを使う範囲を決めるため |
| 前処理の方式 | 説明変数のクリップと補完の方式を、設定ファイルで選べる（2026-11-26 ユーザー判断。2026-11-26 に実装。「前処理」） | データによって適した前処理が違うため |
| 本体の形 | 処理の本体は決定論的な Python コードにする。Skill は「ユーザーの指定を CLI の引数に変換し、結果を説明する」だけにする | どの列を除外するか、どう評価するか、何を保存するかを、実行のたびに変わらないようにするため |
| 入口 | **Python API が本体**。Skill 向けに薄い CLI を用意する。CLI は Python API を呼ぶだけにする | 人は Notebook や既存コードから API で使い、エージェントは設定を渡して 1 回実行し、終了コードと `run.json` で結果を確かめる。センサ除外・特徴量選択・HPO の実装は 1 か所に置く |
| API の形 | `optimize_model(data, config)` が学習済み Pipeline と評価結果を返し、`result.save(output_dir)` で `model.pkl` と `run.json` を保存する | 同上 |
| データの渡し方 | API には Parquet のパス、または Polars の DataFrame を渡せる。CLI には Parquet のパスを渡す | 同上 |
| 無限大 | 説明変数の +inf・-inf は、学習時も予測時も欠損とみなす。目的変数に +inf・-inf がある行は、学習から外す（2026-11-26 ユーザー判断で、「説明変数に無限大がある行は行ごと外す」から変更。2026-11-26 に実装。未決 18） | 予測時は行ごとに値を返すので行を外せず、学習時だけ行を外すと、学習と予測で扱いが食い違うため。プロセスデータの無限大は、ゼロ除算やヒストリアンのエラー値など「値が正しく取れていない」ことが多く、欠損として扱うのが一般的。無限大が多い列は欠損率のしきい値で除外される |
| 欠損率のしきい値 | 説明変数を除外する欠損率のしきい値を、利用側リポジトリの `config/mloom.toml` で変えられる（2026-11-26 ユーザー判断。「設定ファイル」） | データによって適したしきい値が違うため |
| 目的変数が欠損した行 | 説明変数だけから計算する統計量（クリップの上下限、補完の値、標準化の平均・標準偏差）に、目的変数が欠損した行も使う。欠損率は「目的変数がある行」と「全行」の両方で見て、どちらかがしきい値を超えたら除外する。`config/mloom.toml` で使わないようにも切り替えられる（既定は使う）（2026-11-26 ユーザー判断） | ソフトセンサのように分析値がまばらなデータでは、統計量が安定するため。欠損率は、学習で実際の値から学べる量（目的変数がある行）と、予測時に補完に頼る量（全行）の両方を満たす列だけを残すため。非定常運転などで使いたくないデータもあるので切り替えられるようにする |

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
                                             │
                      ┌──────────────────────┴───────────────────────┐
                      ▼                                              ▼
       mloom predict / mloom.predict                   pickle.load（mloom なしで読める）
```

| 使う場面 | 入口 | 使い方 |
| --- | --- | --- |
| Notebook や既存の Python コード | Python API | Parquet のパスか手元の Polars DataFrame を渡し、学習済み Pipeline とスコアをその場で確かめる |
| Skill やシェルから実行するエージェント | CLI | 引数を渡して 1 回実行し、終了コードと `run.json` で結果を確かめる。Python スクリプトを毎回生成しなくてよい |
| mloom が無い環境での予測 | `pickle.load` | `model.pkl` の Pipeline とフィルタの定義を取り出し、予測する側でフィルタを適用してから `predict` する（「`model.pkl`」） |

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
| `filters` | | `[]` | 学習と予測に効く行のフィルタの条件（下の「行のフィルタ」）。CLI では、条件を書いた JSON ファイルのパスを `--filters` で渡す（2026-11-26 ユーザー判断。未決 19）。ファイルには `filters` と `train_filters` を書く |
| `train_filters` | | `[]` | 学習だけに効く行のフィルタの条件（下の「行のフィルタ」）。CLI では `--filters` のファイルに書く |
| `unavailable_sensors` | | `config/unavailable-sensors.json` | 使えないセンサの定義ファイル。カレントディレクトリからの相対パス |
| `settings` | | `config/mloom.toml` | 利用側の設定ファイル（下の「設定ファイル」）。カレントディレクトリからの相対パス。`None` なら mloom 側の既定値だけを読む |
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

### 設定ファイル（`config/mloom.toml`）

決定事項（2026-11-26 ユーザー判断）:

- 処理の調整値を、利用側リポジトリの `config/mloom.toml` で変えられる。利用側の `config/settings.toml` などと衝突しないように、ファイル名に `mloom` を入れる。
- 利用側で定義されていないキーは、mloom リポジトリ（Skill 側）の `config/mloom.local.toml`、`config/mloom.toml` の順に探す。`config/mloom.local.toml` は git 対象外。

読む順（上ほど優先。キーごとに、最初に見つかった値を使う）:

| 順 | ファイル | 役割（`run.json` の `settings.files[].role`） | 読む条件 |
| --- | --- | --- | --- |
| 1 | 利用側の `config/mloom.toml`（実行時のカレントディレクトリから。`--settings` で別のファイルも指定できる） | `project` | 常に |
| 2 | mloom リポジトリの `config/mloom.local.toml`（git 対象外） | `local_defaults` | mloom を clone から実行したとき |
| 3 | mloom リポジトリの `config/mloom.toml` | `defaults` | 常に。clone が無いとき（wheel から入れたとき）は、wheel に同梱した写し（`mloom/defaults.toml`）を読む |

- 既定値は 3 のファイルだけに書く。コードには既定値を持たない（2 か所に書くと食い違うため）。3 のファイルにすべてのキーが無ければエラーで止める（終了コード 1）。
- 「clone から実行した」かどうかは、読み込んだ mloom パッケージが `<リポジトリ>/src/mloom` にあり、`<リポジトリ>/config/mloom.toml` があることで判定する。
- 2 は、その PC だけの既定値（例: 扱うデータの傾向に合わせたしきい値）を、利用側の各リポジトリに書かずに変えるためのもの。
- `config/unavailable-sensors.json` は名前を変えない。中身は mloom の設定ではなくプラントの事実で、他の分析やツールからも読めるように汎用の名前にしておく。

キー（3 のファイルの内容。`[clip]`・`[impute]` は 2026-11-26 に実装。意味は「前処理」）:

```toml
[columns]
# 欠損率がこの値を超える説明変数を除外する（high_missing）。+inf・-inf も欠損として数える。0 〜 1。既定 0.5
max_missing_rate = 0.5

[preprocessing]
# 目的変数が欠損した行を、欠損率と、クリップ・補完・標準化の統計量に使うか。既定 true
use_unlabeled_rows = true

[clip]
# none、quantile、iqr、mad、sigma。既定 none
method = "none"
# quantile の下側と上側の分位点。0 <= lower_quantile < upper_quantile <= 1。既定 0.01 と 0.99
lower_quantile = 0.01
upper_quantile = 0.99
# iqr の k（Q1 - k·IQR 〜 Q3 + k·IQR）。正の数。既定 1.5
iqr_k = 1.5
# mad の k（中央値 ± k × 1.4826 × MAD）。正の数。既定 3.0
mad_k = 3.0
# sigma の k（平均 ± k·σ。σ は母標準偏差）。正の数。既定 3.0
sigma_k = 3.0

[impute]
# auto、none、median、mean、most_frequent、constant、knn、iterative。既定 auto（モデルごとの既定）
method = "auto"
# constant で埋める値。有限の数値。既定 0.0
fill_value = 0.0
# knn の近傍の数。1 以上の整数。既定 5
n_neighbors = 5
```

- キーは「表とキーの名前の組」で区別する（2026-11-26 実装で決定）。`[clip] method` と `[impute] method` は別のキーで、ファイルごとの優先順位もキーごとに決まる。
- `lower_quantile < upper_quantile` は、1 つのファイルの中で両方が書かれていれば、そのファイルで確かめる。加えて、キーごとに併合した後の値でも確かめる（2026-11-26 実装で決定）。片方だけを書くファイルがあると、併合した後に初めて食い違うため（例: 利用側で `lower_quantile = 0.995` だけを書く）。
- `fill_value` は有限の数値に限る（2026-11-26 実装で決定）。TOML は `nan`・`inf` を書けるが、`nan` で埋めると補完にならないため。

| 状況 | 扱い |
| --- | --- |
| 1 や 2 のファイルが無い | 無視して次を探す。`run.json` の `settings.files[].found` が `false` になる |
| キーが無い | 次のファイルを探す |
| どのファイルでも、TOML として読めない、知らない表やキーがある、値が範囲外 | エラーで止める（終了コード 2）。打ち間違いで黙って既定値になるのを防ぐため |

- 使った値と、各キーを読んだファイルを `run.json` の `settings` に残す（`sources`）。値も `sources` も、TOML と同じく表ごとに入れ子にする（2026-11-26 実装で決定。`settings.columns.max_missing_rate`、`settings.sources.clip.method` など）。`[clip] method` と `[impute] method` のように、表が違えば同じ名前のキーがあるため。

## 処理の流れ

1. データを読む（Polars）。
2. 学習に使う行を選ぶ（下の「行のフィルタ」「行の除外」）。
3. 列を除外する（下の「列の除外」）。除外した列と理由をすべて記録する。
4. CV の分割を決める（下の「CV」）。
5. Optuna で、特徴量選択とモデルのハイパーパラメータを同時に探索する（下の「特徴量選択」「HPO」）。各試行の評価は CV の平均スコア。
6. 最良の設定で、学習に使う行のすべてを使って Pipeline を学習し直す。
7. `model.pkl` と `run.json` を保存する。

評価側のデータの情報が学習に漏れないように、次のように fit する。

- 特徴量選択のランク付けとモデル（目的変数を使う）は、CV の分割ごとに、学習側の目的変数がある行だけで計算・fit する。ランク付けは Pipeline の外で、分割ごとに 1 回だけ行う（「特徴量選択」）。モデルは Pipeline の中で fit する。
- クリップの上下限と、補完・標準化（目的変数を使わない）は、CV の分割ごとに「学習側の目的変数がある行」と「学習側に当たる目的変数が欠損した行」（「CV」）で先に計算・fit して固定し（scikit-learn の `FrozenEstimator`）、Pipeline に入れる。試行ごとには fit し直さない。
- 最終モデルのクリップ・補完・標準化は、目的変数がある行と、目的変数が欠損した行のすべてで fit する。ランク付けは、目的変数がある行のすべてで行う。

### 行のフィルタ

決定事項（2026-11-26 ユーザー判断。2026-11-26 に実装、`mloom.filters`）:

- 列と値の条件（`filters`）を満たす行だけを、学習と予測に使う。条件どうしは AND だけ（OR は書けない）。
- 定義は `model.pkl` と `run.json` の両方に残す。予測時に、条件を満たさない行の予測値は NaN にする（「`model.pkl`」「予測」）。
- 定義は polars の式（`pl.Expr`）ではなく、列名・演算子・値を並べたデータにする。読むのに polars も要らず、`run.json` にもそのまま書けるため。`pl.Expr` の pickle は polars の版をまたいで形式が保証されないと理解している（未確認）。

学習だけに効くフィルタ（`train_filters`。2026-11-26 ユーザー判断）:

- `filters` とは別のリストに書く。形と演算子は `filters` と同じ。例: 1 年分のデータのうち 9 月までで学習し、1 年分すべてを予測する。
- 学習時は、`filters` と `train_filters` の両方を満たす行だけを使う。`train_filters` で外れた行は、補完・標準化の統計量にも使わない。学習期間の外の説明変数の分布を、学習に入れないため。
- 予測時は `train_filters` を使わない。外れた行も予測し、`passed_filter` は `filters` だけで決まる。学習に使っていないことは、`used_for_training` が false になることでわかる。
- `run.json` と `model.pkl` の両方に残す。`model.pkl` の `train_filters` は、予測では行を絞らず、`used_for_training` の判定にだけ使う（「予測」）。
- 学習期間の外の行でのスコアは、`mloom optimize` では計算しない（2026-11-26 ユーザー判断）。HPO と特徴量選択は学習期間の中の CV だけで決める。学習期間の外のスコアを `run.json` に出すと、それを見ながら設定を変えて実行を繰り返し、学習期間の外のデータでチューニングしたのと同じことになりやすいため。学習期間の外での評価は、`mloom predict` の役目にする（「予測」）。

形式。CLI の `--filters` に渡す JSON ファイルの中身（どちらのキーも省略できる。ほかのキーはエラー）:

```json
{
  "filters": [
    {"column": "FI-101", "op": "gt", "value": 0},
    {"column": "grade_code", "op": "is_in", "value": [1, 2]},
    {"column": "TI-120", "op": "is_between", "value": [80, 120]},
    {"column": "PI-300", "op": "is_not_null"},
    {"column": "operation_mode", "op": "str_contains_any", "value": ["定常", "steady"]}
  ],
  "train_filters": [
    {"column": "timestamp", "op": "lt", "value": "2026-10-01T00:00:00"}
  ]
}
```

| `op` | 意味 | `value` |
| --- | --- | --- |
| `eq`、`ne`、`lt`、`le`、`gt`、`ge` | 値と比べる（polars の同名のメソッド） | スカラー |
| `is_in` | 値の一覧に含まれる | 配列 |
| `is_between` | 範囲に入る（両端を含む） | `[下限, 上限]` |
| `is_null`、`is_not_null` | 欠損か、欠損でないか | なし |
| `str_contains_all` | 配列の文字列をすべて含む（2026-11-26 ユーザー判断） | 文字列の配列 |
| `str_contains_any` | 配列の文字列のどれかを含む（2026-11-26 ユーザー判断） | 文字列の配列 |

- 比較の演算子では、欠損の行は条件を満たさない（polars の `filter` と同じ）。
- `str_contains_all`・`str_contains_any` は、正規表現ではなく、文字の並びそのままで一致を見る（2026-11-26 ユーザー判断）。`.` や `(` を含むタグ名でも、思わぬ一致が起きないため。文字列の列でだけ使え、欠損の行は条件を満たさない。文字列の列は説明変数からは外れる（`non_numeric`）が、フィルタには使える。
- 条件どうしの OR（入れ子）は足さない（2026-11-26 ユーザー判断）。「どれかを含む」は `str_contains_any` で書く。
- 日時・日付の列（Datetime、Date）と比べる値は、ISO 8601 の文字列で書き、列の型に変換して比べる。
  - タイムゾーン付きの列と比べる値に、オフセットが無ければエラーで止める。タイムゾーンが無い列と比べる値に、オフセットがあってもエラーで止める。黙って変換すると、時刻のずれに気づけないため。
  - オフセット付きの値は、その時点（UTC の同じ瞬間）として、列のタイムゾーンに変換して比べる。
  - 値は、列の時間の単位（ns、us、ms）の整数に、文字列の小数秒から直接変換する。Python の `datetime.fromisoformat` はマイクロ秒より下の桁を捨てるため、そのまま使うと `Datetime("ns")` の列と正しく比べられない（クロスレビューの指摘）。列の単位で表せない値（ms の列に `00:00:00.0015` など）は、切り捨てずにエラーで止める（2026-11-26 実装で決定）。小数秒は日時の本体からだけ読む。小数秒を含む UTC オフセット（`+01:00:00.5` など）は、本体の小数秒と取り違えやすく、使う場面も無いので、エラーで止める（2026-11-26 実装で決定。クロスレビューの指摘）。
  - Date の列には日付（`YYYY-MM-DD`）だけを書ける。Datetime の列に日付だけを書くと、その日の 0 時になる（Python の `datetime.fromisoformat`）。

実装で決めたこと（2026-11-26、Issue #4）:

- 学習時は、「行の除外」の前に、まず `filters`、次に `train_filters` を適用する。目的変数が欠損した行（統計量に使う行）にも適用する。外した行数を `run.json` の `data` に残す（`filters` で外した行は `n_rows_filtered_out`、さらに `train_filters` で外した行は `n_rows_train_filtered_out`）。`n_rows` は入力の行数のまま。`n_rows_without_target` などの「行の除外」の行数は、両方のフィルタの後に残った行で数える。フィルタで外した行を、目的変数の欠損などで重ねて数えないため。
- 列がデータに無い、`op` を知らない、知らないキーがある、`value` の形が `op` に合わない、列の型と値の型が合わないときは、学習を始める前に `InputError`（終了コード 2）で止める。データが要らない検査（キー、`op`、`value` の形）は `Config` を作った時点で行う。
  - `value` の形: 比較の演算子はスカラー（数値・文字列・真偽値。null は不可。欠損は `is_null` で書く）、`is_in` は空でない配列、`is_between` は 2 要素の配列、`str_contains_*` は空でない文字列の配列。
  - 列の型と値の型: 数値の列（整数・浮動小数・Decimal）には数値（真偽値は不可）、真偽値の列には真偽値、文字列の列（String・Categorical・Enum）には文字列、日時・日付の列には ISO 8601 の文字列。ほかの型の列（Duration、Time、List など）には `is_null`・`is_not_null` だけを使える。黙って行が消えるのを防ぐため。
- 浮動小数の列の NaN は欠損として扱う。比較・`is_in`・`is_between` の条件を満たさず、`is_null` を満たす。polars の比較では NaN が最大の値として扱われ、`gt` などを満たしてしまうため（2026-11-26 polars 2.2.0 で確認）。説明変数の扱い（「列の除外」）とも揃う。
- Categorical・Enum の列は、文字列にしてから比べる（`lt` などは文字列の順）。
- `is_in` は、値ごとの `eq` のどれかを満たすこと、`is_between` は `ge` と `le` の両方を満たすことで判定する。日時の変換と欠損の扱いを、比較の演算子と同じにするため。
- `filters` か `train_filters` を適用した後に 1 行も残らなければエラーで止める（終了コード 2）。
- `time_column`・`group_column` の欠損の検査は、フィルタの後の行で行う。フィルタで外した行は使わないので、その行の欠損では止めない（`is_not_null` のフィルタで、欠損した行を外せる）。
- フィルタに使う列を、それだけの理由では説明変数から外さない。外すかどうかは、ほかの列と同じく「列の除外」で決まる。外したいときは `exclude_columns` で指定する。
- `run.json` と `model.pkl` には、渡した条件をそのまま残す（`run.json` の一番上の `filters`・`train_filters`）。

### 行の除外

「行のフィルタ」を適用した後の行を、次の順に分け、外し、それぞれの行数を `run.json` の `data` に残す。

| 順 | 扱い | `run.json` |
| --- | --- | --- |
| 1 | 目的変数が欠損（null、NaN）の行を、学習と評価から外す。`use_unlabeled_rows` が true なら、統計量用の「目的変数が欠損した行」として残し、false なら捨てる | `n_rows_without_target` |
| 2 | 学習用の行のうち、目的変数が +inf・-inf の行を外す（2026-11-26 ユーザー判断。未決 18） | `n_rows_with_infinite_target`。列ごとの無限大の行数は `infinite_by_column` |
| 結果 | 学習と評価に使う行、統計量だけに使う目的変数が欠損した行 | `n_rows_used`、`n_unlabeled_rows_used` |

- 説明変数の +inf・-inf では行を外さず、欠損とみなす（2026-11-26 ユーザー判断で「行ごと外す」から変更。2026-11-26 に実装）。予測時は行を外せない（行ごとに予測を返すため）ので、学習時だけ行を外すと、学習と予測で扱いが食い違うため。
  - Pipeline の中で、列の選択の直後（クリップより前）に NaN に変える。`FunctionTransformer(np.nan_to_num, kw_args={"nan": np.nan, "posinf": np.nan, "neginf": np.nan})` を使い、mloom の関数を pickle に入れない（2026-11-26 scikit-learn 1.9.1 で確認）。学習時も予測時も同じ処理を通る。
  - クリップより前に置くのは、無限大がクリップで上下限まで切られて、もっともらしい値として扱われるのを防ぐため。
  - 列の除外（`all_null`・`constant`・`high_missing`）でも、無限大を欠損として数える（「列の除外」）。
- `infinite_by_column` は、目的変数と「説明変数になりうる列」の、無限大の行数。学習用の行と残した目的変数が欠損した行で、目的変数が無限大の行を外す前に数える。無限大が出ていた列を後から確かめられるようにするため。
  - 「説明変数になりうる列」は、列名で除外される列（目的変数、`time_column`・`group_column`、`exclude_columns`、使えないセンサ）を除いた浮動小数の列。列名で除外される列の無限大は数えない。
  - 無限大が多い列は `infinite_by_column` で見つけられる。欠損率のしきい値で除外されなければ、`exclude_columns` か `unavailable-sensors.json` で外す。
- 1 行も残らなければエラーで止める（終了コード 2）。
- たたき台の実装（2026-11-26 まで）は、目的変数か説明変数になりうる列に無限大がある行を、目的変数が欠損した行も含めて外していた（`n_rows_with_infinite`）。予測時は入力に無限大があると scikit-learn の `ValueError` になっていた。

### 列の除外

特徴量選択の前に、目的変数を使わずに決められる列を外す。外した列は理由とともに `run.json` の `excluded_columns` に残す。

| 理由 | 対象 |
| --- | --- |
| `target` | 目的変数 |
| `role` | `time_column`、`group_column` |
| `user_excluded` | `exclude_columns` で指定した列 |
| `unavailable_sensor` | 使えないセンサ |
| `non_numeric` | 数値でも真偽値でもない列（案）。真偽値は 0/1 の数値として扱う |
| `all_null` | すべて欠損の列（すべて無限大の列も含む） |
| `constant` | 欠損を除いて値が 1 種類の列 |
| `high_missing` | 「目的変数がある行での欠損率」か「目的変数が欠損した行も含めた全行での欠損率」のどちらかが、設定ファイルの `max_missing_rate`（既定 0.5）を超える列。`detail` に両方の欠損率を残す |

- 浮動小数の列の NaN と +inf・-inf は欠損として扱う（無限大は 2026-11-26 に実装。Pipeline の中で NaN になるため。「行の除外」）。
- 欠損率は「行の除外」の後の行で計算する。`use_unlabeled_rows` が false なら、2 つの欠損率は同じになる。
- `all_null` と `constant` は、目的変数がある行だけで判定する（モデルが学ぶのはこの行だけのため）。
- 列の除外は CV の前に 1 回だけ行う。時系列の CV でも、欠損率は評価期間を含む全期間で計算する（目的変数を使わないので、ラベルの情報は漏れない）。
- `high_missing` に当たらない欠損は、Pipeline の中で `[impute] method` の方式で補完するか、補完せずにモデルに渡す（「前処理」）。
- `time_column`・`group_column` に欠損があれば、分割できないのでエラーで止める（終了コード 2）。

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
- 分割に使うのは目的変数がある行だけ。目的変数が欠損した行は評価側に入らない。

分割ごとに、補完・標準化の統計量に加える「目的変数が欠損した行」（2026-11-26 ユーザー判断。`run.json` の `cv.folds[].n_unlabeled`）:

| `cv` | 加える行 | 理由 |
| --- | --- | --- |
| `kfold` | すべて | 評価側に当たる行が無い |
| `timeseries` | 時刻が、学習側の最後の時刻以前の行 | 評価期間以降の説明変数の分布を見ないため |
| `group` | 評価側のグループに属さない行 | 評価側のロットの説明変数の分布を見ないため |

### 前処理

決定事項（2026-11-26 ユーザー判断。2026-11-26 に実装、`mloom.pipeline`・`mloom.clip`）:

- Pipeline の前処理の順は、次のとおり。
  1. 列の選択
  2. 無限大を NaN に変える
  3. クリップ
  4. 欠損の補完
  5. 標準化（モデルによる）
  6. 採用した列の選択（「特徴量選択」）。この後に推定器が続く
- どの部品も scikit-learn・numpy にあるものだけで組み、mloom のクラスや関数、lambda を入れない（「`model.pkl`」）。
- クリップの上下限と補完の統計量は、統計量用の行（「処理の流れ」）で先に計算し、固定する。
- 2〜5 は 1 つの Pipeline（保存する Pipeline の `preprocess` の段。`FrozenEstimator` で固定）にまとめる。段の名前は `infinite`・`clip`・`impute`・`scale` で、使わない段は入れない。

#### クリップ

- すべての説明変数に、設定ファイルの `[clip] method` で選んだ 1 つの方式を使う。既定は `none`（切らない）。

| `method` | 下限 〜 上限 |
| --- | --- |
| `none` | 切らない |
| `quantile` | `lower_quantile` 分位点 〜 `upper_quantile` 分位点（既定 1% 点 〜 99% 点） |
| `iqr` | Q1 − k·IQR 〜 Q3 + k·IQR（k は `iqr_k`。既定 1.5。3.0 などにもできる） |
| `mad` | 中央値 ± k × 1.4826 × MAD（k は `mad_k`。既定 3.0） |
| `sigma` | 平均 ± k·σ（k は `sigma_k`。既定 3.0） |

- 上下限の計算には欠損と無限大を使わない（無限大は先に NaN にしてから `np.nanquantile` などで計算する）。欠損はクリップの後も欠損のまま残る。
- σ は母標準偏差（`ddof=0`。`np.nanstd` の既定）。
- `FunctionTransformer(np.clip, kw_args={"a_min": 下限の配列, "a_max": 上限の配列})` で実装し、mloom の関数を pickle に入れない（2026-11-26 scikit-learn 1.9.1 で確認）。
- 上下限が 1 点に潰れる列はクリップしない。下限を `-inf`、上限を `+inf` にする。`iqr` で IQR が 0、`mad` で MAD が 0、`sigma` で σ が 0 の列（ほとんど一定で、ときどきだけ動くセンサなど）がこれに当たる。そのまま切ると、値がすべて同じになるため。
  - `quantile` で 2 つの分位点が同じ値になる列も、同じ理由でクリップしない（2026-11-26 実装で決定）。
  - 統計量用の行に有限の値が 1 つも無い列（CV の分割の学習側でだけすべて欠損の列など）は、上下限が求まらないので、同じくクリップしない（2026-11-26 実装で決定）。
  - 「1 点に潰れる」は、下限と上限が浮動小数の丸めの範囲（相対 1e-12）で等しいことで判定する。σ は、値がすべて同じでも丸めで 0 にならないことがあるため。
- `run.json` の `preprocessing.clip` に、最終モデルの値で次の 3 つを残す。CV の分割ごとの値は書かない（`run.json` が大きくなりすぎるため）。
  - `method`、`bounds`（クリップした列ごとの `[下限, 上限]`）、`unclipped_columns`（上の理由でクリップしなかった列）。`method` が `none` なら `bounds` と `unclipped_columns` は空。

```json
{"method": "iqr", "bounds": {"TI-102": [-3.1, 3.2]}, "unclipped_columns": ["PI-900"]}
```

#### 欠損の補完

- 設定ファイルの `[impute] method` で方式を選ぶ。

| `method` | 補完 |
| --- | --- |
| `auto`（既定） | モデルごとの既定（下の表） |
| `none` | 補完しない。欠損をそのまま扱えるモデルでだけ選べる。ほかのモデルで選ぶとエラーで止める（終了コード 2） |
| `median`、`mean`、`most_frequent` | `SimpleImputer` のその方式 |
| `constant` | `SimpleImputer(strategy="constant", fill_value=fill_value)` |
| `knn` | `KNNImputer(n_neighbors=n_neighbors)` |
| `iterative` | `IterativeImputer(random_state=seed)` |

- どの部品にも `keep_empty_features=True` を付ける。列の数を変えず、採用した列の番号がずれないようにするため。統計量用の行ですべて欠損の列は 0 で埋まる（`constant` は `fill_value`）。
- `none` を欠損を扱えないモデルで選んだときは、データを読んで設定を確かめた直後、学習を始める前に止める（`mloom.pipeline.plan_preprocessing`）。
- 欠損を扱えるかどうかと、`auto` の方式は、`mloom.models.ModelSpec` の `handles_missing`・`default_impute` に持つ。

| `model` | `auto` の補完 | 理由 |
| --- | --- | --- |
| `ridge`、`pls` | `median` | 欠損を扱えない |
| `hist_gradient_boosting`、`random_forest` | `none` | 分割のたびに欠損の行を送る側を学習するので、欠損であること自体を情報として使える。中央値で埋めると、欠損の行が「たまたま中央値だった行」と区別できなくなる |

- `RandomForestRegressor`・`RandomForestClassifier` は、欠損を含む X で fit・predict できる（2026-11-26 scikit-learn 1.9.1 で確認。2000 行・8 列、20% を NaN にしたデータで、例外なく予測値が返った）。そのため `random_forest` の `auto` は `none` のまま。
- 学習時に欠損が無かった列に、予測時だけ欠損が来たとき、`HistGradientBoosting` は、その分割で学習の行が多かったほうの子に送る（2026-11-26 scikit-learn 1.9.1 で確認。欠損の無いデータで 1 回だけ分割する木を作ると、根の `missing_go_to_left` が 0 で、右の子の行数 1050 が左の 950 より多く、その列を NaN にした行の予測が右の子の値と一致した）。mloom の処理は変えない。
- `IterativeImputer` は、作るときは `sklearn.experimental.enable_iterative_imputer` の import が要るが、pickle を読む側では要らない（2026-11-26 scikit-learn 1.9.1 で確認）。

### 特徴量選択

- 各列を目的変数との関係でランク付けし（回帰は `f_regression`、分類は `f_classif`）、上位 `k` 列を使う。
- `k` は HPO で探索するハイパーパラメータの 1 つにする（1 〜 候補列数、対数スケール）。
- 最終学習で選ばれた列を「採用した列」として `run.json` の `selected_columns` に残す（候補列の順）。`best_params.k` と数が一致する。
- ランク付けの方法（`mutual_info_regression`、モデルの重要度、相関の高い列の間引きなど）は「未決事項」。

ランク付けを Pipeline の外で行う（2026-11-26 決定。2026-11-26 に実装、`mloom.selection`）:

- たたき台の実装は、Pipeline の中の `SelectKBest(f_regression / f_classif)` で選んでいた。`SelectKBest` は NaN を受け付けず、補完しないモデルの前に置くと、欠損を含むデータで fit が `ValueError: Input X contains NaN` になる（2026-11-26 scikit-learn 1.9.1 で確認。クロスレビューの指摘）。補完しない既定（「前処理」）と両立させるため、Pipeline から外した。
- CV の分割ごとに 1 回だけ、学習側の目的変数がある行でランク付けし、試行をまたいで使い回す。最終モデルでは、学習に使う行のすべてで 1 回行う。目的変数を使うランク付けは学習側の行だけで行うので、評価側の情報は漏れない。
- 使う値は、前処理（2〜5）の出力。出力に欠損が残る場合（補完しない場合）は、その出力を、統計量用の行の前処理の出力の中央値で補完した写しを使う。この写しはランク付けにだけ使い、推定器には補完していない値（欠損を含む）を渡す。
- NaN のスコア（一定の列など）は、最も低い順位にする。同じスコアの列は、候補列の順で前の列を上にする（結果を決定的にするため）。
- 試行の `k` に応じて、その分割の上位 `k` 列の番号（候補列の順に並べ直したもの）を、Pipeline の `select` の段の `ColumnTransformer([("keep", "passthrough", 採用した列の番号)], remainder="drop")` に渡す。
- `KNNImputer` や `IterativeImputer` は、ほかの列の値から補完する。そのため、前処理は候補列のすべてで行い、採用した列はその後で選ぶ。
- 2026-11-26 に scikit-learn 1.9.1 で確かめたこと: 列の番号で選ぶ構成は、欠損と無限大を含む polars の表で学習・予測でき、pickle に `mloom` という文字列が含まれなかった（`tests/test_predict.py`。クリップ `iqr` と補完 `knn`・`iterative` でも同じ）。

### HPO（案）

- Optuna の `TPESampler(seed=seed)`、試行数 `n_trials`、打ち切り時間 `timeout`。
- 目的関数は CV の平均スコア（指標の本来の値。RMSE なら正の値）。`rmse`、`mae`、`log_loss` は最小化、`r2`、`accuracy`、`f1_macro`、`roc_auc` は最大化。
- CV のどれかの分割で学習や評価に失敗した試行は、警告を標準エラーに出し、失敗として記録して比べない（`run.json` の `hpo.n_trials_failed`）。すべての試行が失敗したら `OptimizationError`（終了コード 1）。
- 進捗は Optuna のログとして標準エラーに出る。
- 探索空間はモデルごとに mloom 側で固定する（下の「モデル」）。Skill やユーザーには探索空間を書かせない。

### モデル（案）

モデル名と `task` から、推定器・前処理・探索空間を引くレジストリを mloom の中に持つ。

| `model` | 回帰の推定器 | 分類の推定器 | 前処理 | 探索するパラメータ（案） |
| --- | --- | --- | --- | --- |
| `ridge` | `Ridge` | `LogisticRegression`（L2） | 補完（既定は中央値）、標準化 | `alpha`: 1e-4 〜 1e3（対数）。分類では `C = 1 / alpha` |
| `pls` | `PLSRegression`（`scale=False`） | 未対応（指定するとエラー。未決 1） | 補完（既定は中央値）、標準化 | `n_components`: 1 〜 min(20, 候補列数)。`k` を超える分は `k` に切り詰める |
| `random_forest` | `RandomForestRegressor` | `RandomForestClassifier` | 既定は補完しない（「前処理」） | `n_estimators`: 50 〜 300、`max_depth`: 2 〜 32、`min_samples_leaf`: 1 〜 20、`max_features`: 0.1 〜 1.0 |
| `hist_gradient_boosting`（2026-11-26 に実装。`lightgbm` を置き換えた） | `HistGradientBoostingRegressor` | `HistGradientBoostingClassifier` | 既定は補完しない（「前処理」） | `learning_rate`: 1e-3 〜 0.3（対数）、`max_iter`: 50 〜 1000（整数、対数）、`max_leaf_nodes`: 8 〜 256（整数、対数）、`min_samples_leaf`: 5 〜 100（整数、対数）、`l2_regularization`: 1e-8 〜 10（対数）、`max_features`: 0.5 〜 1.0（`lightgbm` の範囲を対応する引数に移した。行のサブサンプリングは無い） |

- 探索範囲の正は `mloom.models`。上の表は 2026-11-26 のたたき台の実装の値。
- 再現性のため、モデルの `n_jobs` は 1 に固定し、`random_state` に `seed` を渡す。
  - `HistGradientBoosting` には `n_jobs` が無く、OpenMP で全コアを使う。スレッド数は結果を変えないので、固定しない（2026-11-26 scikit-learn 1.9.1 で確認。同じ seed で `threadpoolctl.threadpool_limits` を 1 と 4 にして、2000 行・8 列・20% 欠損のデータで回帰・分類の予測値（分類は確率）が完全に一致した。20000 行・30 列で 1・4・12 スレッドでも一致した）。
  - `hist_gradient_boosting` は `early_stopping=False`、`random_state=seed` に固定する。

- Pipeline の構成: 入力列の選択 → 前処理（無限大を NaN に → クリップ → 補完 →（標準化）。fit 済みで固定）→ 採用した列の選択 → 推定器。前処理の fit に使う行は「処理の流れ」、各段は「前処理」「特徴量選択」。
- 分類の `ridge` に `RidgeClassifier` ではなく `LogisticRegression` を当てるのは、`log_loss` と `roc_auc` に要る確率（`predict_proba`）を出せるため（案）。
- たたき台の実装にあった `lightgbm`（追加の依存 `mloom[lightgbm]`）は、`hist_gradient_boosting` に置き換えてなくした（2026-11-26 に実装）。`pyproject.toml` の追加の依存と dev の依存、`run.json` の `versions.lightgbm` も消した。
- `hist_gradient_boosting` の `early_stopping` は `False` に固定する（2026-11-26 ユーザー判断。2026-11-26 に実装）。既定の `"auto"` は、1 万行を超えると学習データの 10% を内部の検証に取ってしまい、CV と最終学習で使う行が変わるため。
- `task` に対応しないモデルを指定したら、学習を始める前にエラーで止める（終了コード 2）。

## 出力

`output_dir` に次を保存する。

| ファイル | 内容 |
| --- | --- |
| `model.pkl` | 学習済みの Pipeline 全体（列の選択・前処理・特徴量選択・モデル）、行のフィルタの定義、学習に使った行の照合用のハッシュ。予測時も同じ処理を通す |
| `run.json` | 採用した列、除外した列と理由、最良のパラメータ、CV の分割、スコア、seed、ライブラリの版など |

### `model.pkl`

- 標準ライブラリの `pickle` で保存する。
- pickle は保存したときと同じ版の scikit-learn などで読む前提。版は `run.json` に残す。
- 信頼できない pickle を読まない（pickle は読むときに任意のコードを実行できる）。README に書く。

mloom が無い環境でも読める（2026-11-26 ユーザー判断。未決 16。2026-11-26 に実装）:

- たたき台の実装では、Pipeline の先頭が mloom の `ColumnSelector` だったので、pickle を読む環境にも mloom が要った。
- `ColumnSelector` をやめ、scikit-learn の `ColumnTransformer([("keep", "passthrough", 候補列)], remainder="drop")` で列を選ぶ。その後ろは、前処理・採用した列の選択・推定器（「前処理」「特徴量選択」。たたき台の実装では前処理・`SelectKBest`・推定器だった）。
- 読むときは、`pickle.load` の結果の `pipeline` の `predict` に、候補列（`run.json` の `candidate_columns`）を含む表（polars または pandas の DataFrame）を渡す。
- 2026-11-26 に scikit-learn 1.9.1・polars 2.2.0 で確かめたこと:
  - polars と pandas の表のどちらでも予測できる。
  - 列の順を入れ替えても、余分な列があっても、同じ予測になる（`tests/test_optimize.py`）。
  - 候補列が無い表では、`ValueError: columns are missing` になる。
  - null がある整数の列と bool の列は、float64 の NaN になって欠損として扱われる（補完するモデルでも、補完しないモデルでも予測できることを、2026-11-26 に `ridge` と `hist_gradient_boosting` で確かめた）。
  - pickle に `mloom` という文字列が含まれず、`sys.modules["mloom"] = None` で mloom の import を塞いだ Python でも読んで予測できる（`tests/test_predict.py`）。
- 予測する環境には、学習時と同じ版の scikit-learn・numpy・scipy と、入力に使う polars か pandas が要る。版は `run.json` の `versions` を見て揃える。
- pickle の中身は Pipeline ではなく、次の辞書にする。行のフィルタは scikit-learn の Pipeline に入れられないため（部品は行の数を変えられない。学習時は目的変数とずれ、予測時は入力の行と予測が対応しなくなる）。辞書と Pipeline には、mloom のクラス・関数・lambda を入れない。

```python
{
    "pipeline": Pipeline,   # 列の選択 → 前処理 → 特徴量選択 → 推定器
    "filters": [...],       # 「行のフィルタ」の定義。run.json の filters と同じ
    "train_filters": [...], # 学習だけに効くフィルタ。予測では used_for_training の判定にだけ使う
    "training_rows": {      # 学習に使った行の照合用（「予測」）
        "columns": [...],       # ハッシュにする列（候補列 → 目的変数 → time_column → group_column の順）
        "target": "quality",    # 目的変数の列名
        "time_column": "timestamp",  # 指定しなければ None
        "group_column": None,        # 指定しなければ None
        "hashes": numpy.ndarray,     # 最終モデルの学習に使った行のハッシュ（uint64、昇順で重複なし）
        "last_time": datetime(...),  # 最終モデルの学習に使った行の time_column の最大値。指定しなければ None
    },
    "shap_background": numpy.ndarray,  # 線形の SHAP の基準（「SHAP の寄与」）
}
```

- `training_rows` に `target`・`time_column`・`group_column` の名前を入れる（2026-11-26 実装で決定）。予測では目的変数の列があるかで `used_for_training` を null にするかを決め、スコアでは `time_column` が要るため。`columns` の位置からは、`time_column` と `group_column` の片方だけを指定した実行で、どちらなのかを区別できない。
- `last_time`（2026-11-26 に Issue #6 で実装、`mloom.training_period`）は、最終モデルの学習に使った行の `time_column` の最大値。polars から取り出した Python の値（`datetime`・`date`・`int`・`float`・`decimal.Decimal`・`bool`・`str`）なので、pickle の依存は増えない。`time_column` を指定していない実行では `None`。スコアの対象の行（「スコア」）を決めるのに使う。
  - `time_column` の型は限らない（行のハッシュが扱える数値・真偽値・文字列・日時・日付なら学習できる）。値は、`timeseries` の CV が行を並べる順（polars の順）で比べる。文字列（`Categorical`・`Enum` は文字列にしてから）は文字の順で比べる（2026-11-26 実装で決定）。ISO 8601 の文字列の時刻で学習する使い方を止めないため。また、CV の並びと「学習期間の後」がずれないようにするため。
  - 日時は、マイクロ秒の単位に変換してから最大値をとる。予測時も、入力の列をマイクロ秒に変換してから比べる（2026-11-26 実装で決定）。Python の `datetime` はマイクロ秒までしか持てない。ns の列の最大値をそのまま取り出すと、マイクロ秒未満が落ちて、最後の学習行が `last_time` より「後」になってしまう。両側に同じ変換をかけるので、学習に使った行が後になることはない。代わりに、最後の学習行と同じマイクロ秒の中の行は、学習に使っていなくても対象から外れる（行のハッシュもマイクロ秒で比べる）。
  - タイムゾーン付きの列では、`last_time` を UTC の `datetime`（`datetime.UTC`）にする。タイムゾーンのデータベースに依らずに読めるようにするため。
  - 予測時に、入力の `time_column` の値の種類（数値・真偽値・文字列・日時・日付）が `last_time` と違うとき、または日時のタイムゾーンの有無が違うときは、`InputError`（終了コード 2）で止める。整数・浮動小数・Decimal の列のように、種類が同じなら比べる（Decimal はクロスレビューの指摘で 2026-11-26 に対応）。
  - `time_column` が null か NaN の行は、学習期間の後の行にしない。
- `shap_background`（2026-11-26 に Issue #6 で実装）は、最終モデルの学習に使った行の、推定器への入力（前処理と、採用した列の選択の後の値）の列ごとの平均（numpy の配列）。欠損は除いて平均する（欠損を扱えるモデルは補完しないことがあるため。基準は線形のモデルにだけ使い、線形のモデルは必ず補完する）。予測時には学習データが無いので、線形の SHAP の基準を `model.pkl` に入れる。
- `Result.pipeline` は Pipeline を返す。`Result.model_bundle()` が上の辞書を作り、`Result.save` がそれを `model.pkl` に保存する。
- mloom を使わずに予測する手順（README に書いた）: `pickle.load` → `filters` の条件を AND で評価 → 条件を満たす行だけ `pipeline.predict` → 満たさない行は NaN。条件を満たす行が 0 行なら `predict` を呼ばない（scikit-learn は 0 行の表でエラーになるため）。

### 予測

決定事項（2026-11-26 ユーザー判断。未決 10）: `mloom predict` の CLI と、同じ処理の Python API を用意する（2026-11-26 に実装、`mloom.prediction`。スコア・混同行列・SHAP は同日に Issue #6 で実装）。

- 処理: `model.pkl` を読み、フィルタを適用し、条件を満たす行だけを予測し、満たさない行は NaN にして、入力と同じ行数・同じ順で返す。
- 説明変数の無限大は Pipeline の中で欠損になるので、予測の入口では特別に扱わない（「行の除外」。2026-11-26 に実装。その行にも予測値が返る）。
- この入口を使う環境には mloom が要る。`model.pkl` 自体は mloom なしでも読める。

出力の列（2026-11-26 ユーザー判断。未決 20）。列の順は、`--keep-columns` の列 → `prediction` → `passed_filter` → `used_for_training` → `proba_<クラス>`（`pipeline.classes_` の順、クラス名は `str()`）。入力を確かめる列が先にあると読みやすいため。その後ろに、`error`・`rmse`・`r2`（回帰のとき）→ `cm`・`accuracy`・`precision`・`recall`（閾値を与えたとき）→ `shap_base`（多値分類では `shap_base_<クラス>`）と `shap_*`（`--shap` を付けたとき）の順で続ける（2026-11-26 Issue #6 で決定）。

| 列 | 内容 | 出す条件 |
| --- | --- | --- |
| `prediction` | 予測値。フィルタの条件を満たさない行は NaN（分類では null。型はクラスの型） | 常に |
| `--keep-columns` で指定した列 | 入力の列をそのまま出す（例: `timestamp`、`lot`）。予測と元の行を結び付けるため | 指定したとき |
| `passed_filter` | フィルタの条件を満たしたか（bool）。予測値が NaN の理由が、フィルタの外だからかどうかを見分けるため | 常に |
| `used_for_training` | 最終モデルの学習に使った行か（bool）。予測する表に目的変数の列が無ければ null | 常に |
| `proba_<クラス>` | 分類のクラスごとの確率（`predict_proba`）。フィルタの条件を満たさない行は NaN | 分類のとき |
| `shap_<列名>`、`shap_base` | 採用した列（`selected_columns`）ごとの SHAP 寄与と、基準値（「SHAP の寄与」） | `--shap` を付けたとき |
| `error` | 予測 − 実測（「スコア」） | 回帰のとき |
| `rmse`、`r2` | スコアの対象の行全体での RMSE と R²。全行に同じ値（「スコア」） | 回帰のとき |
| `cm` | 混同行列のどこに入るか（`tp`・`fp`・`fn`・`tn`）（「混同行列」） | 閾値（分類では陽性のクラス）を与えたとき |
| `accuracy`、`precision`、`recall` | スコアの対象の行全体の `cm` の数から計算した値。全行に同じ値（「混同行列」） | 閾値（分類では陽性のクラス）を与えたとき |

- 行数と行の順は、入力と同じ。
- `prediction`・`proba_*` は、`passed_filter` が true の行だけで計算する。true の行が 0 行なら `predict` を呼ばず、すべての行を NaN（分類の `prediction` は null）にする（2026-11-26 実装で決定）。
- 次のときは `InputError`（終了コード 2）で止める（2026-11-26 実装で決定）:
  - `model.pkl` が無い、または上の辞書でない。
  - 入力に候補列か `filters` の列が無い。`filters` の値が列の型に合わない（学習時と同じ検査）。`passed_filter` が決まらないため。
  - 入力に目的変数があるのに、ハッシュにする列（`time_column`・`group_column`）が無い。
  - `--keep-columns` の列が入力に無い、出力の列と同じ名前（上書きを防ぐため）、または 2 回以上書かれている。出力の列は、その呼び出しで出す列すべて（`error`・`rmse`・`r2`・`cm`・`accuracy`・`precision`・`recall`・`shap_*` を含む。2026-11-26 Issue #6 で決定）。
  - 混同行列の引数がモデルに合わない（「混同行列」）。
  - 出力に同じ名前の列が 2 つできる。たとえば、`--shap` を付けたとき、採用した列に `base` があると、その寄与の `shap_base` が基準値の `shap_base` と重なる。多値分類では、クラス名と列名の組み合わせで `shap_<クラス>_<列名>` が重なることもある（2026-11-26 クロスレビューの指摘で決定）。列の名前は spec で決めたとおりに保ち、重なるときだけ、列かクラスの名前を変えて学習し直すよう伝えて止める。名前を自動で変えると、同じ名前の列が、モデルによって違う意味になるため。`--shap` なしなら予測できる。
  - スコアか混同行列を出すとき（回帰のとき、または分類で `positive_class` を与えたとき）に、入力に目的変数があり、`last_time` があるのに、`time_column` の値が `last_time` と比べられない（「`model.pkl`」）。`time_column` の列が無いときは、上のハッシュにする列の検査で止まる。
  - 回帰のモデルで、スコアを計算するとき（入力に目的変数の列があり、`last_time` があるとき）に、入力の目的変数が数値でない。

学習に使った行の判定（`used_for_training`。2026-11-26 ユーザー判断）:

- 対象は、最終モデルの学習に使った行だけ。`filters` と `train_filters` を通り、目的変数があり、目的変数が無限大でない行。補完・標準化の統計量にだけ使った行（目的変数が欠損した行）と、`train_filters` で外した行は false。
- 予測する表とは、行の値で照合する。学習時に、最終モデルの学習に使った行の値を行ごとにハッシュにして、`model.pkl` の `training_rows` に保存する。ハッシュにする値は、候補列・目的変数、それに `time_column`・`group_column` の指定があればその値。
- 予測時は、同じ列で同じハッシュを作る。`filters` の条件を満たし（`passed_filter`）、`train_filters` の条件も満たし、かつハッシュが一致する行を true にする。
  - ユーザーが照合用の列を指定しなくて済む。値が後から直された行は、学習に使っていないと判定される。
  - 目的変数をハッシュに入れるのは、説明変数が同じで目的変数の有無だけが違う行を区別するため。入れないと、統計量にだけ使った行も true になる（クロスレビューの指摘）。
  - `filters`・`train_filters` との AND をとるのは、フィルタに使った列が候補列に無いとき（文字列の列など）に、条件の外の行が、ハッシュにする値の同じ学習行と一致してしまうのを防ぐため（クロスレビューの指摘）。
  - `train_filters` に使う列が予測する表に無いとき、またはその列の型が値に合わないとき（学習時と型が変わったとき）は、`used_for_training` を null（判定できない）にする。`train_filters` は予測の行を絞らないので、エラーにはしない（型が合わないときの扱いは 2026-11-26 実装で決定）。
  - 予測する表に目的変数の列が無いときは、`used_for_training` を null（判定できない）にする。新しいデータの予測では、目的変数の列が無いのがふつうなので、エラーにはしない。
  - 目的変数以外の照合に使う列が予測する表に無いときは、エラーで止める（終了コード 2）。
  - ハッシュは、polars の `hash_rows` を使わずに作る。`hash_rows` は polars の版をまたいで同じ値になることが保証されず、mloom の版を上げると古い `model.pkl` と照合できなくなるため。
  - 対象は、最終モデルの学習に使った行（「行の除外」の後の目的変数がある行を、`order_rows` で並べた行）だけ。同じ値の行が複数あっても、ハッシュは 1 つにまとめる（一致するかどうかだけを見る）。64 bit なので、衝突で別の行が true になる確率はあるが、極めて小さいので受け入れる。

行のハッシュの作り方（2026-11-26 実装で決定、`mloom.row_hash`）:

- 値を次の正規形のバイト列にし、列の順に並べて、行ごとに `hashlib.blake2b(digest_size=8)` で 64 bit（uint64、リトルエンディアン）にする。値ごとに、型を表す 1 バイト、本体、区切りの 1 バイト（`0x1f`）を置く。本体は固定長か、長さを前に付けるので、列の区切りがずれない。
  - 欠損（null、浮動小数の NaN、型が `Null` の列の値）: `N` と 8 バイトの 0。型によらず同じ印にする。
  - 数値（整数・浮動小数・Decimal）・真偽値: `F` と、Float64 にした値の IEEE 754 のビット列（リトルエンディアン）。`-0.0` は `0.0` にそろえる。+inf・-inf はそのまま。整数の列と浮動小数の列で、同じ値は同じハッシュになる（学習と予測で列の型が変わっても照合できるように）。
  - 文字列（String・Categorical・Enum）: `S` と、UTF-8 のバイト列の長さ（uint64、リトルエンディアン）と、バイト列。Categorical・Enum は文字列にしてから作る。分類の目的変数やグループの列が Categorical のことがあるため。
  - Datetime: `T` と、マイクロ秒の整数（int64、リトルエンディアン）。タイムゾーン付きの列は UTC の値、タイムゾーンが無い列はその値をそのまま整数にする。時間の単位（ns、us、ms）が違っても同じになる。
  - Date: `D` と、1970-01-01 からの日数（int64、リトルエンディアン）。
- ほかの型（Duration、Time、List、Binary など）の列はハッシュにできない。目的変数・`time_column`・`group_column` がそうなら、学習を始める前に `InputError`（終了コード 2）で止める（候補列は数値か真偽値だけなので当たらない）。
- 固定した小さな表のハッシュの値を、テスト（`tests/test_row_hash.py`）で固定している。正規形を変えると、それまでに保存した `model.pkl` と照合できなくなる。

SHAP の寄与（2026-11-26 ユーザー判断。2026-11-26 に Issue #6 で実装、`mloom.explain`）:

- `--shap` を付けたときだけ出す。計算に時間がかかることがあり、列も説明変数の数だけ増えるため。
- shap パッケージを、mloom の通常の依存にする。使うのは `mloom predict` の側だけで、`model.pkl` の依存は増えない。
- `hist_gradient_boosting`・`random_forest` は `shap.TreeExplainer`、`ridge`・`pls` は線形の SHAP（特徴量の独立を仮定）で計算する。どのモデルかは、推定器のクラスで決める（`HistGradientBoosting*`・`RandomForest*` なら木、ほかは線形）。
- 寄与は、推定器への入力（前処理と、採用した列の選択の後の値）に対して計算し、採用した列の名前で出す。標準化するモデルでは、標準化した値の単位の寄与になる（合計は同じ）。
- 線形の SHAP（2026-11-26 実装で決定）: 列 j の寄与は w_j ×（x_j − 基準_j）、基準値は基準の点での出力。基準は `model.pkl` の `shap_background`。重み w は、基準の点と、基準から列ごとに 1 だけ動かした点での、出力の差で求める。出力は入力の 1 次式なので、差は係数と一致する。推定器の係数の持ち方（`PLSRegression` は内部で入力を中心化し、`intercept_` は中心化した後の切片）に依らないため。出力は、回帰は `predict`、分類（`LogisticRegression`）は `decision_function`。
- shap は `import` を `--shap` のときの計算の中だけで行う（2026-11-26 実装。`--shap` なしでは import しないことをテストで確認）。
- 2 値分類は `classes_[1]` に対する値だけを出す（列が 2 倍に増えるのを防ぐため。もう一方のクラスは符号が逆になるだけ）。
- 2026-11-26 に shap 0.54.1・scikit-learn 1.9.1・numpy 2.5.5 で確かめたこと（欠損が 10% のデータ）:
  - `TreeExplainer` は、`HistGradientBoostingRegressor`・`HistGradientBoostingClassifier`・`RandomForestRegressor` で動いた。
  - 基準値と寄与の合計は、予測値（分類は `decision_function` の対数オッズ）と、1e-13 程度の誤差で一致した。
  - 1000 行にかかった時間は、`HistGradientBoosting` が約 1.5 秒、`RandomForest`（木 100 本、深さの制限なし）が約 52 秒。`random_forest` が遅いことは受け入れる。`--shap` を付けたときだけ計算するので、既定の予測は遅くならない。
- 2026-11-26 に shap 0.54.1・scikit-learn 1.9.1 で確かめた `TreeExplainer.shap_values` の形（行 × 列 × クラス）: `HistGradientBoostingRegressor`・`RandomForestRegressor` と 2 値の `HistGradientBoostingClassifier` は（行, 列）で `expected_value` は 1 つ。多値の `HistGradientBoostingClassifier` は（行, 列, クラス）。`RandomForestClassifier` は 2 値でも（行, 列, クラス）なので、2 値では `classes_[1]` の側を取る。
- 2026-11-26 に、4 つのモデルとタスク（回帰、2 値分類、3 値分類。`pls` は回帰だけ）の 10 の組み合わせで、行ごとの基準値と寄与の合計が、そのモデルの出力と 1e-6 以内で一致することをテストで確かめた（`tests/test_predict.py`）。
- 列は、採用した列ごとの `shap_<列名>` と、基準値の `shap_base`。採用されなかった列は、寄与が常に 0 なので出さない。列の順は、`shap_base`、`selected_columns` の順の `shap_<列名>`。
- 分類の寄与の単位は、モデルによって違う（対数オッズか確率か）。多値分類では、クラスごとに列を出す（`classes_` の順に、`shap_base_<クラス>`、`shap_<クラス>_<列名>`。クラス名は `str()`）。
- フィルタの外の行は、SHAP の列も NaN にする。

スコア（2026-11-26 ユーザー判断。2026-11-26 に Issue #6 で実装、`mloom.evaluation`）:

- 回帰では、`error`（予測 − 実測）、`rmse`、`r2` を標準で出す。`rmse` と `r2` は、スコアの対象の行全体で 1 つの値で、全行に同じ値を入れる。
  - `error` の向きを「予測 − 実測」にしたのは、ユーザーの言い方に合わせたため（統計でいう残差は逆の向き）。プラスなら予測が過大。
- スコアの対象の行は、時刻が「最終モデルの学習に使った行の最も新しい時刻」より後の行だけ。その時刻と同じか、それより前の行は、`error`・`cm` を None にし、`rmse` などの計算にも入れない。学習期間の中の行で測ると、成績が良く見えすぎるため。
  - 最も新しい時刻は、学習時に `model.pkl` に保存する（`training_rows` の中の `last_time`。「`model.pkl`」）。
  - 同じ時刻の行を含めないのは、最も新しい時刻の行は学習に使った行だから。
  - 時刻の列は `time_column`。`time_column` を指定していない実行（`kfold`・`group` の CV など）では、「学習期間の後」が決められないので、スコアと混同行列の列をすべて None にする（2026-11-26 ユーザー判断）。
- 時刻が後でも、実測（目的変数）が欠損か無限大の行、フィルタの外で予測値が NaN の行は、`error`・`cm` を None にし、計算にも入れない。
- 予測する表に目的変数の列が無いときは、スコアと混同行列の列をすべて None にする。このときは `time_column` の列が無くてもよい（`used_for_training` と同じ。新しいデータには、目的変数の列も時刻の列も無いことがあるため）。
- 対象の行が 0 行なら `rmse`・`r2` は None。1 行だけなら `r2` は None（R² は 2 行以上ないと計算できないため）。`rmse` は `sklearn.metrics.root_mean_squared_error`、`r2` は `sklearn.metrics.r2_score` で計算する。
- 分類では、`error`・`rmse`・`r2` を出さない（2026-11-26 ユーザー判断）。分類の成績は「混同行列」で見る。

混同行列（2026-11-26 ユーザー判断。2026-11-26 に Issue #6 で実装、`mloom.evaluation`）:

- 閾値を与えたときだけ、`cm`・`accuracy`・`precision`・`recall` を出す。回帰でも分類でも出す。
- 回帰: `--cm-actual-threshold 0 --cm-predicted-threshold 1 --cm-positive above` のように、実測の閾値、予測の閾値、陽性の向き（`above` か `below`）を与える。
  - `above` では閾値より大きい（>）値、`below` では閾値より小さい（<）値を陽性とする。ちょうど閾値の値は陰性。scikit-learn の `Binarizer` も「閾値より大きい」を 1 にする。
  - 実測が陽性かどうかと、予測が陽性かどうかの組み合わせで、`tp`（実測も予測も陽性）・`fp`（予測だけ陽性）・`fn`（実測だけ陽性）・`tn`（どちらも陰性）を決める。
- 分類: 陽性のクラス（`--positive-class bad` など）と、その確率の閾値（`--cm-predicted-threshold`。既定 0.5）を与える。
  - 予測は「陽性のクラスの確率が閾値より大きいか」で、実測は「実測のクラスが陽性のクラスか」で決める。
  - 多値分類では、「指定したクラスか、それ以外か」で判定する。
  - 分類では、陽性のクラスを与えたときに出す。
- 対象の行は、スコアと同じ（学習期間の後の行だけ）。それ以外の行の `cm` は None。
- `accuracy` は (tp + tn) / 全数、`precision` は tp / (tp + fp)、`recall` は tp / (tp + fn)。分母が 0 なら None。
- 引数（API は `cm_actual_threshold`・`cm_predicted_threshold`・`cm_positive`・`positive_class`）が次のときは `InputError`（終了コード 2）:
  - 回帰で、3 つのうち一部だけを与えた。`cm_positive` が `above`・`below` でない。`positive_class` を与えた。
  - 分類で、`cm_actual_threshold` か `cm_positive` を与えた。`positive_class` が学習時のクラスに無い（クラス名は `str()` で照合する）。
  - 分類で、`positive_class` なしに `cm_predicted_threshold` を与えた（2026-11-26 実装で決定。与えた閾値が黙って無視されるのを防ぐため）。
- 分類の実測が陽性かどうかも、`str(実測のクラス) == str(positive_class)` で決める。

### `run.json`

2026-11-26 の実装の形（値は例。正は `mloom.result.Result.to_dict`）。

- 一番上の `filters` と `train_filters`（渡した条件そのまま）、`data` の `n_rows_filtered_out` と `n_rows_train_filtered_out` を足した（2026-11-26、「行のフィルタ」）。

- 2026-11-26 に、次を変えた（「行の除外」「設定ファイル」「前処理」）。
  - `data.n_rows_with_infinite` を `n_rows_with_infinite_target`（目的変数の無限大で外した行数）にした。`infinite_by_column` は残す。
  - `settings` を表ごとの入れ子にし、`[clip]` と `[impute]` の値を足した。`sources` も表ごとの入れ子。
  - `preprocessing.clip` に、最終モデルのクリップの方式・上下限・クリップしなかった列を足した。
  - `versions.lightgbm` をなくした。

```json
{
  "mloom_version": "0.1.0",
  "created_at": "2026-11-26T10:00:00+09:00",
  "data": {
    "path": "data/train.parquet", "n_rows": 12500, "n_columns": 320,
    "n_rows_filtered_out": 300, "n_rows_train_filtered_out": 200, "n_rows_used": 1990, "n_unlabeled_rows_used": 9950,
    "n_rows_without_target": 10000, "n_rows_with_infinite_target": 2, "infinite_by_column": {"TI-115": 60, "quality": 2}
  },
  "target": "quality",
  "task": "regression",
  "model": "ridge",
  "metric": "rmse",
  "metric_direction": "minimize",
  "seed": 0,
  "filters": [{"column": "FI-101", "op": "gt", "value": 0}],
  "train_filters": [{"column": "timestamp", "op": "lt", "value": "2026-10-01T00:00:00"}],
  "unavailable_sensors": {
    "path": "config/unavailable-sensors.json",
    "found": true,
    "sensors": {"TI-101": "2026-10 から故障中", "FI-205": "廃止"},
    "not_in_data": ["FI-205"]
  },
  "settings": {
    "columns": {"max_missing_rate": 0.3},
    "preprocessing": {"use_unlabeled_rows": true},
    "clip": {"method": "iqr", "lower_quantile": 0.01, "upper_quantile": 0.99, "iqr_k": 1.5, "mad_k": 3.0, "sigma_k": 3.0},
    "impute": {"method": "auto", "fill_value": 0.0, "n_neighbors": 5},
    "sources": {
      "columns": {"max_missing_rate": "config/mloom.toml"},
      "preprocessing": {"use_unlabeled_rows": "C:/.../mloom/config/mloom.toml"},
      "clip": {"method": "config/mloom.toml", "lower_quantile": "C:/.../mloom/config/mloom.toml", "...": "..."},
      "impute": {"method": "C:/.../mloom/config/mloom.toml", "...": "..."}
    },
    "files": [
      {"role": "project", "path": "config/mloom.toml", "found": true},
      {"role": "local_defaults", "path": "C:/.../mloom/config/mloom.local.toml", "found": false},
      {"role": "defaults", "path": "C:/.../mloom/config/mloom.toml", "found": true}
    ]
  },
  "preprocessing": {
    "clip": {"method": "iqr", "bounds": {"TI-102": [-3.1, 3.2]}, "unclipped_columns": ["PI-900"]}
  },
  "excluded_columns": [
    {"name": "timestamp", "reason": "role", "detail": null},
    {"name": "TI-101", "reason": "unavailable_sensor", "detail": "2026-10 から故障中"},
    {"name": "FI-950", "reason": "high_missing", "detail": "labeled 0.798, all 0.412"}
  ],
  "candidate_columns": ["..."],
  "selected_columns": ["..."],
  "cv": {
    "method": "timeseries", "splitter": "TimeSeriesSplit", "n_splits": 5,
    "time_column": "timestamp", "group_column": null,
    "folds": [{"n_train": 330, "n_test": 330, "n_unlabeled": 1650}]
  },
  "hpo": {"sampler": "TPESampler", "n_trials": 100, "n_trials_completed": 100, "n_trials_failed": 0, "timeout": null, "best_trial": 57},
  "best_params": {"alpha": 0.31, "k": 42},
  "cv_scores": {"folds": [0.81, 0.79, 0.84, 0.80, 0.82], "mean": 0.812, "std": 0.017},
  "versions": {"python": "3.13.13", "mloom": "0.1.0", "scikit-learn": "1.9.1", "optuna": "5.1.0", "polars": "2.2.0", "numpy": "...", "scipy": "..."}
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

予測（2026-11-26 に実装）:

```python
predictions = mloom.predict(
    "outputs/ridge/model.pkl",     # または pickle.load で読んだ辞書
    "data/new.parquet",            # または polars.DataFrame
    keep_columns=("timestamp",),
    shap=False,                    # True で SHAP の寄与を足す
    cm_actual_threshold=None,      # 回帰の混同行列: 3 つをそろえて与える
    cm_predicted_threshold=None,   # 分類では陽性のクラスの確率の閾値（None なら 0.5）
    cm_positive=None,              # "above" か "below"
    positive_class=None,           # 分類の混同行列の陽性のクラス
)
# polars.DataFrame。入力と同じ行数・同じ順。フィルタの条件を満たさない行は NaN（「予測」）
```

- `model` に読み込んだ辞書も渡せるのは、Notebook で読み込んだものを使い回せるようにするため。
- 予測のモジュールは `mloom.prediction` にする（2026-11-26 実装で決定）。`mloom.predict` という名前のモジュールにすると、`mloom/__init__.py` が再エクスポートする関数 `mloom.predict` が、サブモジュールの属性を隠してしまうため。

- 設定は `mloom.Config`（frozen dataclass）で渡す。CLI の引数と 1 対 1 に対応させる。データが要らない検査（モデル名、`task` とモデルの組み合わせ、CV に要る列の指定など）は `Config` を作った時点で行う。
- 不正な入力（目的変数が無い、モデル名が無い、候補列が 0 列など）は、学習を始める前に `mloom.InputError` を投げる。

## CLI（案）

```powershell
uv run mloom optimize data/train.parquet --target quality --task regression --model ridge --cv timeseries --time-column timestamp --output outputs/ridge
```

- CLI は引数を `Config` に変換し、`optimize_model` と `result.save` を呼ぶだけにする。
- 標準出力の最後に `run.json` のパスを出す。進捗やログは標準エラーに出す。

フィルタ（2026-11-26 に実装。未決 19）:

```powershell
uv run mloom optimize data/train.parquet ... --filters config/filters.json
```

予測（2026-11-26 に実装。`--shap` と混同行列の引数は同日に Issue #6 で実装。出力の列は「予測」）:

```powershell
uv run mloom predict outputs/ridge/model.pkl data/new.parquet --output outputs/ridge/predictions.parquet [--keep-columns COL ...] [--shap] [--cm-actual-threshold 0 --cm-predicted-threshold 1 --cm-positive above | --positive-class bad [--cm-predicted-threshold 0.5]]
```

- CLI は Python API の予測を呼んで、結果を Parquet に保存するだけにする。`--shap`・`--cm-actual-threshold`・`--cm-predicted-threshold`・`--cm-positive`・`--positive-class` は、同じ名前の API の引数にそのまま渡す。`--keep-columns` は空白で区切って並べる（`--keep-columns timestamp lot`。`--exclude-columns` と同じ）。
- 標準出力の最後の行に、出力の Parquet のパスを出す。終了コードは `mloom optimize` と同じ。

| 終了コード | 意味 |
| --- | --- |
| 0 | 成功。`model.pkl` と `run.json`（`predict` では予測の Parquet）を保存した |
| 1 | 想定外のエラー |
| 2 | 入力の誤り（引数、データ、`unavailable-sensors.json`、`mloom.toml`、`--filters` のファイル、`model.pkl`） |

## Skill

- 本リポジトリの `skills/model-optimization/SKILL.md` に置く（2026-11-26 たたき台を作成。同日 Issue #7 で、配布と実行、`--filters`、`mloom predict`、前処理の設定に合わせて書き直した）。
- 5 つの入力のうち足りないものはユーザーに聞く。CV の方法は Skill が推測して決めず、知っている列名を材料に聞き返す。
- 役割は 2 つだけ: ユーザーの指定を CLI（`mloom optimize`・`mloom predict`）の引数に変換して実行する、結果（`run.json` と予測の出力）を読んで説明する。
- どの列を除外するか、どう評価するか、探索空間、前処理の方式、何を保存するかを Skill で決めない。それらは Python 側で固定する。スコアや指標も Skill で計算せず、mloom が出した値を伝える。

Skill が引数を作るときの決まり（2026-11-26 Issue #7 で決定）:

- フィルタ: ユーザーが行の条件（期間、運転条件、文字列を含むかどうかなど）を示したときだけ、`--filters` の JSON ファイルを作る。条件を、ユーザーが示したもの以外に足したり、推測で補ったりしない。
  - 予測でも効く条件か、学習だけの条件か（`filters` か `train_filters` か）が示されていなければ、ユーザーに聞く。予測の結果が変わるので、Skill が決めてはいけないため。
  - ファイルは `<output>/filters.json` に書き、`--output` のディレクトリが無ければ作る。モデルと一緒に残り、利用側の `config/` を Skill が書き換えずに済むため。
- 予測: ユーザーが予測を頼んだときに `mloom predict` を実行する。`--output` は、ユーザーが示さなければ `<model.pkl のディレクトリ>/predictions.parquet` にする（`mloom optimize` の `--output` を `outputs/<target>-<model>-<cv>` にするのと同じく、Skill が引数を作るための既定）。
  - 混同行列の閾値・陽性の向き・陽性のクラスは、ユーザーが示したときだけ渡す。一部だけ示されたら、残りを聞く。
  - `--shap` は、ユーザーが寄与を頼んだときだけ付ける。`--keep-columns` は、ユーザーが示した列だけ。
- 設定: 利用側の `config/mloom.toml`（`max_missing_rate`・`use_unlabeled_rows`・`[clip]`・`[impute]`）と `config/unavailable-sensors.json` を、ユーザーに頼まれない限り書き換えない。頼まれたときも、ユーザーが示したキーを、示した値にするだけ。

### 配布と実行

決定事項（2026-11-26 ユーザー判断。未決 12・13。2026-11-26 Issue #7 で `SKILL.md` と README の「導入」に実装）:

- mloom は PC に 1 つ、利用側リポジトリとは別の場所に clone する。利用側リポジトリには mloom を入れない。
- Skill は、各ベンダー（Claude Code など）の既定の Skill の置き場所に、clone の `skills/model-optimization` を指すジャンクションを置いて配る。`git pull` すれば、mloom と Skill が同じ版に揃う。
- Skill は、利用側リポジトリの直下で `uv run --project <clone> mloom optimize ...`（予測は `mloom predict ...`）を実行する。
  - `--project` は使う環境（clone の `.venv` と `uv.lock`）を決めるだけで、カレントディレクトリは利用側リポジトリのまま。利用側の `config/mloom.toml`・`config/unavailable-sensors.json`・データ・出力先は、利用側の相対パスで指定できる。
  - mloom の `uv.lock` の版で動くので、予測する環境の scikit-learn などの版を揃えやすい。
  - mloom は clone から動くので、`config/mloom.local.toml` も読まれる（「設定ファイル」）。
- `<clone>` は、Skill のディレクトリ（ジャンクション）の実際のパスから 2 つ上のディレクトリとする。ジャンクションのパスのまま `..\..` をたどると、clone ではなくジャンクションを置いた場所の親に着くので、先に実際のパスを調べる。
  - [auto] 手順（2026-11-26 Issue #7 で決定。Python を呼ばずに済むため）: PowerShell の `(Get-Item -LiteralPath <Skillのディレクトリ>).Target` の先頭の値を実際のパスとし、値が無ければ（ジャンクションでなければ）Skill のディレクトリそのもの（`FullName`）を実際のパスとする。末尾の `\` を除いてから 2 つ上をとる。
  - `<clone>` に `pyproject.toml` と `config/mloom.toml` が無ければ、実行せずに、ジャンクションの置き方の誤りとしてユーザーに伝える。
  - `.Target` の型は、Windows PowerShell 5.1 では `String[]`、PowerShell 7 では `String`（2026-11-26 確認）。`@(...)[0]` でどちらも先頭の値になる。
  - 2026-11-26 に、Windows PowerShell 5.1 と PowerShell 7 の両方で、ジャンクションと、ジャンクションでない `skills/model-optimization` のどちらからも、末尾の区切りの有無や `/` 区切りによらず、clone のルートの絶対パスが求まることを確かめた（`tmp/` の使い捨てのディレクトリ）。
- リポジトリ全体をジャンクションにする形は採らない。Claude Code は `<置き場所>/<名前>/SKILL.md` の 1 段下を探す（下の「確かめたこと」）ので、`skills/model-optimization/` の 2 段下にある `SKILL.md` が見つからないため。Skill のディレクトリのジャンクションなら、どちらのベンダーでも見つかる。

確かめたこと（2026-11-26 Issue #7）:

| ベンダー | 置き場所（ユーザー単位） | 深さ | ジャンクション | 出典 |
| --- | --- | --- | --- | --- |
| Claude Code | `~/.claude/skills/<名前>/SKILL.md`（プロジェクトでは `.claude/skills/<名前>/SKILL.md`） | 1 段（`<名前>/SKILL.md`）。ドキュメントに、それより深い場所を探す記述は無い | ドキュメントでは「`<名前>` はほかの場所のディレクトリへのシンボリックリンクでもよい」。この PC では、`~/.claude/skills/<名前>` のジャンクション（例: `pr-cross-review`）の Skill が、Claude Code のセッションの Skill の一覧に出る | https://code.claude.com/docs/en/skills 、この PC の `~/.claude/skills` |
| Codex | `~/.agents/skills`（リポジトリでは `.agents/skills`。`~/.codex/skills` はドキュメントに無い） | 置き場所の下を再帰的に探す。この PC で、`.agents/skills/nest/deep-probe-skill/SKILL.md`（2 段下）が一覧に出た | ジャンクションをたどる。`.agents/skills/model-optimization` のジャンクションの `SKILL.md` と、`skills/` 全体を指すジャンクションの下の `model-optimization/SKILL.md` が、どちらも一覧に出た | https://developers.openai.com/codex/skills （2026-11-26 時点で https://learn.chatgpt.com/docs/build-skills に転送）、この PC の codex-cli 0.172.0 の `codex debug prompt-input`（モデルを呼ばずに、モデルに渡す Skill の一覧を出す） |

- ジャンクションを作るコマンド（README の「導入」）:
  - Claude Code: `New-Item -ItemType Junction -Path "$HOME\.claude\skills\model-optimization" -Target "<clone>\skills\model-optimization"`
  - Codex: `New-Item -ItemType Junction -Path "$HOME\.agents\skills\model-optimization" -Target "<clone>\skills\model-optimization"`
- Claude Code の深さは、ドキュメントの記述と、この PC の 1 段のジャンクションで確かめた。2 段下に置いた `SKILL.md` が見つからないことは、Claude Code のセッションを別に起こさないと確かめられないので、試していない。1 段のジャンクションなら、どちらにしても見つかる。
- 2026-11-26 に、`tmp/` に偽の利用側リポジトリ（サンプルデータ、`config/mloom.toml`、`config/unavailable-sensors.json`、`<output>/filters.json`）を作り、clone に `config/mloom.local.toml` を置いて、その直下で `SKILL.md` のテンプレートどおりに `uv run --project <clone> mloom optimize ... --filters ...` と `mloom predict ...`（回帰で `--shap` と混同行列、分類で `--positive-class`）を実行した。どれも終了コード 0 で、出力は偽の利用側リポジトリの中にでき、`run.json` の `settings.files` の `project` は利用側の `config\mloom.toml`（カレントディレクトリからの相対パス）、`local_defaults`・`defaults` は clone のファイル、`unavailable_sensors.path` は利用側の `config\unavailable-sensors.json` だった。予測の出力の列は `SKILL.md` の表と一致した。

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
| Pipeline、前処理、CV、特徴量選択、モデル（勾配ブースティングも） | scikit-learn |
| HPO | optuna |
| 予測時の SHAP の寄与 | shap |
| パッケージ管理 | uv |
| lint、型チェック、テスト | ruff、ty、pytest |

## パッケージ構成

| モジュール | 責務 |
| --- | --- |
| `mloom` | 公開 API の再エクスポート（`optimize_model`、`predict`、`Config`、`Result`、例外） |
| `mloom.errors` | 例外（`InputError` は終了コード 2、それ以外は 1） |
| `mloom.config` | `Config` の定義と、データが要らない検査 |
| `mloom.metrics` | 評価指標と、タスクごとに選べる指標 |
| `mloom.models` | モデルのレジストリ（推定器、標準化の要否、欠損を扱えるかと `auto` の補完、探索空間） |
| `mloom.data` | データの読み込み |
| `mloom.sensors` | `unavailable-sensors.json` の読み込み |
| `mloom.settings` | 設定ファイル（`mloom.toml`）とその既定値の読み込み |
| `mloom.rows` | 学習に使う行の選択（目的変数の欠損と無限大）と、列ごとの無限大の行数 |
| `mloom.columns` | 列の除外と理由の記録 |
| `mloom.cv` | ユーザーが指定した CV の分割 |
| `mloom.pipeline` | Pipeline の組み立て（先頭の `ColumnTransformer` で列を選ぶ。前処理の方式の決定と fit、採用した列の選択）と、保存した Pipeline の候補列・採用した列・推定器への入力の取り出し |
| `mloom.clip` | クリップの上下限の計算と、`run.json` の `preprocessing.clip` |
| `mloom.selection` | 特徴量選択のランク付け（Pipeline の外で、分割ごとに 1 回） |
| `mloom.optimize` | `optimize_model`：HPO と最終学習 |
| `mloom.result` | `Result` と保存（`model.pkl`、`run.json`） |
| `mloom.dtypes` | 列の型を、フィルタと行のハッシュが比べる値の種類（数値、真偽値、文字列、日時、日付）として読む |
| `mloom.filters` | 行のフィルタの定義の検査と適用 |
| `mloom.row_hash` | 学習に使った行の照合用の、値による行のハッシュ |
| `mloom.prediction` | 予測の入口（`predict`）。`mloom.predict` にしない理由は「Python API」 |
| `mloom.training_period` | 学習期間の終わり（`last_time`）と、その後の行の判定 |
| `mloom.evaluation` | 学習期間の後の行での、予測のスコアと混同行列 |
| `mloom.explain` | 採用した列ごとの SHAP の寄与と、線形の SHAP の基準（`shap_background`） |
| `mloom.cli` / `mloom.__main__` | CLI |

## 未決事項

| # | 論点 | 案 |
| --- | --- | --- |
| 1 | 分類の `pls` | PLS-DA（目的変数を one-hot にして `PLSRegression`、最大の列をクラスとする）を mloom で実装するか、分類では `pls` を選べないことにするか。たたき台の実装は後者（指定するとエラー） |
| 2 | 分類の範囲 | 2 値と多値の両方。クラスの不均衡への対応（`class_weight` を探索するか）は後回し |
| 3 | CV で選べる方法 | `timeseries`、`group`、`kfold` の 3 つ。時系列かつグループ（ロットごとに時刻順）の分割が要るか |
| 4 | `timeseries` の時刻の列 | `time_column` を必須にする。行の並び順を時刻順とみなす指定も許すか |
| 5 | 特徴量選択の方法 | `f_regression`・`f_classif` のランクの上位 `k` 列（`k` を HPO で探索）。相関の高い列の間引きを足すか |
| 6 | `unavailable-sensors.json` の形式と照合 | `{"sensors": [{"name", "reason"}]}`、列名の完全一致。接頭辞やパターンでの照合が要るか |
| 7 | 数値でない列の扱い | 除外する（エンコードしない） |
| 8 | 欠損の扱い | **決定**（2026-11-26）: 補完の方式を設定ファイルで選べる。既定は、欠損を扱えるモデル（`hist_gradient_boosting`、`random_forest`）では補完せず、ほかは中央値。→「前処理」 |
| 9 | 選べるモデル | 表の 4 つで始める。API から任意の推定器と探索空間を渡せるようにするか |
| 10 | 予測の入口 | **決定**（2026-11-26）: `mloom predict` の CLI と Python API を用意する。`pickle.load` だけでも予測できるようにする。→「予測」 |
| 11 | HPO の試行の記録 | `run.json` は最良の試行だけ。全試行を `trials.csv` に出すか |
| 12 | Skill の配布方法 | **決定**（2026-11-26）: 各ベンダーの既定の置き場所に、clone の Skill を指すジャンクションを置く。→「配布と実行」 |
| 13 | 利用環境への導入 | **決定**（2026-11-26）: 利用側リポジトリとは別に clone し、`uv run --project <clone>` で実行する。→「配布と実行」 |
| 14 | 並列実行 | Optuna の `n_jobs` やモデルの `n_jobs` を使うか（再現性とのトレードオフ）。`HistGradientBoosting` の OpenMP のスレッド数は結果を変えない（2026-11-26 確認。「モデル」） |
| 15 | 複数の目的変数 | 「まずは 1 個」の次の段階で、目的変数ごとに別の Pipeline を作るか、1 つのモデルで同時に予測するか |
| 16 | pickle の読み込みに mloom が要る | **決定**（2026-11-26）: mloom への依存をなくし、`ColumnTransformer` で列を選ぶ。→「`model.pkl`」 |
| 17 | CV の方法の警告 | 日時型の列があるのに `kfold` を指定したときに、`run.json` と標準エラーに警告を出すか |
| 18 | 予測時の無限大 | **決定**（2026-11-26）: 説明変数の無限大は、学習時も予測時も欠損とみなす。→「行の除外」 |
| 19 | フィルタの渡し方 | **決定**（2026-11-26）: API は `Config.filters` と `Config.train_filters` に条件のリストを渡す。CLI では、両方を書いた JSON ファイルのパスを `--filters` で渡す。→「設定」「行のフィルタ」 |
| 20 | 予測の出力の形 | **決定**（2026-11-26）: 予測値に加えて、指定した入力の列、フィルタの判定、学習に使った行か、分類の確率、`--shap` を付けたときの SHAP の寄与を出す。→「予測」 |

## 完了条件（v0.1、案）

- Parquet と 5 つの入力（データ、`target`、`task`、`model`、`cv`）を渡すと、API と CLI のどちらからでも `model.pkl` と `run.json` が保存される。
- 回帰と分類、`cv` の 3 つの方法のそれぞれで動く。
- `config/unavailable-sensors.json` にあるセンサの列が、採用した列に入らず、`run.json` に理由つきで残る。
- `model.pkl` を `pickle.load` して、学習時と同じ列を持つ表に `predict` できる。
- 同じ入力と `seed` で 2 回実行すると、採用した列と最良のパラメータが一致する。
- 表の 4 つのモデルそれぞれで、上の 4 つがテストで確かめられている。

2026-11-26 のたたき台の実装で、上の条件はすべてテスト（`tests/`）で確かめた。ただし分類の `pls` は未対応（未決 1）。

2026-11-26 の決定で、次の条件を足す（確かめたものには、その旨を書いた。ほかは未実装）:

- mloom が無い環境で `model.pkl` を `pickle.load` し、README の手順で予測できる（2026-11-26 テストと README の手順で確認）。
- `mloom predict` で予測すると、入力と同じ行数・同じ順で予測が返り、フィルタの条件を満たさない行は NaN になる（2026-11-26 テストで確認）。
- 学習に使った表をそのまま `mloom predict` に渡すと、最終モデルの学習に使った行だけで `used_for_training` が true になる（説明変数が同じで目的変数だけが欠損した行は false になる）（2026-11-26 テストで確認）。
- 欠損を含むデータで、補完しない既定のまま `hist_gradient_boosting` と `random_forest` を学習できる（2026-11-26 テストで確認）。
- `--shap` を付けると、行ごとに `shap_base` と SHAP の寄与の合計が予測値（分類は対数オッズか確率）と一致する（2026-11-26 テストで確認）。
- `train_filters` で 9 月までに絞って学習すると、1 年分を `mloom predict` に渡したとき、10 月以降の行も予測され、`used_for_training` が false になる（2026-11-26 テストで、`train_filters` で外した行が予測されて false になることを確認）。
- 回帰の `mloom predict` で、学習期間の後の行だけに `error` が入り、`rmse`・`r2` がその行だけで計算される（2026-11-26 テストで確認）。
- 閾値を与えると、`cm` と `accuracy`・`precision`・`recall` が、学習期間の後の行だけで計算される（2026-11-26 テストで確認）。
- `str_contains_all`・`str_contains_any` と、日時の列との比較で、学習の行を絞れる（2026-11-26 テストで確認）。
- 説明変数に無限大がある行も、学習から外されず、予測でも値が返る（2026-11-26 テストで確認）。
- クリップと補完のそれぞれの方式で、学習と予測ができる（2026-11-26 テストで確認）。
- 上の 4 つのモデルは、`lightgbm` を `hist_gradient_boosting` に置き換えたもので確かめる（2026-11-26 テストで確認）。
- ジャンクションで配った Skill の手順で clone の場所が求まり、利用側リポジトリの直下から `uv run --project <clone>` で `mloom optimize` と `mloom predict` を実行できる（2026-11-26 Issue #7 で `tmp/` の偽の clone の置き場所と利用側リポジトリで確認。実際の Claude Code のセッションからの実行は未確認で、merge 後にユーザーが確かめる）。
