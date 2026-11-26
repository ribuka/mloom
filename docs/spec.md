# mloom 仕様

Parquet のデータと、ユーザーが選んだモデルを渡すと、**列の除外 → 特徴量選択 → ハイパーパラメータ探索（HPO）→ 最終学習 → pickle 出力**までを自動で行う Python ライブラリ。エージェントからは、同じリポジトリに置く Skill（`SKILL.md`）から薄い CLI を通して使う。

- 本書はたたき台（2026-11-26 作成）。出典は会話記録「モデル最適化 Skill 化案」の抜粋と、その会話の最初のユーザーの入力。
- 「決定」は会話で合意した事項、「案」は本書で提案する未確定の事項。案は「未決事項」で確認してから決定に移す。
- 案のまま、試しに動かせるたたき台を実装した（2026-11-26、v0.1.0）。実装で値を決めた箇所は、その旨を書いている。
- たたき台の実装の後に、未決 8・10・12・13・16・18 を決め、行のフィルタ、クリップ、補完の方式の選択、`hist_gradient_boosting` を足すことを決めた（2026-11-26 ユーザー判断）。これらはまだ実装していない。該当する箇所には「（未実装）」と書き、実装は後続の PR で行う。

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
| 勾配ブースティング | LightGBM をやめ、scikit-learn の `HistGradientBoostingRegressor`・`HistGradientBoostingClassifier`（`hist_gradient_boosting`）に置き換える（2026-11-26 ユーザー判断、未実装） | `model.pkl` を読む環境に lightgbm を要らなくするため。同じヒストグラム型の勾配ブースティングで、欠損をそのまま扱える。結果は LightGBM と一致しない（サンプルデータの時系列 5 分割で、既定値の RMSE は LightGBM 3.450、HistGradientBoosting 3.487）。行のサブサンプリング（LightGBM の `subsample`）は無い |
| 出力 | 学習済みの Pipeline 全体と行のフィルタの定義を `model.pkl` に、実行の記録を `run.json` に保存する | モデル本体だけを保存すると、予測する側で列の選択や欠損補完を再現する必要がある。Pipeline ごと保存すれば同じ処理を通せる |
| `model.pkl` の依存 | `model.pkl` は mloom が無い環境でも読めるようにする。Pipeline には scikit-learn・numpy などの部品と関数だけを入れ、mloom のクラスや関数を入れない（2026-11-26 ユーザー判断、未実装。未決 16） | 予測する環境に mloom を入れなくて済むようにするため |
| 予測の入口 | `pickle.load` に加えて、`mloom predict` の CLI と、同じ処理の Python API を用意する（2026-11-26 ユーザー判断、未実装。未決 10） | フィルタの適用と、フィルタの外の行を NaN にする処理を 1 か所にまとめ、Skill から予測まで実行できるようにするため |
| 行のフィルタ | 列と値の条件で、学習と予測に使う行を絞れる。条件は AND だけで、定義を `model.pkl` と `run.json` に残す（2026-11-26 ユーザー判断、未実装。「行のフィルタ」） | 運転条件などで、モデルを使う範囲を決めるため |
| 前処理の方式 | 説明変数のクリップと補完の方式を、設定ファイルで選べる（2026-11-26 ユーザー判断、未実装。「前処理」） | データによって適した前処理が違うため |
| 本体の形 | 処理の本体は決定論的な Python コードにする。Skill は「ユーザーの指定を CLI の引数に変換し、結果を説明する」だけにする | どの列を除外するか、どう評価するか、何を保存するかを、実行のたびに変わらないようにするため |
| 入口 | **Python API が本体**。Skill 向けに薄い CLI を用意する。CLI は Python API を呼ぶだけにする | 人は Notebook や既存コードから API で使い、エージェントは設定を渡して 1 回実行し、終了コードと `run.json` で結果を確かめる。センサ除外・特徴量選択・HPO の実装は 1 か所に置く |
| API の形 | `optimize_model(data, config)` が学習済み Pipeline と評価結果を返し、`result.save(output_dir)` で `model.pkl` と `run.json` を保存する | 同上 |
| データの渡し方 | API には Parquet のパス、または Polars の DataFrame を渡せる。CLI には Parquet のパスを渡す | 同上 |
| 無限大 | 説明変数の +inf・-inf は、学習時も予測時も欠損とみなす。目的変数に +inf・-inf がある行は、学習から外す（2026-11-26 ユーザー判断で、「説明変数に無限大がある行は行ごと外す」から変更。未実装。未決 18） | 予測時は行ごとに値を返すので行を外せず、学習時だけ行を外すと、学習と予測で扱いが食い違うため。プロセスデータの無限大は、ゼロ除算やヒストリアンのエラー値など「値が正しく取れていない」ことが多く、欠損として扱うのが一般的。無限大が多い列は欠損率のしきい値で除外される |
| 欠損率のしきい値 | 説明変数を除外する欠損率のしきい値を、利用側リポジトリの `config/mloom.toml` で変えられる（2026-11-26 ユーザー判断。「設定ファイル」） | データによって適したしきい値が違うため |
| 目的変数が欠損した行 | 説明変数だけから計算する統計量（補完の中央値、標準化の平均・標準偏差）に、目的変数が欠損した行も使う。欠損率は「目的変数がある行」と「全行」の両方で見て、どちらかがしきい値を超えたら除外する。`config/mloom.toml` で使わないようにも切り替えられる（既定は使う）（2026-11-26 ユーザー判断） | ソフトセンサのように分析値がまばらなデータでは、統計量が安定するため。欠損率は、学習で実際の値から学べる量（目的変数がある行）と、予測時に補完に頼る量（全行）の両方を満たす列だけを残すため。非定常運転などで使いたくないデータもあるので切り替えられるようにする |

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
       mloom predict / mloom.predict（未実装）       pickle.load（mloom なしで読める。未実装）
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
| `filters` | | `[]` | 学習と予測に効く行のフィルタの条件（下の「行のフィルタ」。未実装）。CLI では、条件を書いた JSON ファイルのパスを `--filters` で渡す（2026-11-26 ユーザー判断。未決 19）。ファイルには `filters` と `train_filters` を書く |
| `train_filters` | | `[]` | 学習だけに効く行のフィルタの条件（下の「行のフィルタ」。未実装）。CLI では `--filters` のファイルに書く |
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

キー（3 のファイルの内容）:

```toml
[columns]
# 欠損率がこの値を超える説明変数を除外する（high_missing）。0 〜 1。既定 0.5
max_missing_rate = 0.5

[preprocessing]
# 目的変数が欠損した行を、欠損率と、補完・標準化の統計量に使うか。既定 true
use_unlabeled_rows = true
```

追加するキー（2026-11-26 決定、未実装。キーの名前と既定値は案。意味は「前処理」）:

```toml
[clip]
# none、quantile、iqr、mad、sigma。既定 none
method = "none"
# quantile の下側と上側の分位点。既定 0.01 と 0.99
lower_quantile = 0.01
upper_quantile = 0.99
# iqr の k（Q1 - k·IQR 〜 Q3 + k·IQR）。既定 1.5
iqr_k = 1.5
# mad の k（中央値 ± k × 1.4826 × MAD）。既定 3.0
mad_k = 3.0
# sigma の k（平均 ± k·σ）。既定 3.0
sigma_k = 3.0

[impute]
# auto、none、median、mean、most_frequent、constant、knn、iterative。既定 auto（モデルごとの既定）
method = "auto"
# constant で埋める値。既定 0.0
fill_value = 0.0
# knn の近傍の数。既定 5
n_neighbors = 5
```

| 状況 | 扱い |
| --- | --- |
| 1 や 2 のファイルが無い | 無視して次を探す。`run.json` の `settings.files[].found` が `false` になる |
| キーが無い | 次のファイルを探す |
| どのファイルでも、TOML として読めない、知らない表やキーがある、値が範囲外 | エラーで止める（終了コード 2）。打ち間違いで黙って既定値になるのを防ぐため |

- 使った値と、各キーを読んだファイルを `run.json` の `settings` に残す（`sources`）。

## 処理の流れ

1. データを読む（Polars）。
2. 学習に使う行を選ぶ（下の「行のフィルタ」「行の除外」）。
3. 列を除外する（下の「列の除外」）。除外した列と理由をすべて記録する。
4. CV の分割を決める（下の「CV」）。
5. Optuna で、特徴量選択とモデルのハイパーパラメータを同時に探索する（下の「特徴量選択」「HPO」）。各試行の評価は CV の平均スコア。
6. 最良の設定で、学習に使う行のすべてを使って Pipeline を学習し直す。
7. `model.pkl` と `run.json` を保存する。

評価側のデータの情報が学習に漏れないように、次のように fit する。

- 特徴量選択とモデル（目的変数を使う）は Pipeline の中に置き、CV の分割ごとに、学習側の目的変数がある行だけで fit する。特徴量選択のランク付けは Pipeline の外に移す（「特徴量選択」。未実装）が、学習側の目的変数がある行だけで計算するのは変わらない。
- 補完と標準化（目的変数を使わない）は、CV の分割ごとに「学習側の目的変数がある行」と「学習側に当たる目的変数が欠損した行」（「CV」）で先に fit して固定し（scikit-learn の `FrozenEstimator`）、Pipeline に入れる。試行ごとには fit し直さない。クリップの上下限も同じ行で先に計算する（「前処理」。未実装）。
- 最終モデルの補完と標準化は、目的変数がある行と、目的変数が欠損した行のすべてで fit する。

### 行のフィルタ

決定事項（2026-11-26 ユーザー判断、未実装）:

- 列と値の条件（`filters`）を満たす行だけを、学習と予測に使う。条件どうしは AND だけ（OR は書けない）。
- 定義は `model.pkl` と `run.json` の両方に残す。予測時に、条件を満たさない行の予測値は NaN にする（「`model.pkl`」「予測」）。
- 定義は polars の式（`pl.Expr`）ではなく、列名・演算子・値を並べたデータにする。読むのに polars も要らず、`run.json` にもそのまま書けるため。`pl.Expr` の pickle は polars の版をまたいで形式が保証されないと理解している（未確認）。

学習だけに効くフィルタ（`train_filters`。2026-11-26 ユーザー判断、未実装）:

- `filters` とは別のリストに書く。形と演算子は `filters` と同じ。例: 1 年分のデータのうち 9 月までで学習し、1 年分すべてを予測する。
- 学習時は、`filters` と `train_filters` の両方を満たす行だけを使う。`train_filters` で外れた行は、補完・標準化の統計量にも使わない。学習期間の外の説明変数の分布を、学習に入れないため。
- 予測時は `train_filters` を使わない。外れた行も予測し、`passed_filter` は `filters` だけで決まる。学習に使っていないことは、`used_for_training` が false になることでわかる。
- `run.json` と `model.pkl` の両方に残す。`model.pkl` の `train_filters` は、予測では行を絞らず、`used_for_training` の判定にだけ使う（「予測」）。
- 学習期間の外の行でのスコアは、`mloom optimize` では計算しない（2026-11-26 ユーザー判断）。HPO と特徴量選択は学習期間の中の CV だけで決める。学習期間の外のスコアを `run.json` に出すと、それを見ながら設定を変えて実行を繰り返し、学習期間の外のデータでチューニングしたのと同じことになりやすいため。学習期間の外での評価は、`mloom predict` の役目にする（「予測」）。

形式（キーの名前は案）。CLI の `--filters` に渡す JSON ファイルの中身:

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
- 日時・日付の列（Datetime、Date）と比べる値は、ISO 8601 の文字列で書き、列の型に変換して比べる（案）。タイムゾーン付きの列と比べる値には、オフセットを付けないとエラーで止める（案）。

以下は案:

- 学習時は、「行の除外」の前にフィルタを適用する。目的変数が欠損した行（統計量に使う行）にも適用する。外した行数を `run.json` の `data` に残す（`filters` で外した行は `n_rows_filtered_out`、`train_filters` で外した行は `n_rows_train_filtered_out`）。
- 列がデータに無い、`op` を知らない、`value` の形が `op` に合わないときは、学習を始める前にエラーで止める（終了コード 2）。
- 1 行も残らなければエラーで止める（終了コード 2）。
- フィルタに使う列を、それだけの理由では説明変数から外さない。外すかどうかは、ほかの列と同じく「列の除外」で決まる。外したいときは `exclude_columns` で指定する。

### 行の除外

次の順に行を分け、外し、それぞれの行数を `run.json` の `data` に残す。

| 順 | 扱い | `run.json` |
| --- | --- | --- |
| 1 | 目的変数が欠損（null、NaN）の行を、学習と評価から外す。`use_unlabeled_rows` が true なら、統計量用の「目的変数が欠損した行」として残し、false なら捨てる | `n_rows_without_target` |
| 2 | 学習用の行と、残した目的変数が欠損した行のうち、目的変数か、説明変数になりうる列に +inf・-inf がある行を外す（2026-11-26 ユーザー判断） | `n_rows_with_infinite`、列ごとの行数は `infinite_by_column` |
| 結果 | 学習と評価に使う行、統計量だけに使う目的変数が欠損した行 | `n_rows_used`、`n_unlabeled_rows_used` |

- 「説明変数になりうる列」は、列名で除外される列（目的変数、`time_column`・`group_column`、`exclude_columns`、使えないセンサ）を除いた浮動小数の列。列名で除外される列の無限大では行を外さない。
- 無限大の判定は、欠損率などで列を除外する前に行う。そのため、あとで `high_missing` などで除外される列の無限大でも行は外れる。無限大が多い列は `infinite_by_column` で見つけ、`exclude_columns` か `unavailable-sensors.json` で外す。
- 1 行も残らなければエラーで止める（終了コード 2）。
- 予測時（`model.pkl` の `predict`）は行を外せない（行ごとに予測を返すため）。たたき台の実装では、入力に無限大があると scikit-learn の `ValueError` になる。

変更（2026-11-26 ユーザー判断、未実装。未決 18）: 説明変数の無限大では行を外さず、欠損とみなす。

- 順 2 で行を外すのは、目的変数に +inf・-inf がある行だけにする（キーの名前は案。`n_rows_with_infinite_target`）。
- 説明変数になりうる列の +inf・-inf は、Pipeline の中で、列の選択の直後（クリップより前）に NaN に変える。`FunctionTransformer(np.nan_to_num, kw_args={"nan": np.nan, "posinf": np.nan, "neginf": np.nan})` を使えば、mloom の関数を pickle に入れずに済む（2026-11-26 scikit-learn 1.9.1 で確認）。学習時も予測時も同じ処理を通る。
- クリップより前に置くのは、無限大がクリップで上下限まで切られて、もっともらしい値として扱われるのを防ぐため。
- 欠損率（`high_missing`）は、無限大も欠損として数える。
- 列ごとの無限大の行数は、今と同じく `infinite_by_column` に残す。無限大が出ていた列を後から確かめられるようにするため。

### 列の除外

特徴量選択の前に、目的変数を使わずに決められる列を外す。外した列は理由とともに `run.json` の `excluded_columns` に残す。

| 理由 | 対象 |
| --- | --- |
| `target` | 目的変数 |
| `role` | `time_column`、`group_column` |
| `user_excluded` | `exclude_columns` で指定した列 |
| `unavailable_sensor` | 使えないセンサ |
| `non_numeric` | 数値でも真偽値でもない列（案）。真偽値は 0/1 の数値として扱う |
| `all_null` | すべて欠損の列 |
| `constant` | 欠損を除いて値が 1 種類の列 |
| `high_missing` | 「目的変数がある行での欠損率」か「目的変数が欠損した行も含めた全行での欠損率」のどちらかが、設定ファイルの `max_missing_rate`（既定 0.5）を超える列。`detail` に両方の欠損率を残す |

- 浮動小数の列の NaN は欠損として扱う。
- 欠損率は「行の除外」の後の行で計算する。`use_unlabeled_rows` が false なら、2 つの欠損率は同じになる。
- `all_null` と `constant` は、目的変数がある行だけで判定する（モデルが学ぶのはこの行だけのため）。
- 列の除外は CV の前に 1 回だけ行う。時系列の CV でも、欠損率は評価期間を含む全期間で計算する（目的変数を使わないので、ラベルの情報は漏れない）。
- `high_missing` に当たらない欠損は、Pipeline の中で中央値で補完する（「モデル」）。補完の方式を選べるようにする（「前処理」。未実装）。
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

決定事項（2026-11-26 ユーザー判断、未実装）:

- Pipeline の前処理の順は、次のとおり。
  1. 列の選択
  2. 無限大を NaN に変える
  3. クリップ
  4. 欠損の補完
  5. 標準化（モデルによる）
  6. 採用した列の選択（「特徴量選択」）。この後に推定器が続く
- どの部品も scikit-learn・numpy にあるものだけで組み、mloom のクラスや関数、lambda を入れない（「`model.pkl`」）。
- クリップの上下限と補完の統計量は、統計量用の行（「処理の流れ」）で先に計算し、固定する。

#### クリップ

- すべての説明変数に、設定ファイルの `[clip] method` で選んだ 1 つの方式を使う。既定は `none`（切らない）。

| `method` | 下限 〜 上限 |
| --- | --- |
| `none` | 切らない |
| `quantile` | `lower_quantile` 分位点 〜 `upper_quantile` 分位点（既定 1% 点 〜 99% 点） |
| `iqr` | Q1 − k·IQR 〜 Q3 + k·IQR（k は `iqr_k`。既定 1.5。3.0 などにもできる） |
| `mad` | 中央値 ± k × 1.4826 × MAD（k は `mad_k`。既定 3.0） |
| `sigma` | 平均 ± k·σ（k は `sigma_k`。既定 3.0） |

- 上下限の計算には欠損を使わない。欠損はクリップの後も欠損のまま残る。
- `FunctionTransformer(np.clip, kw_args={"a_min": 下限の配列, "a_max": 上限の配列})` で実装すれば、mloom の関数を pickle に入れずに済む（2026-11-26 scikit-learn 1.9.1 で確認）。
- IQR か MAD が 0 の列（ほとんど一定で、ときどきだけ動くセンサなど）は、上下限が 1 点に潰れて値がすべて同じになる。そうした列はクリップせず、`run.json` に残す（案）。
- 使った方式と、列ごとの上下限、クリップしなかった列を `run.json` に残す（形は案）。

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

| `model` | `auto` の補完 | 理由 |
| --- | --- | --- |
| `ridge`、`pls` | `median` | 欠損を扱えない |
| `hist_gradient_boosting`、`random_forest` | `none` | 分割のたびに欠損の行を送る側を学習するので、欠損であること自体を情報として使える。中央値で埋めると、欠損の行が「たまたま中央値だった行」と区別できなくなる |

- `random_forest` が欠損をそのまま扱えるのは、scikit-learn 1.4 以降と理解している。実装の前に scikit-learn 1.9.1 で確かめる（未確認）。
- 学習時に欠損が無かった列に、予測時だけ欠損が来たとき、`HistGradientBoosting` は学習で行が多かったほうの枝に送ると理解している（scikit-learn のドキュメントの記述。未確認）。
- `IterativeImputer` は、作るときは `sklearn.experimental.enable_iterative_imputer` の import が要るが、pickle を読む側では要らない（2026-11-26 scikit-learn 1.9.1 で確認）。

### 特徴量選択（案）

- Pipeline の中で、各列を目的変数との関係でランク付けし（回帰は `f_regression`、分類は `f_classif`）、上位 `k` 列を使う（`SelectKBest`）。
- `k` は HPO で探索するハイパーパラメータの 1 つにする（1 〜 候補列数、対数スケール）。
- 最終学習で選ばれた列を「採用した列」として `run.json` に残す。
- ランク付けの方法（`mutual_info_regression`、モデルの重要度、相関の高い列の間引きなど）は「未決事項」。

変更（2026-11-26、未実装。案）: 補完しない既定（「前処理」）と両立させるため、`SelectKBest` を Pipeline から外す。

- `SelectKBest(f_regression / f_classif)` は NaN を受け付けない。補完しないモデルの前に置くと、欠損を含むデータで fit が `ValueError: Input X contains NaN` になる（2026-11-26 scikit-learn 1.9.1 で確認。クロスレビューの指摘）。
- ランク付けは Pipeline の外で行う。CV の分割ごとに 1 回だけ、学習側の目的変数がある行で、前処理の後の値（補完しないモデルでは、ランク付けのためだけに中央値で補完した写し）を使って `f_regression` か `f_classif` で計算する。
- 推定器には、補完していない元の値（欠損を含む）を渡す。
- 試行ごとの `k` に応じて、その分割の上位 `k` 列を選ぶ。目的変数を使うランク付けは学習側の行だけで行うので、評価側の情報は漏れない。ランクは試行をまたいで使い回す。
- Pipeline の中では、前処理の後に、`ColumnTransformer([("keep", "passthrough", 採用した列の番号)], remainder="drop")` で列を選ぶだけにする。
- 2026-11-26 に scikit-learn 1.9.1 で確かめたこと: 列の番号で選ぶ構成は、欠損と無限大を含む polars の表で学習・予測でき、pickle に `mloom` という文字列が含まれなかった。
- `KNNImputer` や `IterativeImputer` は、ほかの列の値から補完する。そのため、前処理は候補列のすべてで行い、採用した列はその後で選ぶ。

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
| `ridge` | `Ridge` | `LogisticRegression`（L2） | 中央値で補完、標準化 | `alpha`: 1e-4 〜 1e3（対数）。分類では `C = 1 / alpha` |
| `pls` | `PLSRegression`（`scale=False`） | 未対応（指定するとエラー。未決 1） | 中央値で補完、標準化 | `n_components`: 1 〜 min(20, 候補列数)。`k` を超える分は `k` に切り詰める |
| `random_forest` | `RandomForestRegressor` | `RandomForestClassifier` | 中央値で補完（未実装の変更: 既定は補完しない。「前処理」） | `n_estimators`: 50 〜 300、`max_depth`: 2 〜 32、`min_samples_leaf`: 1 〜 20、`max_features`: 0.1 〜 1.0 |
| `lightgbm` | `LGBMRegressor` | `LGBMClassifier` | 中央値で補完 | `num_leaves`: 8 〜 256、`learning_rate`: 1e-3 〜 0.3、`n_estimators`: 50 〜 1000、`min_child_samples`: 5 〜 100、`subsample`・`colsample_bytree`: 0.5 〜 1.0、`reg_lambda`: 1e-8 〜 10 |
| `hist_gradient_boosting`（未実装。`lightgbm` を置き換える） | `HistGradientBoostingRegressor` | `HistGradientBoostingClassifier` | 補完しない（「前処理」） | `learning_rate`: 1e-3 〜 0.3、`max_iter`: 50 〜 1000、`max_leaf_nodes`: 8 〜 256、`min_samples_leaf`: 5 〜 100、`l2_regularization`: 1e-8 〜 10、`max_features`: 0.5 〜 1.0（いずれも案。`lightgbm` の範囲を対応する引数に移した。行のサブサンプリングは無い） |

- 探索範囲の正は `mloom.models`。上の表は 2026-11-26 のたたき台の実装の値。
- 再現性のため、モデルの `n_jobs` は 1 に固定し、`random_state` に `seed` を渡す。
  - `HistGradientBoosting` には `n_jobs` が無く、OpenMP で全コアを使う。スレッド数が結果を変えないかは、実装の前に確かめる（未決 14）。

- Pipeline の構成: 入力列の選択 → 前処理（欠損の補完 →（標準化）。fit 済みで固定）→ 特徴量選択 → 推定器。前処理の fit に使う行は「処理の流れ」。前処理を足す変更は「前処理」（未実装）。
- 分類の `ridge` に `RidgeClassifier` ではなく `LogisticRegression` を当てるのは、`log_loss` と `roc_auc` に要る確率（`predict_proba`）を出せるため（案）。
- `lightgbm` は追加の依存（`mloom[lightgbm]`）にする（案）。入っていない環境で `model: lightgbm` を指定したら、インストール方法を示してエラーで止める。`hist_gradient_boosting` に置き換えるときに、`lightgbm` と追加の依存をなくす（未実装）。
- `hist_gradient_boosting` の `early_stopping` は `False` に固定する（2026-11-26 ユーザー判断、未実装）。既定の `"auto"` は、1 万行を超えると学習データの 10% を内部の検証に取ってしまい、CV と最終学習で使う行が変わるため。
- `task` に対応しないモデルを指定したら、学習を始める前にエラーで止める（終了コード 2）。

## 出力

`output_dir` に次を保存する。

| ファイル | 内容 |
| --- | --- |
| `model.pkl` | 学習済みの Pipeline 全体（前処理・特徴量選択・モデル）。予測時も同じ処理を通す。行のフィルタの定義も入れる（未実装） |
| `run.json` | 採用した列、除外した列と理由、最良のパラメータ、CV の分割、スコア、seed、ライブラリの版など |

### `model.pkl`

- 標準ライブラリの `pickle` で保存する。
- 読むときは `pickle.load` の結果の `predict` に、候補列（`run.json` の `candidate_columns`）を含む表（polars または pandas の DataFrame）を渡す。Pipeline の先頭の `ColumnSelector` が列を名前で選ぶので、列の順は問わず、余分な列は無視される（2026-11-26 実装・テストで確認）。
- `ColumnSelector` は mloom のクラスなので、**pickle を読む環境にも mloom が要る**（2026-11-26 実装で判明した制約）。
- pickle は保存したときと同じ版の scikit-learn などで読む前提。版は `run.json` に残す。
- 信頼できない pickle を読まない（pickle は読むときに任意のコードを実行できる）。README に書く。

変更（2026-11-26 ユーザー判断、未実装。未決 16）: mloom が無い環境でも読めるようにする。

- `ColumnSelector` をやめ、scikit-learn の `ColumnTransformer([("keep", "passthrough", 候補列)], remainder="drop")` で列を選ぶ。
- 2026-11-26 に scikit-learn 1.9.1・polars 2.2.0 で確かめたこと:
  - polars と pandas の表のどちらでも予測できる。
  - 列の順を入れ替えても、余分な列があっても、同じ予測になる。
  - 候補列が無い表では、`ValueError: columns are missing` になる。
  - null がある整数の列と bool の列は、float64 の NaN になって補完される。
  - pickle に `mloom` という文字列が含まれない。
- 予測する環境には、学習時と同じ版の scikit-learn・numpy・scipy と、入力に使う polars か pandas が要る。版は `run.json` の `versions` を見て揃える。
- pickle の中身は Pipeline ではなく、次の辞書にする（キーの名前は案）。行のフィルタは scikit-learn の Pipeline に入れられないため（部品は行の数を変えられない。学習時は目的変数とずれ、予測時は入力の行と予測が対応しなくなる）。

```python
{
    "pipeline": Pipeline,   # 列の選択 → 前処理 → 特徴量選択 → 推定器
    "filters": [...],       # 「行のフィルタ」の定義。run.json の filters と同じ
    "train_filters": [...], # 学習だけに効くフィルタ。予測では used_for_training の判定にだけ使う
    "training_rows": {...}, # 学習に使った行の照合用（「予測」）。照合に使う列、行ごとのハッシュ、最も新しい時刻（last_time）
}
```

- mloom を使わずに予測する手順（README に書く）: `pickle.load` → `filters` の条件を AND で評価 → 条件を満たす行だけ `pipeline.predict` → 満たさない行は NaN。

### 予測（未実装）

決定事項（2026-11-26 ユーザー判断。未決 10）: `mloom predict` の CLI と、同じ処理の Python API を用意する。

- 処理: `model.pkl` を読み、フィルタを適用し、条件を満たす行だけを予測し、満たさない行は NaN にして、入力と同じ行数・同じ順で返す。
- 説明変数の無限大は Pipeline の中で欠損になるので、予測の入口では特別に扱わない（「行の除外」）。
- この入口を使う環境には mloom が要る。`model.pkl` 自体は mloom なしでも読める。

出力の列（2026-11-26 ユーザー判断。未決 20。列の名前は案）:

| 列 | 内容 | 出す条件 |
| --- | --- | --- |
| `prediction` | 予測値。フィルタの条件を満たさない行は NaN（分類では null） | 常に |
| `--keep-columns` で指定した列 | 入力の列をそのまま出す（例: `timestamp`、`lot`）。予測と元の行を結び付けるため | 指定したとき |
| `passed_filter` | フィルタの条件を満たしたか（bool）。予測値が NaN の理由が、フィルタの外だからかどうかを見分けるため | 常に |
| `used_for_training` | 最終モデルの学習に使った行か（bool）。予測する表に目的変数の列が無ければ null | 常に |
| `proba_<クラス>` | 分類のクラスごとの確率（`predict_proba`） | 分類のとき |
| `shap_<列名>`、`shap_base` | 採用した列（`selected_columns`）ごとの SHAP 寄与と、基準値 | `--shap` を付けたとき |
| `error` | 予測 − 実測（「スコア」） | 回帰のとき |
| `rmse`、`r2` | スコアの対象の行全体での RMSE と R²。全行に同じ値（「スコア」） | 回帰のとき |
| `cm` | 混同行列のどこに入るか（`tp`・`fp`・`fn`・`tn`）（「混同行列」） | 閾値（分類では陽性のクラス）を与えたとき |
| `accuracy`、`precision`、`recall` | スコアの対象の行全体の `cm` の数から計算した値。全行に同じ値 | 閾値（分類では陽性のクラス）を与えたとき |

学習に使った行の判定（`used_for_training`。2026-11-26 ユーザー判断）:

- 対象は、最終モデルの学習に使った行だけ。`filters` と `train_filters` を通り、目的変数があり、目的変数が無限大でない行。補完・標準化の統計量にだけ使った行（目的変数が欠損した行）と、`train_filters` で外した行は false。
- 予測する表とは、行の値で照合する。学習時に、最終モデルの学習に使った行の値を行ごとにハッシュにして、`model.pkl` の `training_rows` に保存する。ハッシュにする値は、候補列・目的変数、それに `time_column`・`group_column` の指定があればその値。
- 予測時は、同じ列で同じハッシュを作る。`filters` の条件を満たし（`passed_filter`）、`train_filters` の条件も満たし、かつハッシュが一致する行を true にする。
  - ユーザーが照合用の列を指定しなくて済む。値が後から直された行は、学習に使っていないと判定される。
  - 目的変数をハッシュに入れるのは、説明変数が同じで目的変数の有無だけが違う行を区別するため。入れないと、統計量にだけ使った行も true になる（クロスレビューの指摘）。
  - `filters`・`train_filters` との AND をとるのは、フィルタに使った列が候補列に無いとき（文字列の列など）に、条件の外の行が、ハッシュにする値の同じ学習行と一致してしまうのを防ぐため（クロスレビューの指摘）。
  - `train_filters` に使う列が予測する表に無いときは、`used_for_training` を null（判定できない）にする（案）。`train_filters` は予測の行を絞らないので、エラーにはしない。
  - 予測する表に目的変数の列が無いときは、`used_for_training` を null（判定できない）にする（案）。新しいデータの予測では、目的変数の列が無いのがふつうなので、エラーにはしない。
  - 目的変数以外の照合に使う列が予測する表に無いときは、エラーで止める（終了コード 2。案）。
  - ハッシュは、polars の `hash_rows` を使わずに作る（案）。`hash_rows` は polars の版をまたいで同じ値になることが保証されず、mloom の版を上げると古い `model.pkl` と照合できなくなるため。

SHAP の寄与（2026-11-26 ユーザー判断）:

- `--shap` を付けたときだけ出す。計算に時間がかかることがあり、列も説明変数の数だけ増えるため。
- shap パッケージを、mloom の通常の依存にする。使うのは `mloom predict` の側だけで、`model.pkl` の依存は増えない。
- `hist_gradient_boosting`・`random_forest` は `shap.TreeExplainer`、`ridge`・`pls` は線形の SHAP（特徴量の独立を仮定）で計算する（案）。
- 2026-11-26 に shap 0.54.1・scikit-learn 1.9.1・numpy 2.5.5 で確かめたこと（欠損が 10% のデータ）:
  - `TreeExplainer` は、`HistGradientBoostingRegressor`・`HistGradientBoostingClassifier`・`RandomForestRegressor` で動いた。
  - 基準値と寄与の合計は、予測値（分類は `decision_function` の対数オッズ）と、1e-13 程度の誤差で一致した。
  - 1000 行にかかった時間は、`HistGradientBoosting` が約 1.5 秒、`RandomForest`（木 100 本、深さの制限なし）が約 52 秒。
- 列は、採用した列ごとの `shap_<列名>` と、基準値の `shap_base`。採用されなかった列は、寄与が常に 0 なので出さない。
- 分類の寄与の単位は、モデルによって違う（対数オッズか確率か）。多値分類では、クラスごとに列を出す（`shap_<クラス>_<列名>`）。
- フィルタの外の行は、SHAP の列も NaN にする。

スコア（2026-11-26 ユーザー判断、未実装。列の名前は案）:

- 回帰では、`error`（予測 − 実測）、`rmse`、`r2` を標準で出す。`rmse` と `r2` は、スコアの対象の行全体で 1 つの値で、全行に同じ値を入れる。
  - `error` の向きを「予測 − 実測」にしたのは、ユーザーの言い方に合わせたため（統計でいう残差は逆の向き）。プラスなら予測が過大。
- スコアの対象の行は、時刻が「最終モデルの学習に使った行の最も新しい時刻」より後の行だけ。その時刻と同じか、それより前の行は、`error`・`cm` を None にし、`rmse` などの計算にも入れない。学習期間の中の行で測ると、成績が良く見えすぎるため。
  - 最も新しい時刻は、学習時に `model.pkl` に保存する（キーの名前は案。`training_rows` の中の `last_time`）。
  - 同じ時刻の行を含めないのは、最も新しい時刻の行は学習に使った行だから。
  - 時刻の列は `time_column`。`time_column` を指定していない実行（`kfold`・`group` の CV など）では、「学習期間の後」が決められないので、スコアと混同行列の列をすべて None にする（2026-11-26 ユーザー判断）。
- 時刻が後でも、実測（目的変数）が欠損か無限大の行、フィルタの外で予測値が NaN の行は、`error`・`cm` を None にし、計算にも入れない。
- 予測する表に目的変数の列が無いときは、スコアと混同行列の列をすべて None にする。
- 対象の行が 0 行なら `rmse`・`r2` は None。1 行だけなら `r2` は None（案。R² は 2 行以上ないと計算できないため）。
- 分類では、`error`・`rmse`・`r2` を出さない（2026-11-26 ユーザー判断）。分類の成績は「混同行列」で見る。

混同行列（2026-11-26 ユーザー判断、未実装。列と引数の名前は案）:

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

### `run.json`

2026-11-26 のたたき台の実装の形（値は例。正は `mloom.result.Result.to_dict`）。2026-11-26 の決定で、次を変える（未実装。キーの名前は案）:

- `filters` と `train_filters` を足す（「行のフィルタ」）。`data` に `n_rows_filtered_out` と `n_rows_train_filtered_out` を足す。
- `data.n_rows_with_infinite` を `n_rows_with_infinite_target`（目的変数の無限大で外した行数）にする。`infinite_by_column` は残す。
- `settings` に `[clip]` と `[impute]` の値を足す。クリップの上下限と、クリップしなかった列を残す。
- `versions.lightgbm` をなくす。

```json
{
  "mloom_version": "0.1.0",
  "created_at": "2026-11-26T10:00:00+09:00",
  "data": {
    "path": "data/train.parquet", "n_rows": 12000, "n_columns": 320, "n_rows_used": 1990, "n_unlabeled_rows_used": 9950,
    "n_rows_without_target": 10000, "n_rows_with_infinite": 60, "infinite_by_column": {"TI-115": 60}
  },
  "target": "quality",
  "task": "regression",
  "model": "ridge",
  "metric": "rmse",
  "metric_direction": "minimize",
  "seed": 0,
  "unavailable_sensors": {
    "path": "config/unavailable-sensors.json",
    "found": true,
    "sensors": {"TI-101": "2026-10 から故障中", "FI-205": "廃止"},
    "not_in_data": ["FI-205"]
  },
  "settings": {
    "max_missing_rate": 0.3, "use_unlabeled_rows": true,
    "sources": {"max_missing_rate": "config/mloom.toml", "use_unlabeled_rows": "C:/.../mloom/config/mloom.toml"},
    "files": [
      {"role": "project", "path": "config/mloom.toml", "found": true},
      {"role": "local_defaults", "path": "C:/.../mloom/config/mloom.local.toml", "found": false},
      {"role": "defaults", "path": "C:/.../mloom/config/mloom.toml", "found": true}
    ]
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
  "versions": {"python": "3.13.13", "mloom": "0.1.0", "scikit-learn": "1.9.1", "optuna": "5.1.0", "polars": "2.2.0", "numpy": "...", "scipy": "...", "lightgbm": null}
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

予測（未実装。関数名と戻り値は案）:

```python
predictions = mloom.predict("outputs/ridge/model.pkl", "data/new.parquet")  # または polars.DataFrame
# 入力と同じ行数・同じ順。フィルタの条件を満たさない行は NaN
```

- 設定は `mloom.Config`（frozen dataclass）で渡す。CLI の引数と 1 対 1 に対応させる。データが要らない検査（モデル名、`task` とモデルの組み合わせ、CV に要る列の指定など）は `Config` を作った時点で行う。
- 不正な入力（目的変数が無い、モデル名が無い、候補列が 0 列など）は、学習を始める前に `mloom.InputError` を投げる。

## CLI（案）

```powershell
uv run mloom optimize data/train.parquet --target quality --task regression --model ridge --cv timeseries --time-column timestamp --output outputs/ridge
```

- CLI は引数を `Config` に変換し、`optimize_model` と `result.save` を呼ぶだけにする。
- 標準出力の最後に `run.json` のパスを出す。進捗やログは標準エラーに出す。

フィルタ（未実装。未決 19）:

```powershell
uv run mloom optimize data/train.parquet ... --filters config/filters.json
```

予測（未実装。引数の名前は案。出力の列は「予測」）:

```powershell
uv run mloom predict outputs/ridge/model.pkl data/new.parquet --output outputs/ridge/predictions.parquet [--keep-columns timestamp,lot] [--shap] [--cm-actual-threshold 0 --cm-predicted-threshold 1 --cm-positive above | --positive-class bad [--cm-predicted-threshold 0.5]]
```

- CLI は Python API の予測を呼んで、結果を Parquet に保存するだけにする。

| 終了コード | 意味 |
| --- | --- |
| 0 | 成功。`model.pkl` と `run.json` を保存した |
| 1 | 想定外のエラー |
| 2 | 入力の誤り（引数、データ、`unavailable-sensors.json`、`mloom.toml`） |

## Skill

- 本リポジトリの `skills/model-optimization/SKILL.md` に置く（2026-11-26 たたき台を作成）。
- 5 つの入力のうち足りないものはユーザーに聞く。CV の方法は Skill が推測して決めず、知っている列名を材料に聞き返す。
- 役割は 2 つだけ: ユーザーの指定を CLI の引数に変換して実行する、`run.json` を読んで結果を説明する（採用した列、除外した列と理由、スコア、最良のパラメータ）。
- どの列を除外するか、どう評価するか、何を保存するかを Skill で決めない。それらは Python 側で固定する。

### 配布と実行（未実装）

決定事項（2026-11-26 ユーザー判断。未決 12・13）:

- mloom は PC に 1 つ、利用側リポジトリとは別の場所に clone する。利用側リポジトリには mloom を入れない。
- Skill は、各ベンダー（Claude Code など）の既定の Skill の置き場所に、clone の `skills/model-optimization` を指すジャンクションを置いて配る。`git pull` すれば、mloom と Skill が同じ版に揃う。
- Skill は、利用側リポジトリの直下で `uv run --project <clone> mloom optimize ...` を実行する。
  - `--project` は使う環境（clone の `.venv` と `uv.lock`）を決めるだけで、カレントディレクトリは利用側リポジトリのまま。利用側の `config/mloom.toml`・`config/unavailable-sensors.json`・データ・出力先は、利用側の相対パスで指定できる。
  - mloom の `uv.lock` の版で動くので、予測する環境の scikit-learn などの版を揃えやすい。
  - mloom は clone から動くので、`config/mloom.local.toml` も読まれる（「設定ファイル」）。
- `<clone>` は、Skill のディレクトリ（ジャンクション）の実際のパスから 2 つ上のディレクトリとする。ジャンクションのパスのまま `..\..` をたどると、clone ではなくジャンクションを置いた場所の親に着くので、先に実際のパスを調べる（PowerShell の `(Get-Item <Skillのディレクトリ>).Target` など。手順は案）。
- リポジトリ全体をジャンクションにする形は採らない（案）。`SKILL.md` が `skills/model-optimization/` の 2 段下にあり、`<置き場所>/<名前>/SKILL.md` の 1 段下しか探さないベンダーでは見つからないため（Claude Code はそう理解している。ほかのベンダーは未確認）。

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
| LightGBM（追加の依存。`hist_gradient_boosting` への置き換えでなくす。未実装） | lightgbm |
| 予測時の SHAP の寄与（未実装） | shap |
| パッケージ管理 | uv |
| lint、型チェック、テスト | ruff、ty、pytest |

## パッケージ構成

| モジュール | 責務 |
| --- | --- |
| `mloom` | 公開 API の再エクスポート（`optimize_model`、`Config`、`Result`、例外） |
| `mloom.errors` | 例外（`InputError` は終了コード 2、それ以外は 1） |
| `mloom.config` | `Config` の定義と、データが要らない検査 |
| `mloom.metrics` | 評価指標と、タスクごとに選べる指標 |
| `mloom.models` | モデルのレジストリ（推定器、標準化の要否、探索空間） |
| `mloom.data` | データの読み込み |
| `mloom.sensors` | `unavailable-sensors.json` の読み込み |
| `mloom.settings` | 設定ファイル（`mloom.toml`）とその既定値の読み込み |
| `mloom.rows` | 学習に使う行の選択（目的変数の欠損、無限大） |
| `mloom.columns` | 列の除外と理由の記録 |
| `mloom.cv` | ユーザーが指定した CV の分割 |
| `mloom.pipeline` | Pipeline の組み立てと `ColumnSelector`（`ColumnSelector` は `ColumnTransformer` に置き換える。未実装） |
| `mloom.optimize` | `optimize_model`：HPO と最終学習 |
| `mloom.result` | `Result` と保存（`model.pkl`、`run.json`） |
| `mloom.filters`（未実装。名前は案） | 行のフィルタの定義の検査と適用 |
| `mloom.predict`（未実装。名前は案） | 予測の入口 |
| `mloom.cli` / `mloom.__main__` | CLI |

## 未決事項

| # | 論点 | 案 |
| --- | --- | --- |
| 1 | 分類の `pls` | PLS-DA（目的変数を one-hot にして `PLSRegression`、最大の列をクラスとする）を mloom で実装するか、分類では `pls` を選べないことにするか。たたき台の実装は後者（指定するとエラー） |
| 2 | 分類の範囲 | 2 値と多値の両方。クラスの不均衡への対応（`class_weight` を探索するか）は後回し |
| 3 | CV で選べる方法 | `timeseries`、`group`、`kfold` の 3 つ。時系列かつグループ（ロットごとに時刻順）の分割が要るか |
| 4 | `timeseries` の時刻の列 | `time_column` を必須にする。行の並び順を時刻順とみなす指定も許すか |
| 5 | 特徴量選択の方法 | `SelectKBest` の `k` を HPO で探索。相関の高い列の間引きを足すか |
| 6 | `unavailable-sensors.json` の形式と照合 | `{"sensors": [{"name", "reason"}]}`、列名の完全一致。接頭辞やパターンでの照合が要るか |
| 7 | 数値でない列の扱い | 除外する（エンコードしない） |
| 8 | 欠損の扱い | **決定**（2026-11-26）: 補完の方式を設定ファイルで選べる。既定は、欠損を扱えるモデル（`hist_gradient_boosting`、`random_forest`）では補完せず、ほかは中央値。→「前処理」 |
| 9 | 選べるモデル | 表の 4 つで始める。API から任意の推定器と探索空間を渡せるようにするか |
| 10 | 予測の入口 | **決定**（2026-11-26）: `mloom predict` の CLI と Python API を用意する。`pickle.load` だけでも予測できるようにする。→「予測」 |
| 11 | HPO の試行の記録 | `run.json` は最良の試行だけ。全試行を `trials.csv` に出すか |
| 12 | Skill の配布方法 | **決定**（2026-11-26）: 各ベンダーの既定の置き場所に、clone の Skill を指すジャンクションを置く。→「配布と実行」 |
| 13 | 利用環境への導入 | **決定**（2026-11-26）: 利用側リポジトリとは別に clone し、`uv run --project <clone>` で実行する。→「配布と実行」 |
| 14 | 並列実行 | Optuna の `n_jobs` やモデルの `n_jobs` を使うか（再現性とのトレードオフ） |
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

2026-11-26 の決定で、次の条件を足す（未実装）:

- mloom が無い環境で `model.pkl` を `pickle.load` し、README の手順で予測できる。
- `mloom predict` で予測すると、入力と同じ行数・同じ順で予測が返り、フィルタの条件を満たさない行は NaN になる。
- 学習に使った表をそのまま `mloom predict` に渡すと、最終モデルの学習に使った行だけで `used_for_training` が true になる（説明変数が同じで目的変数だけが欠損した行は false になる）。
- 欠損を含むデータで、補完しない既定のまま `hist_gradient_boosting` と `random_forest` を学習できる。
- `--shap` を付けると、行ごとに `shap_base` と SHAP の寄与の合計が予測値（分類は対数オッズか確率）と一致する。
- `train_filters` で 9 月までに絞って学習すると、1 年分を `mloom predict` に渡したとき、10 月以降の行も予測され、`used_for_training` が false になる。
- 回帰の `mloom predict` で、学習期間の後の行だけに `error` が入り、`rmse`・`r2` がその行だけで計算される。
- 閾値を与えると、`cm` と `accuracy`・`precision`・`recall` が、学習期間の後の行だけで計算される。
- `str_contains_all`・`str_contains_any` と、日時の列との比較で、学習の行を絞れる。
- 説明変数に無限大がある行も、学習から外されず、予測でも値が返る。
- クリップと補完のそれぞれの方式で、学習と予測ができる。
- 上の 4 つのモデルは、`lightgbm` を `hist_gradient_boosting` に置き換えたもので確かめる。
