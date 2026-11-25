# mloom

Parquet のデータと、ユーザーが選んだモデルを渡すと、使えないセンサの除外、特徴量選択、ハイパーパラメータ探索、最終学習までを自動で行い、学習済みの Pipeline を `model.pkl` に、実行の記録を `run.json` に保存する Python ライブラリです。

- Python API が本体です。エージェントは同梱の Skill から薄い CLI を通して使います。
- 仕様と未決事項は [docs/spec.md](docs/spec.md) にあります（現在はたたき台で、未実装です）。

## 注意

- `model.pkl` は pickle です。pickle は読むときに任意のコードを実行できるため、信頼できないファイルは読まないでください。

## 開発

```powershell
uv sync
uv run -m ruff check .
uv run -m ty check
uv run -m pytest --tb=short -q
```
