# Cosmeaへの貢献

[English](CONTRIBUTING.md)

## プロジェクト設定

| 項目 | 正本 |
| --- | --- |
| ライブラリ | `src/cosmea`。CPU・単一学習プロセス・有限dataset |
| Python | 開発版は`.python-version`、対応下限は`pyproject.toml` |
| ツール | `mise.toml`でuvとBetterleaksを固定 |
| 依存関係 | 実行時の範囲と`train` extraは`pyproject.toml`、開発・文書ビルド用の依存は`dev`／`docs`グループ |
| 再現性 | 主な開発・CI環境を`uv.lock`で固定 |
| 品質検査 | Ruff整形/lint、mypy、pytest、MkDocsの厳格ビルド、配布物ビルド、公開内容・ライセンス監査、Betterleaks |
| 文書 | 日英のクイックスタート、利用例、組み込みガイド、設計ガイド。docstringから共通のAPIリファレンスを生成 |
| ライセンス | Apache-2.0。LICENSEとNOTICEをwheel／sdistへ収録 |

package versionの正本は`project.version`で、公開`__version__`も一致させる。metadataの順序・抽出方法・マスク・既定値を変える場合は、API文書と振る舞いのテストを更新する。標本設計・資源の制限は[設計](docs/ja/design.md)を参照。

## セットアップと検査

[mise](https://mise.jdx.dev/getting-started.html)を導入し、checkoutの設定を確認してから実行する。

```bash
git clone https://github.com/tkhshtsh0917/cosmea.git
cd cosmea

mise trust
mise install
mise run setup
mise run check
```

`setup` は固定した学習依存と `dev`、`docs` グループを導入する。`check` は `uv run --frozen --no-sync` と非隔離ビルドを使う。依存を変更する場合は `pyproject.toml` を編集し、`mise exec -- uv lock` を実行してから、差分を確認してsetupとcheckを実行する。整形には `mise run format` を使う。

主なCI jobは同じ固定版のsetup/checkを実行する。互換性jobではPython 3.11とTorch 2.6／NumPy 2.0、Python 3.13と宣言範囲内の依存を明示的に解決し、導入版を記録してテストする。これらは固定した基準環境の代わりではない。Actionsはcommit SHAで固定し、更新時はSHAと版のコメントを一緒に変える。

## テストと利用例

`tests/`直下は本体のテストで、小さな全列挙oracleとTrainer統合を含む。`tests/tooling/`は開発用監査、`tests/experiments/`は合成実験の指標を検証する。すべて`mise run check`の対象とする。

実際のDataLoader workerテストはプロセス初期化を通すため、OSの共有メモリが必要。制限環境では次の2群に分け、未実行の群があれば明記する。

```bash
mise exec -- uv run --frozen --no-sync pytest -m 'not workers' -q
mise exec -- uv run --frozen --no-sync pytest -m workers -q
```

## オフライン学習例と任意の実験

`mise run smoke`は、小さなローカルBERTと合成データで通常版・Cached版の例を実行する。モデルのダウンロードは不要。標準のcheckでは採掘負例・最後の1行バッチを含むTrainer学習を検証するため、smokeは任意実行とする。

長めの合成データによる品質比較は次のコマンドで実行する。

```bash
mise exec -- uv run --frozen --no-sync python experiments/toy_two_tower.py --seeds 3 --out results.json
```

回帰テストとは別に統計的性質を観察する場合は次を使う。

```bash
mise exec -- uv run --frozen --no-sync python experiments/diagnostics/estimators.py --trials 2000
mise exec -- uv run --frozen --no-sync python experiments/diagnostics/stationarity.py --max-negatives 2
```

これらは特定の分布での観察であり、一般的な分散の優劣や一致性の証明ではない。全列挙の計算量は指数的に増える。結果には引数・seed・ソース版・環境を添える。この改訂では既定の品質実験全体を再実行していない。指標と前提は[設計](docs/ja/design.md)を参照。

[SciFactベンチマーク](docs/ja/benchmark.md)では、実データで初期モデル・標準MNRL・Cosmeaを比較する。

## レビューと読みやすいコード

`src/cosmea`・`examples`・`tests`・`experiments`・`scripts`の責務を分ける。入力検証・計算・状態更新の間に意味のある空行を入れる。複数行の呼出しは1行1引数と末尾カンマを使い、短い単純な呼出しは1行でよい。docstringにはshape・失敗条件・重要な制約を説明する。

修正には、修正前に失敗するテストを用意し、可能なら独立した数式のoracleを使う。必要な境界を検証し、同じ性質をすべての設定の組合せで繰り返さない。合成データの接続検査は実データの品質を保証しない。性能の主張には入力・版・測定方法を記録する。

日英文書を同時に更新する。公開資料はローカルノートなしで完結させ、個人パス・非公開リンクを含めない。エージェント固有の操作規則は[AGENTS.md](AGENTS.md)に置く。

## ドキュメントサイト

`mise run docs` でCIと同じ厳格モードのビルドを実行する。生成先は `build/docs/`。手元で表示を確認するには `mise exec -- uv run --frozen --no-sync mkdocs serve` を実行する。文書だけを編集する環境では、先に `mise exec -- uv sync --locked --only-group docs` を実行する。

サイトとGitHubは同じMarkdownを使う。手順はREADMEと利用例、Trainerへの接続は組み込みガイド、数式と前提は設計ガイドに記載する。引数や戻り値の詳細はソースのdocstringを更新する。文書サイトでは[公開APIの一覧](docs/reference.md)に、生成したリファレンスを追加する。日本語のサイト内検索は文中の語を拾えないことがある。日本語メニューから各ガイドを開ける。

公開するには、GitHubの **Settings → Pages → Source** で **GitHub Actions** を選び、Actionsタブから `main` を対象に **Documentation Pages** を実行する。このワークフローは生成した文書サイトを配信する。通常のCIでは文書のビルドまでを確認する。[GitHubの公開手順](https://docs.github.com/ja/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)。

## ビルド・ライセンス・秘密情報の検査

`mise run build`はdistにwheelとsdistを1つずつ作る。`secrets` taskは公開内容・ライセンスを監査してからBetterleaksを実行し、checkから一度だけ呼び出す。監査だけを行う場合は`mise run audit`を使える。

監査はライセンスmetadata、LICENSE／NOTICEのbyte一致、型情報のmarker、sdistの収録物が公開ソース一覧または指定したビルド生成metadataに含まれることを確認する。秘密情報の検出とは別の検査である。PEP 639のためsetuptools 77.0.3以上を使い、開発版はlockする。package versionを変えたら古い成果物を整理する。

Betterleaksは固定版の組込み規則で、公開ファイル・展開した成果物・ローカルの全Git refを検査する。providerへの認証情報検証は無効。検出・実行エラーはいずれも失敗とし、マスク済みレポートはGit除外された`build/security/`へ置く。shallow cloneでは履歴検査が失敗するため、利用者が不足履歴を取得してから再実行する。パターン検出は秘密情報が一切ないことを保証しない。

依存先にはそれぞれのライセンスが適用される。STはApache-2.0で、PyTorch・NumPyの配布物は複数のライセンスを含む。素材を取り込む際は必要な第三者表示を保持する。パッケージ公開前にはホストCI・成果物・ライセンスを確認し、checksumを記録して管理者のリリース手順に従う。パッケージ公開jobは設定していない。
