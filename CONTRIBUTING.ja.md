# Cosmeaへの貢献

[English](CONTRIBUTING.md)

## プロジェクト設定

| 項目 | 採用構成・正本 |
| --- | --- |
| 目的・成果物 | 検索モデルの学習用Pythonライブラリ `cosmea` |
| 対象環境 | CPU・単一学習プロセス・有限map-style dataset。macOS arm64でローカル検証。Ubuntu 24.04では[GitHub Actions](https://github.com/tkhshtsh0917/cosmea/actions/workflows/ci.yml)で同じ検査を実行 |
| 言語・版 | 開発用Pythonの固定版は `.python-version`、対応下限は `pyproject.toml` |
| ツール・依存管理 | miseが `mise.toml` のuv・Betterleaksを管理。uvがPythonと仮想環境を管理 |
| 依存の正本 | 互換範囲とextrasは `pyproject.toml`、開発環境の解決結果はGit管理する `uv.lock` |
| 整形・静的検査 | Ruff format・lint、全package moduleのmypy。設定は `pyproject.toml` |
| 検証コマンド | `mise run check`：整形、lint、型、テスト、オフライン学習例、ビルド、公開ファイル検査、Betterleaks |
| 互換性 | 公開クラス名とmetadata順序を維持。Qの尺度・mask・既定値の変更時は契約も更新 |
| 資源制約 | `0 <= K <= M`、`1 <= B <= N`、正のCachedチャンクサイズ。計算量・メモリの範囲は [設計](docs/ja/design.md) |
| 公開文書 | README・API・設計・貢献手順を日英で管理 |
| 配布・公開 | wheel／sdistのローカル生成。レジストリ・配信・公開リリースは未設定。プロジェクトのライセンスは未選定：[NOTICE](NOTICE.md) |
| 完了条件 | 品質検査成功、公開文書だけで利用可能、成果物に非公開記録を含めない。実データ性能は保証しない |
| Git操作 | 依頼範囲のローカル編集・検証。commit・履歴整理・push・tag・公開は個別に管理者の明示許可が必要。検査成功から許可を推定しない |

package versionの正本は `project.version`。公開 `__version__` も一致させる。実行時依存はSentence Transformers、PyTorch、NumPy。`train` extraはTrainer連携用のdatasets／Accelerate、`dev`は開発ツールを提供する。STはApache-2.0。PyTorch・NumPyの配布物は複数のライセンスを含み、導入版のmetadataと同梱LICENSEで確認する。これらはCosmea自体のライセンスとは別。STの固定minorを更新するときはコレーター、Trainerのmetadata転送、GradCacheを再確認する。

## セットアップと検証

[mise](https://mise.jdx.dev/getting-started.html) を導入し、checkoutの設定を確認してから実行する。

```bash
git clone https://github.com/tkhshtsh0917/cosmea.git
cd cosmea

mise trust
mise install
mise run setup
mise run check
```

依存導入は `setup` に分離する。検査は `uv run --frozen --no-sync` と非隔離ビルドを使い、ツールや依存を暗黙に変更しない。新しい環境では先にsetupを実行する。意図した更新では正本を変更し、`mise exec -- uv lock` の差分確認後にsetupとcheckを実行する。`mise run format` は整形適用、`format-check` は検査だけを行う。

テストはローカルの合成入力を使う。統計テストは小規模な全列挙・モンテカルロであり、性能ベンチマークではない。workerテストはOSの共有メモリを必要とする。制限環境では分けて実行し、未実行分があれば明記する。

```bash
mise exec -- uv run --frozen --no-sync pytest -m 'not workers' -q
mise exec -- uv run --frozen --no-sync pytest -m workers -q
```

CIも同じsetup/checkをread-only権限で実行する。workflowの作成を、ホスト上のCI成功とはみなさない。公開ジョブは含めない。

## 変更とレビュー

`src/cosmea`、`tests`、`examples`、`experiments`、`scripts`の責務を分ける。検証・主要処理・状態更新などの間に意味のある空行を入れ、テスト・ツールにも同じ可読性規約を適用する。docstringにはshape、失敗条件、数値上の工夫、副作用を記述する。着想元・開発経緯は人間向け文書へ置く。

不具合修正には振る舞いに基づく回帰検証と、関連境界・呼出し経路の確認を加える。可能なら独立した数式oracleと比較する。stationarity検証を証明と呼ばず、合成学習を有効性の証拠にしない。コード・依存・図・文書・配布物を横断してレビューする。性能を主張するときは比較する両版、入力、環境、ウォームアップ、反復、集計、別入力での悪化も記録する。

日英文書を同時に更新し、公開ファイルへ個人パス・非公開参照・秘密情報を含めない。`scripts/check_public.py` はリンク・明らかなローカルパス・配布物を検査するが、一般的な秘密検出器ではない。必要な第三者表示を保持する。非公開記録をGit・配布物から除外し、歴史的snapshotを変更しない。

## ビルドと今後の公開

`mise run build` はdistにwheelとsdistを1つずつ作り、`mise run audit` が両方を検査する。版を変える場合は古いビルド成果物を整理する。第三者は非公開入力なしでsdistからビルドできる。将来の公開には、管理者によるライセンス・公開先の決定、Git／公開操作の許可、ホストCIの成功、公開履歴・作成者情報の確認、成果物checksumの記録が必要。公開は今回の開発作業の範囲外。

## 秘密情報の検査とコード例の改行

`mise run secrets`はビルド後に[Betterleaks](https://github.com/betterleaks/betterleaks/tree/v1.10.1)（MIT）を実行する。版は`mise.toml`で固定し、公開予定ファイル、展開したwheel／sdist、commitがあればローカルの全refのGit履歴を検査する。CIは全履歴をcheckoutし、ローカルのshallow cloneは不足履歴を取得する。初回commit前やsource archiveでは履歴がないことを明示する。固定版の組込み規則を使い、providerへの認証情報検証は無効にする。検出・実行エラーはいずれも検査失敗となる。マスク済みJSONレポートはGit除外された`build/security/`に保存し、包括的なallowlistやbaselineは使わない。パターン検出であり、秘密情報が一切ないことの保証ではない。

複数行の関数呼出しは1行1引数と末尾カンマを基本にする。公開コード例にも同じ規約を適用し、短く単純な呼出しは読みやすければ1行に収める。長いimportも項目ごとに改行する。

## オフライン学習例と任意の実験

`mise run setup`の後、`mise run smoke`で通常版・Cached版の学習例を実行できます。小さなBERTと合成データをローカルで作るため、学習済みモデルのダウンロードは不要です。`mise run check`にもこの検証を含めています。

時間をかけて合成データ上の設定を比較する場合は、次を実行します。

```bash
mise exec -- uv run --frozen --no-sync python experiments/toy_two_tower.py --seeds 3 --out results.json
```

これは通常の検査とは別の任意実験です。出力には引数とseedごとの結果を保存します。比較に使う場合はソースの版と環境も一緒に記録してください。この改訂では既定設定の実験全体を再実行していません。指標の定義は[設計の説明](docs/ja/design.md)を参照してください。

コードの置き場は、ライブラリ本体が`src/cosmea`、利用例が`examples`、検証が`tests`、任意実験が`experiments`、開発用の検査が`scripts`です。
