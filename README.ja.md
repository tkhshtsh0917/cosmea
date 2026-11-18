# Cosmea

[English](README.md)

Cosmea（コスメア）は、Sentence Transformersで検索モデルをファインチューニングするためのPythonライブラリです。**Mixed Negative Sampling（MNS）** で、クリック履歴のない文書も含め、検索対象の文書全体から学習用の比較候補を追加します。**logQ補正** では、文書が学習候補に選ばれやすい度合いに応じて、損失の計算を調整します。

Apache-2.0の開発版です。利用例はCPUで動作します。対応範囲は[設計ガイド](docs/ja/design.md#資源と互換性の制限)を参照してください。損失の計算と学習への接続はテストで検証しており、実データでの検索品質への効果は未検証です。

## クイックスタート

小さな英語モデルを3組のサンプルで学習し、保存したモデルで6件の文書を検索します。[uv](https://docs.astral.sh/uv/)、Git、curlを用意し、Bashなどの互換シェルで実行してください。最初の実行時にHugging Faceから `sentence-transformers/all-MiniLM-L6-v2` をダウンロードします。Linuxでは学習をCPUで行う場合も、既定のPyTorchが数GBのGPU用ライブラリを取得することがあります。

```bash
uv init --python 3.12 cosmea-quickstart
cd cosmea-quickstart

uv add "cosmea[train] @ git+https://github.com/tkhshtsh0917/cosmea"
curl --fail --location --output quickstart.py https://raw.githubusercontent.com/tkhshtsh0917/cosmea/main/examples/quickstart.py

uv run python quickstart.py
```

出力の `Saved model:` と `Top result:` を確認してください。スクリプトは `outputs/quickstart/model` にモデルを保存し、読み込み直して、自転車のタイヤに関する質問の検索結果を1件表示します。学習ペアと検索対象の文書は [quickstart.py](examples/quickstart.py) で確認できます。

学習に使っていない2件のクエリと同じ文書集合で、学習前後の `nDCG@3` と `Recall@3` を表示します。nDCGは関連文書が上位にあるほど高くなり、Recallは既知の関連文書のうち上位3件に入った割合です。この例の学習は1ステップなので、指標が同じでも正常です。実際の品質を測るときは、学習に使っていないクエリを用意し、それぞれの正解文書を指定して評価してください。

既にローカルにあるSentence Transformersモデルを使う場合は、スクリプトに `--model <ディレクトリ>` を渡します。既存のPythonプロジェクトへは、そのディレクトリでインストールコマンドを実行します。`train` extraでTrainerの依存関係も入ります。uvはGitのリビジョンを `uv.lock` に記録します。`uv add --rev <コミットSHA>` で版を固定する場合は、curlのURLの `main` も同じSHAに置き換えてください。

## 手元のデータで使う

- [CSVから学習する](examples/README.ja.md#手元のデータで学習する)では、クエリと文書のペア、文書ID付きの検索対象を用意します。
- [Sentence Transformersに組み込む](docs/ja/api.md)では、学習データの作成、コレーターと損失の設定、検索品質の評価を説明します。
- [難例を採掘して再ランキングする](examples/README.ja.md#難例採掘から再ランキングまで)では、難しい負例も使って学習し、学習済みモデルの上位候補をCrossEncoderで並べ替えた効果を比較します。

## ドキュメント

- [公開API](docs/reference.md)から、クラスと関数のソースやAPIリファレンスの生成手順を参照できます。
- [設計ガイド](docs/ja/design.md)では、抽出と補正の前提、損失の種類、対応範囲を説明します。
- [貢献ガイド](CONTRIBUTING.ja.md)では、開発環境、検証、文書サイトのビルドを案内します。

名前はコスモスの異名 *Cosmea* に由来します。ライセンスと帰属表示は [LICENSE](LICENSE) と [NOTICE](NOTICE.md) を参照してください。開発版はGitHubからインストールできます。
