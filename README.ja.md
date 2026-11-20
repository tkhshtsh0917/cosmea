# Cosmea

[English](README.md)

クリックが一部の文書に集中すると、それらの文書は学習バッチで負例として現れやすくなります。検索対象に含まれていても、クリックされていない文書はバッチに一度も現れないことがあります。Cosmea（コスメア）は、このようにフィードバックが偏ったデータで、Sentence Transformersの検索モデルをファインチューニングするためのPythonライブラリです。

特に、検索対象の一部に学習の正例が集中している場合に向いています。正例として現れる回数が文書間でほぼ同じなら、それらの文書の間でlogQが補正する頻度の偏りは小さくなります。検索対象全体から抽出すれば、学習ペアに含まれない文書も比較候補に加えられます。

## 標準MNRLとの違い

クエリと正例文書のペアで学習するとき、標準のMultiple Negatives Ranking Loss（MNRL）は、同じバッチの別の行にある正例文書を負例に使います。Sentence Transformersは、この使い方で `NO_DUPLICATES` を推奨しています。詳細は[Sentence Transformersの損失リファレンス](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#sentence_transformers.sentence_transformer.losses.MultipleNegativesRankingLoss)を参照してください。

Cosmeaの推奨構成では、学習行をランダムに抽出します。**Mixed Negative Sampling（MNS）** で検索対象全体から一様に抽出した文書をバッチ内の候補に加え、**logQ補正** で各候補が選ばれやすい度合いに応じて損失を調整します。同じクエリの既知の正例がバッチに含まれる場合は、マスクで負例から除きます。詳しくは[バッチサンプラーの選び方](docs/ja/api.md#バッチサンプラーを選ぶ)を参照してください。

既存のTrainerスクリプトで用意した `model`、学習引数 `args`、`train_dataset` を使います。この例の `train_dataset` は、`anchor`、`positive` の順に並ぶテキスト2列だけで構成します。検索対象は、異なる文書テキストを4件以上含む `corpus_texts` として渡します。同じ文書には、学習ペアと検索対象で同じテキストを使ってください。

```python
from sentence_transformers.base.sampler import BatchSamplers

from cosmea import DocumentFrequency, LogQMultipleNegativesRankingLoss, MixedNegativeDataCollator

args.batch_sampler = BatchSamplers.BATCH_SAMPLER

collator = MixedNegativeDataCollator(
    preprocess_fn=model.preprocess,
    document_frequency=DocumentFrequency.from_dataset(train_dataset),
    corpus=corpus_texts,
    num_uniform=4,
)

loss = LogQMultipleNegativesRankingLoss(model)
```

通常版・Cached版ともに、既定では `plugin`、`pooled`、2つの既知正例マスクを使います。従来の既定設定で学習する場合は `variant="yi"` を指定してください。Trainerを作成するときに `data_collator=collator` と `loss=loss` を渡します。モデルにプロンプトが必要な場合は、コレーターの `prompts` に設定してください。Trainer全体の設定、明示的なIDの使い方、`NO_DUPLICATES` と併用する場合の手順は[組み込みガイド](docs/ja/api.md)を参照してください。

## SciFactでの結果

CPUで同じ `all-MiniLM-L6-v2` チェックポイントを使い、学習から取り分けた同じ300件のクエリで全条件を評価しました。各回の学習ペアは256組で、手法間で揃えています。nDCG@10は関連文書が上位10件の中で上にあるほど高くなります。

| nDCG@10 | 初期モデル | 標準MNRL | Cosmea |
| --- | ---: | ---: | ---: |
| 学習ペアを固定し、学習シードを変更 | 0.6239 | 0.6290 | 0.6334 |
| 学習シードを固定し、学習ペアを変更 | 0.6239 | 0.6279 | 0.6327 |

学習後のスコアは各行5回の平均です。学習ペアを変えた場合、標準MNRLに対するCosmeaの差は −0.0077〜+0.0170 でした。時間を繰り返し計測した条件では、学習時間の中央値はCosmeaが8.82秒、標準MNRLが4.89秒でした。各バッチに一様抽出した文書を16件追加するこの条件では約1.8倍です。分割方法、計測条件、logQを無効にした比較は[SciFactベンチマーク](docs/ja/benchmark.md)を参照してください。

## CPU・GPUの検証状況

| 学習環境 | プロジェクトでの検証状況 |
| --- | --- |
| CPU・学習プロセス1つ | 組み込みテストで確認済み。クイックスタートとCSVの利用例は既定でCPUを使用。 |
| Apple MPS・GPU 1台 | pooled `plugin` 損失を使うFP32のCSV学習、モデルの保存と再読み込みを確認済み。[CSVの利用例](examples/README.ja.md#デバイスを選ぶ)を参照。 |
| CUDA・GPU 1台 | CSVの利用例で `--device cuda` を指定可能。学習全体の動作は未検証。 |
| 分散学習・混合精度学習 | 学習全体の動作は未検証。 |

依存ライブラリの対応バージョンや必要なメモリについては[設計ガイド](docs/ja/design.md#資源と互換性の制限)を参照してください。

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

CosmeaはApache-2.0の開発版です。名前はコスモスの異名 *Cosmea* に由来します。ライセンスと帰属表示は [LICENSE](LICENSE) と [NOTICE](NOTICE.md) を参照してください。
