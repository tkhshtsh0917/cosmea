# Cosmea — sampling-aware retrieval training

[English](README.md)

**Cosmea（コスメア）** は、Sentence Transformersへ **logQ補正とMixed Negative Sampling（MNS）** を追加するPythonライブラリです。学習行を選ぶサンプラーは既定のランダム方式を使い、コレーターが一様コーパス負例を追加し、損失がサンプリング確率を補正します。

名前はコスモスの異名 *Cosmea* と、花言葉の「調和」に由来します。

> 開発版 `0.1.0.dev0`。2026-11-18時点でCPU上の数式・統合テストを確認。実データでの有効性は未検証です。logQは負例抽出の偏りを補正するもので、人気集中の緩和や因果的公平性を保証する機能ではありません。

## インストール

お使いのPythonプロジェクトで、[uv](https://docs.astral.sh/uv/)を使ってGitHubからインストールできます。

```bash
uv add "cosmea[train] @ git+https://github.com/tkhshtsh0917/cosmea"
```

`[train]`を付けると、`SentenceTransformerTrainer`で学習するための依存ライブラリも入ります。コレーターや損失関数だけを使う場合は、`[train]`を外してインストールできます。検証にはPython 3.12を使っています。対応する依存ライブラリの範囲は[APIガイド](docs/ja/api.md)を参照してください。

uvは取得したGitコミットをお使いのプロジェクトの`uv.lock`へ記録します。特定の版を使いたい場合は、上のコマンドに`--rev`とコミットSHAを追加してください。

開発への参加、オフラインの学習例、ビルドの手順は[CONTRIBUTING.ja.md](CONTRIBUTING.ja.md)にまとめています。

## 手元のデータで使う

手元のデータを使う場合は、[CSV学習例](examples/README.ja.md#手元のデータで学習する)や[難例採掘から再ランキングまでの例](examples/README.ja.md#難例採掘から再ランキングまで)を参照してください。

`train_dataset`には`anchor, positive, query_id, document_id`の順に列を持つ学習データを用意します。`anchor`はクエリのテキスト、`positive`は対応する文書のテキストです。`corpus_texts`は検索対象のテキスト一覧、`corpus_ids`は同じ順番のID一覧です。IDの付け方と追加の負例列は[APIガイド](docs/ja/api.md)で説明しています。

```python
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.base.sampler import BatchSamplers
from cosmea import (
    DocumentFrequency,
    MixedNegativeDataCollator,
    LogQMultipleNegativesRankingLoss,
)

model = SentenceTransformer("path/to/local/model", device="cpu")
# train_dataset の列順: anchor, positive, query_id, document_id
# corpus_texts と corpus_ids は同じ順番で、同じID体系を使う。
frequency = DocumentFrequency.from_dataset(train_dataset, document_id_column="document_id")

collator = MixedNegativeDataCollator(
    preprocess_fn=model.preprocess,
    document_frequency=frequency,
    corpus=corpus_texts,
    corpus_ids=corpus_ids,
    num_uniform=64,  # コーパス内の異なるIDの数を超えないようにする。
    query_id_column="query_id",
    document_id_column="document_id",
)

loss = LogQMultipleNegativesRankingLoss(
    model,
    variant="plugin",
    combine="pooled",
    logq_strength=1.0,
    mask_same_document=True,
    mask_same_query=True,
)

args = SentenceTransformerTrainingArguments(
    output_dir="outputs/run",
    per_device_train_batch_size=64,
    batch_sampler=BatchSamplers.BATCH_SAMPLER,
    use_cpu=True,
    report_to="none",
    save_strategy="no",
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=loss,
    data_collator=collator,
)

trainer.train()
```

prompt・Routerのtask・長さ制限が必要なら、`prompts`・`router_mapping`・`max_length`を**コレーターへ**渡してください。TrainingArgumentsだけでは独自コレーターへ反映されません。[前処理の例](docs/ja/api.md#前処理を設定する)を参照してください。

未知データの評価には、同じコレーターで`eval_dataset`を処理する代わりに検索用`evaluator`を使います。`eval_dataset`に未知の正例IDがあるとエラーになります。[評価ガイド](docs/ja/api.md#評価と再現性)を参照してください。

標準手順は、上の例のようにランダムな行抽出と2つの既知正例マスクを組み合わせます。頻度は同じ学習splitから作ります。アンカーの正例と、同じクエリに紐づくバッチ内の別の正例を、負例から除外します。一様負例・採掘負例に同じ文書IDがある場合も対象です。

例では`plugin`を使い、既知の正例を負例の和から外しても、正例そのもののスコアは補正しません。APIの既定は引き続き`yi`です。yiに文書マスクを掛けると、単純な負例抽出の例でも学習する分布が変わることがあります。pluginにも小さいバッチでの限界があります。マスクで有効な負例が減る場合は、未知データでの検索指標を見て`num_uniform`を調整してください。[設計の説明](docs/ja/design.md#正例を負例から除くと何が変わるか)と[別のサンプラーを使う手順](docs/ja/api.md#バッチサンプラーを選ぶ)も参照できます。

通常版・Cached版のどちらも、[MatryoshkaLoss](docs/ja/api.md#複数の埋め込み次元で学習する)で包み、複数の埋め込み次元をまとめて学習できます。

## 提供する部品

| API | 役割 |
| --- | --- |
| `DocumentFrequency` | 正例IDの静的カウント。ID・頻度はint64 |
| `MixedNegativeDataCollator` | K個のコーパス負例を非復元抽出し、ID・頻度・情報源をdictラベルで損失へ渡す |
| `LogQMultipleNegativesRankingLoss` | 通常のforward/backward |
| `CachedLogQMultipleNegativesRankingLoss` | STのGradCacheを使い、クエリ行をチャンク計算 |
| `mixed_negative_logq_loss` | 温度スケール済み `(b, C)` スコアから行別損失を計算 |

| 損失 | 正例・負例の扱い | 留意点 |
| --- | --- | --- |
| `yi`（既定） | 正例も補正。`auto`では情報源ごとの損失を平均 | iid提案分布による近似。実際の非復元バッチに有限バッチ一致性を保証しない |
| `plugin` | 正例は無補正、負例は期待出現回数／包含確率で補正 | 分配関数推定が不偏でも損失・勾配は有限標本で偏る |
| `improved`（研究用） | pluginと同じ一階勾配を持つstop-gradient形 | 損失値は単調でなく、早期終了・最良checkpointの選択には使わない |

`num_uniform=0` でlogQのみ、`logq_strength=0` で補正なし、`combine="pooled"` で候補を一つにまとめます。**素のMNRLとの比較には `logq_strength=0, combine="pooled"`** とマスク条件の統一が必要です。`auto`のyiはseparateなので同じ目的関数にはなりません。

採掘負例はseparateで独立した無補正の損失にできます。pooledでは相対的な影響が小さくなる場合があります。

詳細は [API契約](docs/ja/api.md)、[設計と互換性](docs/ja/design.md)、[開発・検証手順](CONTRIBUTING.ja.md) を参照してください。

## 検証と範囲

抽出確率の計算、損失の勾配、Sentence Transformersとの接続を自動検査しています。通常版・Cached版のCPU Trainer統合テストも[CI](https://github.com/tkhshtsh0917/cosmea/actions/workflows/ci.yml)で確認しています。これらは合成データによる検証で、実データでの改善は別途評価する必要があります。

- 検証環境：macOS arm64、Python 3.12.13、ST 6.3.0、PyTorch 2.15.0、Transformers 5.23.0。
- CPU、単一プロセス学習、有限のmap-style datasetが対象です。CUDA／MPS／DDP／cross-device gather／ストリーミング頻度／複数データセットは未対応または未検証です。
- `yi + inclusion` は無効です。包含確率方式を試す場合はpluginまたはimprovedを指定します。
- `mask_same_query=True`（既定）はバッチ内で見える既知正例だけを除くヒューリスティックです。不偏性を主張する検証ではFalseを使います。
- マイニング済み負例の列は無補正の追加項です。既知の提案分布からの標本として扱いません。
- 包含確率は正確に計算しますが大きなバッチでは高価です。通常は `expected_count` を使います。

## 関連研究

Cosmeaは次の研究を参考にしています。自分で作るバッチの抽出方法に合わせて計算を調整しており、論文の実験を再現したという主張ではありません。

- [Yi et al., 2019 — Sampling-Bias-Corrected Neural Modeling](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/)
- [Yang et al., 2020 — Mixed Negative Sampling](https://research.google/pubs/mixed-negative-sampling-for-learning-two-tower-neural-networks-in-recommendations/)
- [Ma & Collins, 2018 — Conditional NCE](https://arxiv.org/abs/1809.01812)
- [Khrylchenko et al., 2025 — Correcting the LogQ Correction](https://arxiv.org/abs/2507.09331)／[著者コード](https://github.com/NonameUntitled/logq/tree/f48221cad05612a2558966288e77020e61f3ff01)

## ライセンス

Cosmeaは[Apache-2.0](LICENSE)で提供します。依存ライブラリにはそれぞれのライセンスが適用されます。第三者資料の扱いは[NOTICE](NOTICE.md)を参照してください。

パッケージの公開リリースはまだ行っていません。ホスト上の検証結果は[GitHub Actions](https://github.com/tkhshtsh0917/cosmea/actions/workflows/ci.yml)で確認できます。
