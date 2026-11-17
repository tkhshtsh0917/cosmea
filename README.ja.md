# Cosmea — sampling-aware retrieval training

[English](README.md)

**Cosmea（コスメア）** は、Sentence Transformersへ **logQ補正とMixed Negative Sampling（MNS）** を追加するPythonライブラリです。学習行を選ぶサンプラーは既定のランダム方式を使い、コレーターが一様コーパス負例を追加し、損失がサンプリング確率を補正します。

名前はコスモスの異名 *Cosmea* と、赤いコスモスの花言葉「調和」に由来します。異なる負例分布を組み合わせ、補正との整合を取る設計を示唆しています。[名称：Kew](https://powo.science.kew.org/taxon/8407-1)／[花言葉：千葉県](https://www.pref.chiba.lg.jp/ryuhan/pbmgm/zukan/kaki/cosmos.html)

> 開発版 `0.1.0.dev0`。2026-11-17時点でCPU上の数式・統合テストを確認。実データでの有効性は未検証です。logQは負例抽出の偏りを補正するもので、人気集中の緩和や因果的公平性を保証する機能ではありません。

## インストールと最小実行

Python 3.12で検証しています。公開パッケージのインストールではなく、このcheckoutを使います。

```bash
# checkoutしたcosmeaディレクトリで実行
mise trust
mise install
mise run setup
mise run smoke
mise run check
```

開発ツールはmise、依存とPythonはuvで管理します。正本と個別検査は [開発手順](CONTRIBUTING.ja.md) を参照してください。通常利用には `pip install -e '.[train]'` も使えます。最小例はtiny BERTと合成データをローカル生成し、依存導入後のモデルダウンロードは不要です。

## 手元のデータで使う

```python
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from cosmea import (
    ItemFrequency,
    MixedNegativeDataCollator,
    LogQMultipleNegativesRankingLoss,
)

model = SentenceTransformer("path/to/local/model", device="cpu")
# train_dataset の列順: query, positive, query_id, item_id
# corpus_texts と corpus_ids は同じ順番で、同じID体系を使う。
frequency = ItemFrequency.from_dataset(train_dataset, item_id_column="item_id")

collator = MixedNegativeDataCollator(
    preprocess_fn=model.preprocess,
    item_frequency=frequency,
    corpus=corpus_texts,
    corpus_ids=corpus_ids,
    num_uniform=64,                  # distinct corpus IDsの数を超える指定は不可
    query_id_column="query_id",
    item_id_column="item_id",
)

loss = LogQMultipleNegativesRankingLoss(
    model,
    variant="yi",
    logq_strength=1.0,
)

args = SentenceTransformerTrainingArguments(
    output_dir="outputs/run",
    per_device_train_batch_size=64,
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

頻度は **そのDataLoaderがランダム抽出する学習split** から作ります。`NO_DUPLICATES`、重み付き・グループ化サンプラーを使うと、この頻度から計算したQは合いません。評価splitに同じコレーターを使い回して得る損失にも同じ問題があるため、最小例は評価データの全コーパスRecallを別に計算します。

## 提供する部品

| API | 役割 |
| --- | --- |
| `ItemFrequency` | 正例IDの静的カウント。ID・頻度はint64 |
| `MixedNegativeDataCollator` | K個のコーパス負例を非復元抽出し、ID・頻度・情報源をdictラベルで損失へ渡す |
| `LogQMultipleNegativesRankingLoss` | 通常のforward/backward |
| `CachedLogQMultipleNegativesRankingLoss` | STのGradCacheを使い、クエリ行をチャンク計算 |
| `mixed_negative_logq_loss` | 温度スケール済み `(b, C)` スコアから行別損失を計算 |

| 損失 | 正例・負例の扱い | 留意点 |
| --- | --- | --- |
| `yi`（既定） | 正例も補正。`auto`では情報源ごとの損失を平均 | iid提案分布による近似。実際の非復元バッチに有限バッチ一致性を保証しない |
| `plugin` | 正例は無補正、負例は期待出現回数／包含確率で補正 | 分配関数推定が不偏でも損失・勾配は有限標本で偏る |
| `improved` | stop-gradientした信頼度重みを掛ける | 2025論文を実際のサンプリング設計に適応。同じ設定のpluginと**一階勾配**が一致。論文・著者コードの逐語的再現ではない |

`num_uniform=0` でlogQのみ、`logq_strength=0` で補正なし、`combine="pooled"` で候補を一つにまとめます。**素のMNRLとの比較には `logq_strength=0, combine="pooled"`** とマスク条件の統一が必要です。`auto`のyiはseparateなので同じ目的関数にはなりません。

詳細は [API契約](docs/ja/api.md)、[設計と互換性](docs/ja/design.md)、[開発・検証手順](CONTRIBUTING.ja.md) を参照してください。

## 検証と範囲

53件のテストで、37件の通常・統合検証、7件の統計検証、1件の2ワーカー検証、5件の公開文書契約、3件の実験指標の回帰検証を確認しています。通常版とCached版のCPU学習例も4ステップ実行しました。小さな合成データの動作確認であり、性能比較ではありません。

- 検証環境：macOS arm64、Python 3.12.13、ST 6.3.0、PyTorch 2.15.0、Transformers 5.23.0。
- CPU、単一プロセス学習、有限のmap-style datasetが対象です。CUDA／MPS／DDP／cross-device gather／ストリーミング頻度／複数データセットは未対応または未検証です。
- `yi + inclusion` は無効です。包含確率方式を試す場合はpluginまたはimprovedを指定します。
- `mask_same_query=True`（既定）はバッチ内で見える既知正例だけを除くヒューリスティックです。不偏性を主張する検証ではFalseを使います。
- マイニング済み負例の列は無補正の追加項です。既知の提案分布からの標本として扱いません。
- 包含確率は正確に計算しますが大きなバッチでは高価です。通常は `expected_count` を使います。
- 元のプロトタイプで報告された性能数値は、本版の結果として転載していません。

## 着想元と出典

ユーザー提供のst-logqプロトタイプを出発点としてレビュー・再編しました。論文の再現結果ではなく、実際の標本設計に合わせた適応実装です。

- [Yi et al., 2019 — Sampling-Bias-Corrected Neural Modeling](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/)
- [Yang et al., 2020 — Mixed Negative Sampling](https://research.google/pubs/mixed-negative-sampling-for-learning-two-tower-neural-networks-in-recommendations/)
- [Ma & Collins, 2018 — Conditional NCE](https://arxiv.org/abs/1809.01812)
- [Khrylchenko et al., 2025 — Correcting the LogQ Correction](https://arxiv.org/abs/2507.09331)／[著者コード](https://github.com/NonameUntitled/logq/tree/f48221cad05612a2558966288e77020e61f3ff01)

ライセンス・出典の状態は [NOTICE](NOTICE.md)。公開リリースはまだ行っていません。CIは設定済みで、ホスト上の実行は未確認です。
