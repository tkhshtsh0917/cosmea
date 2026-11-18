# Public API / 公開API

Use this index to find the public classes and functions in the source. The documentation site appends the reference generated from their docstrings. To render it locally, follow the [documentation build instructions](../CONTRIBUTING.md#documentation-site).

公開クラスと関数の実装は下の一覧から参照できます。文書サイトでは、この一覧に続けてdocstringから生成したリファレンスを表示します。ローカルでの生成手順は[貢献ガイド](../CONTRIBUTING.ja.md#ドキュメントサイト)を参照してください。

[Integration guide](api.md) · [組み込みガイド](ja/api.md)

| Public API | Purpose | 用途 |
| --- | --- | --- |
| [`DocumentFrequency`](https://github.com/tkhshtsh0917/cosmea/blob/main/src/cosmea/frequency.py) | Count document IDs in the training positives | 学習データの正例文書IDを数える |
| [`MixedNegativeDataCollator`](https://github.com/tkhshtsh0917/cosmea/blob/main/src/cosmea/collator.py) | Add uniform corpus negatives and sampling metadata | 一様負例と抽出情報をバッチに加える |
| [`LogQMultipleNegativesRankingLoss`](https://github.com/tkhshtsh0917/cosmea/blob/main/src/cosmea/losses.py) | Train with logQ correction | logQ補正を使って学習する |
| [`CachedLogQMultipleNegativesRankingLoss`](https://github.com/tkhshtsh0917/cosmea/blob/main/src/cosmea/losses.py) | Train with logQ correction and gradient caching | 勾配キャッシュを使ってlogQ補正付きで学習する |
| [`stable_id`](https://github.com/tkhshtsh0917/cosmea/blob/main/src/cosmea/frequency.py) | Convert an external ID to a stable integer | 外部IDを安定した整数に変換する |
| [`mixed_negative_logq_loss`](https://github.com/tkhshtsh0917/cosmea/blob/main/src/cosmea/functional.py) | Compute per-query losses from scaled scores | スケール済みスコアからクエリごとの損失を求める |
| [`candidate_log_q`](https://github.com/tkhshtsh0917/cosmea/blob/main/src/cosmea/functional.py) | Inspect candidate weights in log space | 候補の重みを対数で確認する |
| [`negative_log_partition`](https://github.com/tkhshtsh0917/cosmea/blob/main/src/cosmea/functional.py) | Compute the corrected log sum of negative scores | 負例の指数スコアを補正つきで足し、その対数を求める |

## Compatibility / 互換性

`ItemFrequency` aliases `DocumentFrequency`. For legacy `item_*` and `mask_same_item` keywords, see the [input guide](api.md#prepare-pairs-and-a-searchable-collection).

`ItemFrequency` は `DocumentFrequency` の別名です。旧キーワード `item_*` と `mask_same_item` の対応は[入力の説明](ja/api.md#学習ペアと検索対象を用意する)を参照してください。
