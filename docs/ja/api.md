# APIとデータ契約

[English](../api.md)

## 入力の列とID

`MixedNegativeDataCollator` はID列を取り除いた後の最初の2列をquery、positiveとして扱う。列順を明示し、数値の業務メタデータをテキスト列に混ぜない。`label`、`labels`、`score`、`scores` はサンプリングメタデータと衝突するため受け付けない。

IDはstring／bytes／signed int64。整数はそのまま、テキストはblake2b 64bitへ変換する。文字列 `"12"` と整数 `12` は別ID。hash衝突の絶対保証はないため、大規模運用では一貫したint64辞書を推奨する。IDがない場合はテキストの完全一致を同一itemとして扱い、同じitemの別表記は統合しない。

頻度とコーパスに同じID体系を使う。未知の正例は例外にする。コーパスの重複IDは最初のテキストを代表として統合し、distinct IDに対して一様に抽出する。同じIDのテキストが複数ある場合は、この選択が用途に合うか確認する。

追加の負例列と明示item IDを併用する場合：

```python
collator = MixedNegativeDataCollator(
    preprocess_fn=model.preprocess,
    item_frequency=frequency,
    item_id_column="item_id",
    extra_item_id_columns={"hard_negative": "hard_negative_id"},
)
```

この対応を省くとテキストhashとitem IDが混ざり、同一itemマスクが漏れるため、明示ID付き追加列は対応を必須にした。

## ラベルのshape

B＝学習行数、H＝追加候補列数、K＝一様負例数、C＝B(1+H)+K。候補順はpositive B件、追加列ごとのB件、一様K件。

| キー | shape / dtype | 意味 |
| --- | --- | --- |
| `query_id` | `(B,)` / int64 | クエリの同一性 |
| `cand_id` | `(C,)` / int64 | 候補itemの同一性 |
| `cand_count` | `(C,)` / int64 | 抽出元datasetの正例回数 |
| `cand_source` | `(C,)` / int64 | 0=in-batch、1=uniform、2=other |
| `cand_in_corpus` | `(C,)` / bool | 一様提案分布のsupportに含まれるか |
| `num_rows` / `corpus_size` | scalar / int64 | N / distinct M |

`batch["label"]` にdictとして入り、Trainerから損失へ渡る。モデル入力には混ぜない。通常損失は平均scalar、functional APIは行別 `(b,)` を返す。チャンク処理では `rows` が元のB行へのindexで、正例の列番号もこのindexになる。

## オプションの意味

- `scale`：正の有限値。既定20。`scale * similarity - lambda * log q` の順序を変えない。
- `logq_strength`：有限の非負値。1が基本。0と1の中間値の効果は評価して選ぶ。人気露出の単調制御とはみなさない。
- `combine="auto"`：yiはseparate、plugin/improvedはpooled。
- `separate`：その行で有効負例がある情報源だけを平均する。B=1で空のin-batch項が追加され、損失が半分になる問題を防ぐ。
- `q_measure="expected_count"`：候補の重複を残す。plugin/improvedでは行を固定したN−1母集団の期待回数を使う。
- `q_measure="inclusion"`：行ごとのマスク後にID重複を除き、超幾何分布による包含確率で割る。yiとの組合せは拒否する。
- `mask_same_item=None`：yiは偶発ヒットを残す。plugin/improvedは除く。明示変更は可能だが目的関数と理論条件も変わる。
- `mask_same_query=True`：バッチ内で同一queryに付いた他の正例をマスクする。完全な関連item集合ではない。query依存のフィルタでQの分布も変わり、厳密な不偏性の対象外になる。
- `mini_batch_size`：Cached版の正の整数チャンクサイズ。勾配蓄積は負例プールそのものを増やさない。

`candidate_log_q(..., rows=...)` は `(b,C)`、rowsなしは `(C,)`。前者は自己行の1回を引いた条件付き計算。後者を正例IDにそのまま使う場合はこの違いに注意する。`iid_proposal=True` はyi用のcount/N近似であり、実際の非復元サンプラーをiidへ変える指定ではない。

## 再現性と評価

ワーカーはプロセスごとにPyTorch RNGから別の乱数streamを初期化する。同じ起動条件での再現は意図しているが、途中checkpointからのコレーターRNGの完全復元、ワーカー数変更に対する同一系列は保証しない。

評価データのin-batch proposalは学習splitの頻度とは一般に違う。held-outの指標は全コーパス検索で測定する。`Trainer.evaluate()` のdictラベル受渡しは同一データによるsmokeで確認しているが、これをheld-out性能と混同しない。

理論上の前提と証明の範囲は [設計ノート](design.md) を参照。
