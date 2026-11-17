# APIガイド

[English](../api.md) · [README](../../README.ja.md)

まずは[READMEの利用例](../../README.ja.md#手元のデータで使う)を参照してください。このページでは、データの用意の仕方と設定の意味を説明します。

## 学習ペアと検索対象を用意する

**正例**は、そのクエリに合っていると分かっているitemです。たとえばクリックされた文書が当たります。**負例**は学習時に比較相手として使うitemで、必ずしも無関係とは限りません。**コーパス**は、検索対象となるitem全体を指します。

次の入力を用意します。

| 入力 | 用意するもの |
| --- | --- |
| 学習dataset | `query, positive, query_id, item_id`の順に並んだ列 |
| `corpus` | itemのテキスト一覧。クリック履歴のないitemも含める |
| `corpus_ids` | `corpus`と同じ順番のID一覧 |
| `ItemFrequency` | DataLoaderが実際に抽出する学習datasetから数えた頻度 |

コレーターは、複数の学習例をまとめてモデルへ渡す部品です。ID列を取り除いた後の最初の2つのテキスト列をqueryとpositiveとして読みます。学習に使わない補助列は取り除いてください。`label`、`labels`、`score`、`scores`は内部で使うため、datasetの列名にするとエラーになります。

同じitemには、どの入力でも同じIDを使ってください。IDには文字列、bytes、符号付き64bit整数を使えます。テキストIDは安定した64bit hashへ変換します。文字列の`"12"`と整数の`12`は別IDです。hash衝突の可能性があるため、大規模なコーパスには一貫した整数IDを勧めます。IDを省くと、テキストの完全一致で同一itemかどうかを判断します。

正例IDが`ItemFrequency`にない場合はエラーになります。コーパスに同じIDが複数あれば最初のテキストを残してまとめ、その後の重複しないID全体から一様に抽出します。

## 自分で用意した負例を追加する

別の検索器で見つけた難しい負例などを、追加のテキスト列として渡せます。これらが選ばれる確率は分からないため、追加列の項にはlogQ補正を掛けません。

正例に明示的なitem IDを使う場合は、追加するすべての負例列にもID列を対応付けます。

```python
collator = MixedNegativeDataCollator(
    preprocess_fn=model.preprocess,
    item_frequency=frequency,
    item_id_column="item_id",
    extra_item_id_columns={"hard_negative": "hard_negative_id"},
)
```

これにより、テキストが違っても同じitemだと分かります。必要な対応付けがない場合はエラーになります。

## 主な設定を選ぶ

既定値は比較を始めるための設定です。どのデータでも最も良いという意味ではありません。

| 設定 | 意味 |
| --- | --- |
| `num_uniform` | 各バッチへ追加する、重複しないコーパスitemの数。0ならバッチ内負例だけを使う。コーパスの重複しないID数を超えて指定できない |
| `variant` | 既定は`yi`。`plugin`や`improved`へ変える前に[損失の違い](design.md#3種類の損失)を確認する |
| `logq_strength` | 補正の強さ。0で補正なし、1で通常の補正。有限の非負値が必要で、中間値の効果は評価して選ぶ |
| `scale` | 補正の前に類似度へ掛ける倍率。既定20。有限の正の値が必要で、「温度」の逆数に当たる |
| `combine` | `pooled`は候補をまとめて計算。`separate`は負例の出所ごとに計算し、有効なものだけ平均。`auto`はyiでseparate、それ以外でpooledを選ぶ |
| `q_measure` | `expected_count`は同じitemの重複を残し、平均して何回現れるかを使う。`inclusion`は有効な候補をitemごとに1件にし、少なくとも1回現れる確率を使う。yiではinclusionを使えない |
| `mini_batch_size` | Cached版で一度に計算するクエリ行数。正の整数を指定する。小さくするとスコア行列のメモリを減らせる。勾配の蓄積だけでは候補数は増えない |

separateでは、使える負例がない出所を平均から外します。たとえば学習行が1件のバッチでも、一様負例の損失を空のバッチ内損失と平均して半分にしてしまうことはありません。

## 既知の正例を負例から外す

**マスク**は、その候補を損失の計算対象から外すことです。

- `mask_same_item=None`では、正例itemが偶然ほかの候補にも現れたとき、yiは残し、plugin/improvedは除きます。TrueやFalseで変更できますが、学習の目的関数も変わります。
- `mask_same_query=True`では、現在のバッチにある同じqueryの正例を利用します。コーパス全体の関連itemがすべて分かるわけではありません。この除外は実用上の工夫で、後述の厳密な不偏性の保証には含まれません。

yiで一様負例をseparateとして使う場合は、すべての正例IDがコーパスにも必要です。満たさなければエラーになります。

## 評価と再現性

学習行を一様にランダム抽出する既定サンプラーを使ってください。`NO_DUPLICATES`や、重み・グループに基づく抽出に変えると、同じ頻度による補正は抽出方法に合わなくなります。

頻度は実際に抽出する学習splitから数えます。学習に使っていないデータで品質を測る場合は、コーパス全体を検索して指標を計算してください。別の評価splitへ学習頻度を使い回すと、一般に補正する確率がずれます。Trainerの評価テストは、情報が損失まで届くことの確認であり、未知データでの品質評価ではありません。

DataLoaderの各workerはPyTorchの乱数を元に別々の乱数列を使います。再現には同じ起動条件が必要です。途中checkpointからの完全な乱数復元や、worker数を変えても同じ抽出順になることは保証していません。

対応範囲はSentence Transformers `>=6.3,<6.4`、PyTorch `>=2.15,<2.16`です。CPU・1学習プロセスで検証しています。GPU、分散学習、ストリーミングには追加の検証が必要です。[資源と互換性の制限](design.md#資源と互換性の制限)も参照してください。

## 複数の埋め込み次元で学習する

Cosmeaの通常版・Cached版のどちらも、Sentence Transformers 6.3の`MatryoshkaLoss`で包んで使えます。埋め込みの先頭部分を複数の長さで学習し、検索時に短い埋め込みを選べるようにする仕組みです。

以下は768次元のモデルを使う例です。自分のモデルの出力次元以下の正の値を指定し、元の長さでも使いたい場合はその次元も含めてください。

```python
from sentence_transformers.sentence_transformer.losses import MatryoshkaLoss
from cosmea import LogQMultipleNegativesRankingLoss

base_loss = LogQMultipleNegativesRankingLoss(model, variant="yi")
loss = MatryoshkaLoss(
    model=model,
    loss=base_loss,
    matryoshka_dims=[768, 384, 192],
)
```

この`loss`をTrainerへ渡し、`MixedNegativeDataCollator`はそのまま使います。勾配キャッシュを使う場合は、内側の損失を`CachedLogQMultipleNegativesRankingLoss`に置き換えてください。Cached用のwrapperは元の損失のメソッドを変更するため、wrapperごとに新しいbase lossを作ります。

抽出情報は埋め込みの長さに依存しないので、どの次元でも同じIDと頻度を使えます。重みを$w_m$とすると、全体の損失は$\sum_m w_m\ell_m$です。重みの既定値は1で、自動的に平均するわけではありません。

CPU検証では32・16・8次元で、通常版・Cached版の結果を、次元別に明示計算した損失と比較しています。異なる重みも使い、yiのexpected-count、plugin/improvedのexpected-countとinclusionを確認しました。pluginではTrainerによる学習・評価も確認しています。これは接続と勾配の検証で、短くした後の検索品質を保証するものではありません。ランダムな次元選択（`n_dims_per_step`）、GPU、Matryoshka2dLossはここでは未検証です。

## 詳細：損失へ渡す情報

通常はコレーターに任せて構いません。スコアを直接渡す関数を使う場合の仕様です。

学習行数を$B$、追加の負例列数を$H$、一様負例数を$K$とすると、候補数は$C=B(1+H)+K$です。正例、追加列ごとの候補、一様負例の順に並びます。以下のshapeはテンソルの各次元の大きさです。

| キー | shape / 型 | 意味 |
| --- | --- | --- |
| `query_id` | $(B,)$ / int64 | queryのID |
| `cand_id` | $(C,)$ / int64 | 候補itemのID |
| `cand_count` | $(C,)$ / int64 | 各itemが正例になっている学習行数 |
| `cand_source` | $(C,)$ / int64 | 0=バッチ内、1=一様、2=追加列 |
| `cand_in_corpus` | $(C,)$ / bool | コーパスから抽出できるitemかどうか |
| `num_rows` / `corpus_size` | scalar / int64 | 全学習行数$N$ / コーパスの重複しないitem数$M$ |

これらは`batch["label"]`で損失へ渡し、モデルの入力特徴には混ぜません。損失クラスは平均した1つの値を返します。`mixed_negative_logq_loss`はクエリ行ごとの値をshape $(b,)$で返します。分割計算時の`rows`は、元のバッチの行番号であり、その行の正例の候補列番号でもあります。

`candidate_log_q(..., rows=...)`はshape $(b,C)$で、正例の行が既に選ばれていることを考慮します。`rows`を省くと$(C,)$になり、その1回分を引きません。`iid_proposal=True`はyi用のcount/$N$近似で、コレーターの抽出方法を変える指定ではありません。
