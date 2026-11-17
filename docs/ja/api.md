# APIガイド

[English](../api.md) · [README](../../README.ja.md)

まずは[READMEの利用例](../../README.ja.md#手元のデータで使う)を参照してください。このページでは、データの用意の仕方と設定の意味を説明します。

## 学習ペアと検索対象を用意する

**クエリ**は検索への入力で、**文書**は検索の対象です。学習行ではクエリが**アンカー**になります。**正例**は、そのクエリに合っていると分かっている文書です。たとえばクリックされた文書が当たります。**負例**は学習時に比較相手として使う文書で、必ずしも無関係とは限りません。**コーパス**は、検索対象となる文書全体を指します。

次の入力を用意します。

| 入力 | 用意するもの |
| --- | --- |
| 学習dataset | `anchor, positive, query_id, document_id`の順に並んだ列 |
| `corpus` | 文書のテキスト一覧。クリック履歴のない文書も含める |
| `corpus_ids` | `corpus`と同じ順番のID一覧 |
| `DocumentFrequency` | DataLoaderが実際に抽出する学習datasetから数えた頻度 |

コレーターは、複数の学習例をまとめてモデルへ渡す部品です。ID列を取り除いた後の最初の2つのテキスト列をアンカーと正例として読み、後続のテキスト列を負例として扱います。列名はデータセットに合わせられ、学習時の役割は列の順序で決まります。学習に使わない補助列は取り除いてください。`label`、`labels`、`score`、`scores`は内部で使うため、datasetの列名にするとエラーになります。

同じ文書には、どの入力でも同じIDを使ってください。IDには文字列、bytes、符号付き64bit整数を使えます。テキストIDは安定した64bit hashへ変換します。文字列の`"12"`と整数の`12`は別IDです。hash衝突の可能性があるため、大規模なコーパスには一貫した整数IDを勧めます。IDを省くと、テキストの完全一致で同一文書かどうかを判断します。

正例IDが`DocumentFrequency`にない場合はエラーになります。コーパスに同じIDが複数あれば最初のテキストを残してまとめ、その後の重複しないID全体から一様に抽出します。

既存のコードでは、`ItemFrequency`と引数名の`item_frequency`、`item_id_column`、`extra_item_id_columns`、`mask_same_item`を引き続き使えます。新しいコードでは、`DocumentFrequency`、`document_frequency`、`document_id_column`、`extra_document_id_columns`、`mask_same_document`を使います。

## 前処理を設定する

`prompts`・`router_mapping`・`max_length`は、`MixedNegativeDataCollator`へ直接渡してください。Sentence TransformersはTrainingArgumentsのこれらの設定を、独自コレーターへコピーしません。Trainer側だけに指定すると、必要な前処理が黙って省略される場合があります。

以下は`query`と`document`のタスクを持つRouterモデルの設定例です。対応表のキーはデータセットの列名、値はRouterのタスク名です。promptとtask名は使うモデルに合わせます。通常のモデルでは`router_mapping`を省き、`max_length`を整数にしてください。

```python
collator = MixedNegativeDataCollator(
    preprocess_fn=model.preprocess,
    document_frequency=frequency,
    corpus=corpus_texts,
    corpus_ids=corpus_ids,
    num_uniform=64,
    query_id_column="query_id",
    document_id_column="document_id",
    prompts={"anchor": "query: ", "positive": "passage: "},
    router_mapping={"anchor": "query", "positive": "document"},
    max_length={"query": 64, "document": 256},
)
```

一様負例には、正例列と同じprompt・task・長さ制限を適用します。追加の採掘負例列にも文書用の前処理が必要なら、明示的に設定してください。評価・推論時のpromptも学習時に揃えます。Router設定の転送はテストしていますが、Routerモデル全体の学習は未検証です。

## 自分で用意した負例を追加する

別の検索器で見つけた難しい負例などを、追加のテキスト列として渡せます。これらが選ばれる確率は分からないため、追加列の項にはlogQ補正を掛けません。

正例に明示的な文書 IDを使う場合は、追加するすべての負例列にもID列を対応付けます。

```python
collator = MixedNegativeDataCollator(
    preprocess_fn=model.preprocess,
    document_frequency=frequency,
    document_id_column="document_id",
    extra_document_id_columns={"hard_negative": "hard_negative_id"},
)
```

これにより、テキストが違っても同じ文書だと分かります。必要な対応付けがない場合はエラーになります。

採掘負例を独立した無補正の目的関数として扱うには、`combine="separate"`を使います（yiの既定）。pooledでは、学習母集団が大きくなるにつれて補正済み負例より影響が小さくなる場合があります。[採掘負例と損失の監視](design.md#採掘負例と損失の監視)も参照してください。

## 主な設定を選ぶ

READMEの標準手順に従い、ランダムな行抽出と`mask_same_document=True`、`mask_same_query=True`を使います。例の損失はyi/separate・expected_countです。同じ抽出方法とマスク条件で、plugin/pooledも比較できます。`improved`・inclusionは比較研究向けです。`improved`の評価損失でモデルを選ばないでください。

| 設定 | 意味 |
| --- | --- |
| `num_uniform` | 各バッチへ追加する、重複しないコーパス文書の数。0ならバッチ内負例だけを使う。コーパスの重複しないID数を超えて指定できない |
| `variant` | 既定は`yi`。`plugin`や`improved`へ変える前に[損失の違い](design.md#3種類の損失)を確認する |
| `logq_strength` | 補正の強さ。0で補正なし、1で通常の補正。有限の非負値が必要で、中間値の効果は評価して選ぶ |
| `in_batch_logq_strength` / `uniform_logq_strength` | `combine="separate"`で負例の出所ごとに補正の強さを指定する。有限の非負値が必要。`None`は`logq_strength`を継承する。0でもその出所の損失は計算し、補正だけを省く。pooledでは共通の強さを使う |
| `scale` | 補正の前に類似度へ掛ける倍率。既定20。有限の正の値が必要で、「温度」の逆数に当たる |
| `combine` | `pooled`は候補をまとめて計算。`separate`は負例の出所ごとに計算し、有効なものだけ平均。`auto`はyiでseparate、それ以外でpooledを選ぶ |
| `q_measure` | `expected_count`は同じ文書の重複を残し、平均して何回現れるかを使う。`inclusion`は有効な候補を文書ごとに1件にし、少なくとも1回現れる確率を使う。yiではinclusionを使えない |
| `mini_batch_size` | Cached版で一度に計算するクエリ行数。正の整数を指定する。小さくするとスコア行列のメモリを減らせる。勾配の蓄積だけでは候補数は増えない |

separateの出所は、バッチ内・一様・追加の採掘負例です。使える負例がない出所を平均から外します。たとえば学習行が1件のバッチでも、一様負例の損失を空のバッチ内損失と平均して半分にしてしまうことはありません。

## 既知の正例を負例から外す

**マスク**は、その候補を損失の計算対象から外すことです。

- `mask_same_document=None`では、正例の文書が偶然ほかの候補にも現れたとき、yiは残し、plugin/improvedは除きます。TrueやFalseで変更できますが、学習の目的関数も変わります。
- `mask_same_query=True`では、現在のバッチにある同じクエリの正例を利用します。コーパス全体の関連文書がすべて分かるわけではありません。この除外は実用上の工夫で、後述の厳密な不偏性の保証には含まれません。

yiの既定（`mask_same_document=False`）では、同じクエリに属する正例列だけを除きます。同じ文書が一様列・採掘列にあっても残ります。既知文書の一致を全候補列へ適用するには`mask_same_document=True`を使いますが、yiの目的関数が変わります。

yiでseparateの一様側の補正を有効にする場合は、すべての正例IDがコーパスにも必要です。満たさなければエラーになります。

## バッチサンプラーを選ぶ

Cosmeaの標準手順では`BatchSamplers.BATCH_SAMPLER`と`mask_same_document=True`、`mask_same_query=True`を使います。頻度から計算するQと行の抽出方法を揃え、既知正例を負例から除外します。

既定のランダムサンプラーで`mask_same_document=True`を使うと、アンカーの既知正例と同じ文書IDを負例から除外します。`mask_same_query=True`も使うと、同じクエリに紐づくバッチ内の別の正例も対象になります。一様負例や採掘負例に混ざった正例も除外できます。それ以外の負例文書の重複は残ることがあります。 マスクした候補の枠へ別の学習行を補充する処理はないため、重複が多いデータでは有効な負例数が減ります。一様負例の数は、未知データでの検索指標を見て調整してください。

### 既存のNO_DUPLICATES設定を使う場合

Sentence Transformersは[バッチ内負例を使う損失に`NO_DUPLICATES`を推奨しています](https://sbert.net/docs/package_reference/base/sampler.html)。このサンプラーは列をまたいで値を比較し、同じ値を持つ行が同一バッチに入るのを防ぎます。Sentence Transformers 6.3では明示的なID列も比較対象です。クエリと文書のIDには、`q:123`と`d:123`のように別の接頭辞を使うと、別の対象が同じ値として扱われるのを避けられます。

`NO_DUPLICATES`では、ある文書の行が追加される確率が、すでに選んだ行の値に依存します。現在の頻度ベースのバッチ内Qは、`q_measure="inclusion"`でも学習行の一様抽出を前提にします。重み付き・グループ化サンプラーにも個別の抽出確率が必要です。コーパスからの一様抽出は行サンプラーから独立しているため、補正にはコーパス内の文書数と抽出数を使います。

`NO_DUPLICATES`とMNSを組み合わせて一様負例だけを補正するには、コレーターで`num_uniform > 0`を指定し、Trainerには次の設定を渡します。

```python
from sentence_transformers.base.sampler import BatchSamplers

args = SentenceTransformerTrainingArguments(
    output_dir="outputs/run",
    batch_sampler=BatchSamplers.NO_DUPLICATES,
)
loss = LogQMultipleNegativesRankingLoss(
    model,
    variant="plugin",
    combine="separate",
    in_batch_logq_strength=0,
    uniform_logq_strength=1,
    mask_same_document=True,
)
```

pluginでは正例のスコアが補正されないため、一様負例の補正が損失に反映されます。yiのseparateでは、正例と一様負例すべてに同じlogQ補正がかかるため、softmax内で相殺されます。yiで一様側の補正を有効にした場合、正例は引き続きコーパス内にある必要があります。通常版とCached版のどちらでも同じ補正強度を指定できます。

pluginの一様側の損失は $\log(1+(M/K)^{\lambda_u}S_u/e^{F_p})$ です。$S_u$はマスク後の有効な一様負例の指数スコアの和、$F_p$は正例スコア、$\lambda_u$は一様側の補正強度です。強度1ではコーパス全体の負例の和を推定します。正例に対するマージンを$\log(M/K)$だけ変え、勾配に反映します。文書ごとの人気の違いを補正するものではありません。和が不偏でも、損失や勾配の不偏性、検索品質の改善は保証されません。無補正のバッチ内損失とseparateで平均すると、両者を組み合わせた目的関数になります。この平均には有効な負例がある出所だけが入ります。

## 評価と再現性

頻度は実際に抽出する学習splitから数えます。学習に使っていないデータで品質を測る場合は、コーパス全体を検索して指標を計算してください。別の評価splitへ学習頻度を使い回すと、一般に補正する確率がずれます。Trainerの評価テストは、情報が損失まで届くことの確認であり、未知データでの品質評価ではありません。

未知クエリの品質評価には、Trainerへ検索用`evaluator`を渡し、`eval_dataset`は省いてください。`eval_queries`はquery IDからテキストへの辞書、`eval_relevant_docs`は同じクエリ IDから正解のコーパスID集合への辞書です。IDはコーパスの辞書と揃え、必要ならモデルに合う評価用promptも指定します。

```python
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator

evaluator = InformationRetrievalEvaluator(
    queries=eval_queries,
    corpus=dict(zip(corpus_ids, corpus_texts)),
    relevant_docs=eval_relevant_docs,
    name="held_out",
)
```

学習dataset・コレーターと一緒に`evaluator=evaluator`を渡します。`eval_dataset`に学習頻度にない正例IDが含まれると、`Positive ID absent from document_frequency`で停止します。既知IDだけでも、一般に学習頻度は評価側の抽出確率に合いません。コレーターは全コーパス評価器ではありません。早期終了や最良checkpointの選択には、`improved`の損失値ではなく、未知クエリの検索指標を使ってください。

DataLoaderの各workerはPyTorchの乱数を元に別々の乱数列を使います。再現には同じ起動条件が必要です。途中checkpointからの完全な乱数復元や、worker数を変えても同じ抽出順になることは保証していません。

対応範囲はSentence Transformers `>=6.3,<6.4`、PyTorch `>=2.6,<3`、NumPy `>=2,<3`です。CPU・1学習プロセスで検証しています。GPU、分散学習、ストリーミングには追加の検証が必要です。[資源と互換性の制限](design.md#資源と互換性の制限)も参照してください。

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

CPU検証では32・16・8次元で、通常版・Cached版の結果を、次元別に明示計算した損失と比較しています。wrapperの比較はplugin/expected-countで異なる重みを使い、損失間の恒等式はスコア水準で別に確認します。Trainerでは追加負例と最後の1行バッチを含め、通常yiとMatryoshkaで包んだCached yiの学習・評価も確認します。これは接続と勾配の検証で、短くした後の検索品質を保証するものではありません。ランダムな次元選択（`n_dims_per_step`）、GPU、Matryoshka2dLossはここでは未検証です。

## 詳細：損失へ渡す情報

通常はコレーターに任せて構いません。スコアを直接渡す関数を使う場合の仕様です。

学習行数を$B$、追加の負例列数を$H$、一様負例数を$K$とすると、候補数は$C=B(1+H)+K$です。正例、追加列ごとの候補、一様負例の順に並びます。以下のshapeはテンソルの各次元の大きさです。

| キー | shape / 型 | 意味 |
| --- | --- | --- |
| `query_id` | $(B,)$ / int64 | クエリのID |
| `cand_id` | $(C,)$ / int64 | 候補文書のID |
| `cand_count` | $(C,)$ / int64 | 各文書が正例になっている学習行数 |
| `cand_source` | $(C,)$ / int64 | 0=バッチ内、1=一様、2=追加列 |
| `cand_in_corpus` | $(C,)$ / bool | コーパスから抽出できる文書かどうか |
| `num_rows` / `corpus_size` | scalar / int64 | 全学習行数$N$ / コーパスの重複しない文書数$M$ |

これらは`batch["label"]`で損失へ渡し、モデルの入力特徴には混ぜません。損失クラスは平均した1つの値を返します。`mixed_negative_logq_loss`はクエリ行ごとの値をshape $(b,)$で返します。分割計算時の`rows`は、元のバッチの行番号であり、その行の正例の候補列番号でもあります。

`candidate_log_q(..., rows=...)`はshape $(b,C)$で、正例の行が既に選ばれていることを考慮します。`rows`を省くと$(C,)$になり、その1回分を引きません。`iid_proposal=True`はyi用のcount/$N$近似で、コレーターの抽出方法を変える指定ではありません。

低水準の`candidate_log_q`・`negative_log_partition`・`stable_id`は既存利用との互換性のため残しますが、通常はクラスと`mixed_negative_logq_loss`を使ってください。補助関数だけの`iid_proposal=True, q_measure="inclusion"`は二項分布の包含確率を計算する指定で、yi/inclusionの損失を有効にするものではありません。
