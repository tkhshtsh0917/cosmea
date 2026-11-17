# 設計・目的関数・互換性

[English](../design.md)

## 責務

`src/cosmea`は実行時コード、`examples`は利用例、`tests`は振る舞い・数式検証、`experiments`は任意の比較実験、`scripts`は開発ツール。学習連携の依存は`train` extra、開発ツールは`dev`へ分離する。公開文書だけで利用・ビルド・評価できる構成にする。

生のバッチが見えるコレーターが一様コーパス候補K件とmetadataを追加する。損失が温度スケール後のスコアを補正する。logit補正はサンプラーだけでは実現できない。B行の選択には既定のランダム行サンプラーを使う。

## 提案分布の尺度

N＝学習行数、B＝バッチ行数、M＝distinct corpus item数、K＝一様負例数、p_i＝行iの正例、c_d＝item dの学習回数。行iを固定すると、残りN−1行からn=B−1行を引き、回数はc'_id=c_d−1[d=p_i]となる。コーパス抽出は行抽出と独立とする。

期待出現回数と混合候補の包含確率は次の通り。

```
e_i(d) = (B-1) * c'_id/(N-1) + (K/M) * 1[d in corpus]
pi_in_i(d) = 1 - choose(N-1-c'_id, B-1)/choose(N-1, B-1)
pi_i(d) = 1 - (1-pi_in_i(d)) * (1-(K/M)*1[d in corpus])
```

B=N=1のin-batch成分、K=0のuniform成分は0と定義する。期待出現回数で割るときは重複を残し、包含確率で割るときは有効候補をitemごとに1件へまとめる。例えば `[a,b,b]` から2件を引くと、期待回数は2/3と4/3、包含確率は2/3と1になる。bを1件にした後も4/3で割ると寄与を過小評価する。

pooledの負例分配関数の不偏性は、指定した抽出方式、正例IDの除外、正のsupportを持つitem集合についてのみ扱う。query依存maskや採掘負例にはそのまま適用しない。logや比を取った後の損失・勾配は一般に有限標本で偏る。

## 目的関数と既定値

スコアFのshapeは`(B,C)`。追加負例列H個の場合、C=B(1+H)+K。`F=scale*similarity`を計算してから`lambda*log q`を引く。similarityへの補正を先に行うと別の目的関数になる。マスクした候補は正規化集合から除く。

既定のyi＋expected_count＋separateは経験周辺分布c_d/Nによるiid近似を使う。実際のコレーターは非復元抽出なので、iidランキングNCEの一致性定理は実装の保証にならない。yi＋inclusionは拒否する。uniformを分けるyiには全正例IDのコーパス包含を要求する。固定数の情報源混合、フィルタ、重複除去でも仮定が変わる。

pluginは正例を無補正にする。z=log(Zhat_neg)なら損失は`softplus(z-F_positive)`。improvedは`sg(sigmoid(z-F_positive))*(z-F_positive)`で、sgは重みをstop-gradientする。同じzを使えば一階微分は一致するが、値や高階微分は違う。引用論文の信頼度重み付けを適応したもので、論文のQ′や著者コードの候補生成を逐語的に再現したものではない。

iidの負例と十分な表現力を仮定すると、損失でlogQを引く学習は観測された条件付きitem分布を目指し、無補正では確率比を目指す。ただし正例ログ自体の露出バイアス除去や、人気集中の低減を保証しない。既定値は比較の出発点であり、性能優位の実証結果ではない。

## 上限と数値動作

`0 <= K <= M`、`1 <= B <= N`、正の整数Cachedチャンクサイズを要求する。不正option・ID・頻度は例外になる。未知正例へ架空の回数を与えない。負例が空なら損失・勾配は0。separateでは有効な情報源だけ平均する。half/bfloat16のスコアを損失内部で昇格するが、混合精度モデル学習は未検証。

コーパスのID索引はテキスト本体に加えてO(M)のメモリを使う。通常版スコアのメモリはO(B*C)、Cached版はb行ずつ計算するためO(b*C)だが候補埋め込みのcacheは残る。包含確率は各候補についてB−1回の積を取り、チャンク当たりO((B−1)*b*C)の演算となる。自動メモリ上限・コーパスのストリーミングは実装していない。大規模バッチはまずexpected_countを使う。

## 互換性と評価

ST 6.3の内部コレーター補助関数とGradCacheに依存し、STは`>=6.3,<6.4`、PyTorchは`>=2.15,<2.16`を対象とする。開発環境の厳密版はlockで固定。v5互換は主張しない。範囲更新時はTrainerのmetadata転送とcached勾配を再確認する。

NO_DUPLICATES、weighted/grouped sampler、DDP padding、cross-device gather、ストリーミング／時間窓頻度、複数datasetには別の提案分布定義と検証が必要。GPU実行は未検証。採掘負例を含め一貫したID体系を使う。

実データでは時系列split、train-only頻度、全コーパスRecall/NDCG、人気bucket、coverage、集中度を複数seedと不確実性推定で評価する。baseline間で候補・mask・softmaxの分割を揃える。lambda=0だけではpooled MNRLと一致しない。合成データが検証するのは有限標本のcoverageであり、露出方策の因果効果ではない。

## 任意の合成データ実験

setup後、checkout内で比較ハーネスを実行する。

```bash
mise exec -- uv run --frozen --no-sync python experiments/toy_two_tower.py --seeds 3 --out results.json
```

これは時間のかかる任意実験で、`mise run check`とは分離している。合成クリックログを生成し、KL・検索・tail・coverageを報告する。出力ファイルには引数とseed別の結果が入る。結果を保存するときはソースの版、環境、評価条件も記録する。この改訂では既定設定の全実験を再実行しておらず、高速化・実データ性能の優位は主張しない。

実験指標のheadは人気順の上位`ceil(0.1 * M)`件とし、同率は元のitem順で決める。tailはその補集合で、少なくとも2 itemを必要とする。top-kは`min(k, M)`件を使い、未観測itemの占有率も実際の取得件数で割る。学習stepが0になる設定や、コーパスを超える一様抽出数は学習前に拒否する。
