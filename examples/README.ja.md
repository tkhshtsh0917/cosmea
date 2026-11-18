# 利用例

[English](README.md)

## 手元のデータで学習する

[train_from_csv.py](train_from_csv.py)を使ってください。既存のモデルを読み込み、学習ペアと検索対象をCSVから受け取り、CPUで学習して保存する例です。オフライン検査用のモデル・tokenizerを自作する処理は含みません。

まず[train extra付きでCosmeaをインストール](../README.ja.md#インストール)し、`train_from_csv.py`をお使いのプロジェクトへコピーします。UTF-8のCSVに次の列を用意してください。BOM付きのUTF-8にも対応しています。以下は入力形式を示す例です。

`train.csv`：

```csv
anchor,positive,query_id,document_id
python lists,A guide to Python lists,q1,p1
sql joins,An introduction to SQL joins,q2,p2
```

`corpus.csv`：

```csv
document_id,text
p1,A guide to Python lists
p2,An introduction to SQL joins
p3,A guide to garden flowers
```

`anchor`はクエリのテキスト、`positive`は対応する文書のテキストです。テキスト列の`query`と`document`、旧CSVの`document_id`に相当する`item_id`も読み込めます。

IDはすべて文字列として読みます。コーパスにも同じ正例IDがある場合は、両方のテキストが同じ文書を表すようにしてください。

```bash
uv run python train_from_csv.py \
  --model path/to/local/model \
  --train train.csv \
  --corpus corpus.csv \
  --output outputs/retriever \
  --uniform 2
```

CSVスクリプトはREADMEの標準手順に合わせ、`plugin`・pooled・期待出現回数を使います。`BatchSamplers.BATCH_SAMPLER`、`mask_same_document=True`、`mask_same_query=True`を明示しています。APIで損失の種類を省略した場合の既定は引き続きyiです。

`--model`には、ローカルのSentence Transformersモデルのディレクトリか、Hugging FaceのモデルIDを指定できます。後者ではモデルをダウンロードする場合があります。一様負例の数はコーパスの重複しないID数以下にしてください。学習したモデルは`outputs/retriever/model`へ保存します。保存するrunの場所は`--output`で選べます。

これは学習手順の例であり、品質ベンチマークではありません。検索品質を判断する前に、別のクエリでコーパス全体を検索して評価してください。抽出条件は[APIガイド](../docs/ja/api.md)、短い埋め込みでも学習する方法は[Matryoshkaの利用](../docs/ja/api.md#複数の埋め込み次元で学習する)を参照してください。

promptを使うモデルでは、`--query-prompt "query: "`・`--document-prompt "passage: "`へモデルが要求する接頭辞を指定します。ここで示した文字列は例で、全モデルの既定値ではありません。`--max-length 256`はコレーターのトークン上限です。RouterのtaskはAPIガイドに従い、コレーターへ`router_mapping`を直接渡します。TrainingArgumentsだけではこれらの設定は転送されません。

promptの引数は学習時の前処理を設定するもので、保存モデルの既定promptを書き換えません。再読込後のエンコードにも、同じクエリ／文書の接頭辞を適用してください。

## 難例採掘から再ランキングまで

[mine_train_rerank.py](mine_train_rerank.py)で一連の処理をCPU上で試せます。上記のとおりtrain extra付きでインストールし、CSVの読み込みに使う[train_from_csv.py](train_from_csv.py)と同じ場所に置いてください。既存のSentenceTransformerと、ペアごとにスコアを1つ返す学習済みCrossEncoderを指定します。ローカルのモデルディレクトリとHugging FaceのモデルIDのどちらも使えます。

`corpus.csv`は上の例と同じ形式で、文書IDは一意の文字列です。ここでの`train.csv`と`eval.csv`はどちらも`query_id,query,document_id`の形式で、関連文書を1行に1件書きます。各ファイル内では、同じクエリ本文には同じIDを割り当て、1つのIDには1種類の本文を対応させます。正例が複数ある場合は、同じクエリIDと本文で行を追加します。学習と評価の間では、クエリID・本文の両方を分けてください。各行の`document_id`には`corpus.csv`にある文書IDを指定します。上の`corpus.csv`と以下の2ファイルで動作を試せます。

`train.csv`：

```csv
query_id,query,document_id
q1,python list operations,p1
q2,how to join SQL tables,p2
```

`eval.csv`：

```csv
query_id,query,document_id
q3,flower gardening,p3
```

学習前に採掘結果を確認するには、採掘だけを実行します。

```bash
uv run python mine_train_rerank.py \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --train train.csv \
  --corpus corpus.csv \
  --output outputs/two-stage-preview \
  --mine-only
```

`outputs/two-stage-preview/mined.csv`で文書IDを確認してください。負例に選ばれた文書が実際には関連している場合は、そのIDを既知正例として`train.csv`に追加します。一連の処理を実行する際は採掘時と同じ入力と設定を使います。入力CSVから採掘をやり直すため、関連度の修正は`train.csv`に反映します。実行ごとに新しい出力先を指定してください。

```bash
uv run python mine_train_rerank.py \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --reranker cross-encoder/ms-marco-MiniLM-L6-v2 \
  --train train.csv \
  --eval eval.csv \
  --corpus corpus.csv \
  --output outputs/two-stage \
  --uniform 1 \
  --candidates 2 \
  --k 1
```

1. **採掘する。** 学習前の検索モデルで、学習クエリごとに最も近い候補を1件選びます。そのクエリの既知正例と、既知正例に本文が完全一致する文書を除外します。未判定の候補が実際には関連している場合もあるので、信頼できる負例として使う前に採掘結果を確認してください。
2. **学習する。** ランダムな行抽出とCosmeaの`plugin`／`separate`を使います。バッチ内負例と一様負例には補正を掛け、採掘負例は独立した無補正の項にします。採掘負例列はバッチ内で共有するため、採掘時の除外は別のクエリの正例まで保証しません。学習時の既知正例マスクは同じバッチ内の情報を使います。
3. **評価する。** 学習後の検索モデルで全コーパスを検索し、取得した上位`--candidates`件を固定したCrossEncoderで並べ替えます。検索漏れした正例は補いません。すべての既知正例IDを基準に、同じ`--k`で平均nDCG@k・Recall@k・MRR@kを比較します。候補の再現率は、第一段で見つかった正例の割合をクエリごとに求めた平均です。

学習の前に`mined.csv`へ学習クエリ・正例・採掘負例のIDを保存します。一連の処理では、さらに`model/`、`metrics.json`、`rankings.jsonl`が出力されます。前後のランキングにはどちらも学習済み検索モデルが取得した候補を使います。`before`のスコアはコサイン類似度、`after`はCrossEncoderのロジットです。段階間の比較には順位や指標を使ってください。

学習時のマスクと評価時の関連度判定は文書IDを使います。本文が同じでもIDが異なれば、学習時に負例になったり、評価時に異なる関連度として扱われたりする場合があります。複数のIDが同じ文書を指す場合は、実行前にコーパスを重複排除し、正例のIDも揃えてください。コサイン類似度が同点ならコーパス順、CrossEncoderのスコアが同点なら検索時の候補順を保ちます。

小規模コーパス向けの全件コサイン検索と二値の関連度を使います。未判定文書は評価上の加点対象ではありません。この少量データは動作確認用です。品質は用途に近い別の評価データで確認してください。

既定ではモデルの`query`と`document`のpromptを使い、対応するpromptがなければ接頭辞は空になります。モデルが`passage`や`corpus`など別のキーを使う場合は、その接頭辞を`--document-prompt`で指定します。クエリ側は`--query-prompt`で上書きできます。空文字を明示すると、対応する接頭辞を無効にできます。採掘・学習・評価では同じ接頭辞を使います。保存したモデルには元の既定設定が残るため、読み込み直した後も同じ上書きを適用してください。

## ダウンロードなしで接続を確認する

[train_example.py](train_example.py)は、小さなランダム初期化BERTと合成データをローカルで作ります。通常版・Cached版の動作確認とCIに使うため、tokenizerやモデルを自作する処理が含まれています。実行手順は[CONTRIBUTING.ja.md](../CONTRIBUTING.ja.md#オフライン学習例と任意の実験)にあります。

`experiments/`の任意比較は、合成データ上で損失の設定を比べる研究用ツールです。ライブラリの利用には不要です。実行コマンドと制限もCONTRIBUTINGにまとめています。
