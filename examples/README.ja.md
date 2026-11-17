# 利用例

[English](README.md)

## 手元のデータで学習する

[train_from_csv.py](train_from_csv.py)を使ってください。既存のモデルを読み込み、学習ペアと検索対象をCSVから受け取り、CPUで学習して保存する例です。オフライン検査用のモデル・tokenizerを自作する処理は含みません。

まず[train extra付きでCosmeaをインストール](../README.ja.md#インストール)し、`train_from_csv.py`をお使いのプロジェクトへコピーします。UTF-8のCSVに次の列を用意してください。以下は形式を示すための数行で、このままで実用的な学習データになるわけではありません。

`train.csv`：

```csv
query,positive,query_id,item_id
python lists,A guide to Python lists,q1,p1
sql joins,An introduction to SQL joins,q2,p2
```

`corpus.csv`：

```csv
item_id,text
p1,A guide to Python lists
p2,An introduction to SQL joins
p3,A guide to garden flowers
```

IDはすべて文字列として読みます。既定のyi損失では、すべての正例IDがコーパスにも必要です。`positive`のテキストは、対応するコーパスのitemと同じ内容を表すものにしてください。

```bash
uv run python train_from_csv.py \
  --model path/to/local/model \
  --train train.csv \
  --corpus corpus.csv \
  --output outputs/retriever \
  --uniform 2
```

`--model`には、ローカルのSentence Transformersモデルのディレクトリか、Hugging FaceのモデルIDを指定できます。後者ではモデルをダウンロードする場合があります。一様負例の数はコーパスの重複しないID数以下にしてください。学習したモデルは`outputs/retriever/model`へ保存します。保存するrunの場所は`--output`で選べます。

これは学習手順の例であり、品質ベンチマークではありません。検索品質を判断する前に、別のクエリでコーパス全体を検索して評価してください。抽出条件は[APIガイド](../docs/ja/api.md)、短い埋め込みでも学習する方法は[Matryoshkaの利用](../docs/ja/api.md#複数の埋め込み次元で学習する)を参照してください。

promptを使うモデルでは、`--query-prompt "query: "`・`--document-prompt "passage: "`へモデルが要求する接頭辞を指定します。ここで示した文字列は例で、全モデルの既定値ではありません。`--max-length 256`はコレーターのトークン上限です。RouterのtaskはAPIガイドに従い、コレーターへ`router_mapping`を直接渡します。TrainingArgumentsだけではこれらの設定は転送されません。

promptの引数は学習時の前処理を設定するもので、保存モデルの既定promptを書き換えません。再読込後のエンコードにも、同じquery／documentの接頭辞を適用してください。

## ダウンロードなしで接続を確認する

[train_example.py](train_example.py)は、小さなランダム初期化BERTと合成データをローカルで作ります。通常版・Cached版の動作確認とCIに使うため、tokenizerやモデルを自作する処理が含まれています。実行手順は[CONTRIBUTING.ja.md](../CONTRIBUTING.ja.md#オフライン学習例と任意の実験)にあります。

`experiments/`の任意比較は、合成データ上で損失の設定を比べる研究用ツールです。ライブラリの利用には不要です。実行コマンドと制限もCONTRIBUTINGにまとめています。
