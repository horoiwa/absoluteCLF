## absoluteCLF

xceptionモデルからの転移学習による分類モデル作成の自動化ツール

強力なData augumentation機能

<br>

### 準備

`images/train` および `images/valid` に<br>
カテゴリ数と同数のフォルダを作成し、画像を配置する
```
absoluteCLF
|
|-images
|    |-train
|    |   |-classA
|    |   |-classB
|    |
|    |-valid
|    |   |-classA
|    |   |-classB
|    |
|    |-test
|        |-classA
|        |-classB
|-src
```

<br>

### 使い方

- トレーニングの開始<br>
`python train.py -p -t`

- `images/test`フォルダ内の画像に対して推論<br>
`python test.py`

- 指定フォルダ内の画像に対して推論<br>
`python predict.py -f FOLDER -o OUTPUT.csv`

<br>

### CONFIG
`config.py`へ設定を記述する
