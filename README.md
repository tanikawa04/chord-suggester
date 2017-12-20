# Chord Suggester

[Nextremer Advent Calendar 2017 20日目](https://qiita.com/tanikawa/items/8f1a5b3a33f24ed0a984)のプロジェクトです。


## Environment

- Python 3.5+
- Chainer 2+
- Numpy
- Cupy

## Usage

train.txt と val.txt はダミーデータなので、自前のコード進行のデータセットを用意してください。

### 学習

```
$ python train.py
```

### コード進行生成

```
$ python interactive.py
> CM7 FM7
C F G Am G Dm Csus4 C Csus4 C C G/B Am Dm7 G C G/B Am F

>
```

半角スペース区切りでコードを入力してください。そのコードに続く進行を生成してくれます。使用可能なコードは[こちら](https://gist.github.com/tanikawa04/aec431def7b550bc9b2cc2f8404b6c21#file-chord_list-txt)を参考にしてください（※調の正規化前のコード一覧なので、一部使用できないコードがあります）。
