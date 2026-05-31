# PyTorch で入力正規化を行う方法

このプロジェクトで画像入力を正規化するときの基本をまとめる。

## 1. 入力正規化とは何か

入力正規化は、各画素値のスケールをそろえて学習しやすくする前処理。

`torchvision.transforms.ToTensor()` は、画像を `0` から `255` の整数値ではなく、`0.0` から `1.0` の `float` テンソルへ変換する。

その上で `transforms.Normalize(mean, std)` を使うと、各画素を次の式で変換できる。

```python
x_normalized = (x - mean) / std
```

これにより、値の中心がそろい、勾配降下が安定しやすくなる。

## 2. FashionMNIST での基本形

FashionMNIST はグレースケール画像なので、チャンネル数は 1。

そのため `mean` と `std` は 1 要素のタプルで指定する。

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(lambda x: x.view(-1)),
])
```

この設定では、`0.0` から `1.0` の値をだいたい `-1.0` から `1.0` 付近へ移すことになる。

まずはこの形から始めればよい。

## 3. どこに入れるか

このプロジェクトでは `datasets.FashionMNIST(...)` の `transform=` に渡す。

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(lambda x: x.view(-1)),
])

train_ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transform,
)

valid_ds = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transform,
)
```

`Normalize` は `ToTensor()` の後に置く。

また、MLP で平坦化するなら `view(-1)` はその後でよい。

## 4. mean と std はどう決めるか

最初は `0.5, 0.5` のような簡単な値で十分。

ただし、より厳密にやるなら学習データセットから平均と標準偏差を計算して使う。

考え方は次の通り。

1. 学習データだけを使う
2. 平均 `mean` を計算する
3. 標準偏差 `std` を計算する
4. その値を `Normalize(mean, std)` に入れる

例:

```python
from torchvision import datasets, transforms

raw_train_ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)
```

このあとテンソル全体から平均と標準偏差を集計して、`Normalize((mean,), (std,))` の形で使う。

## 5. まず覚えておくべき実務上のポイント

`ToTensor()` だけでも学習は進むが、正規化を入れた方が安定しやすい。

FashionMNIST のような画像分類では、精度が少し改善することがある。

ただし、精度が大きく伸びない場合は、正規化よりもモデル構造の制約が強いことも多い。

特に MLP は画像の空間構造を使えないので、`90%` を少し超えたい程度なら効くことはあるが、大幅改善までは期待しすぎない方がよい。

## 6. このプロジェクト向けの最小変更

現在の `transform` が次のようになっているなら:

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1)),
])
```

次のように変更する。

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(lambda x: x.view(-1)),
])
```

変更点は `Normalize((0.5,), (0.5,))` を 1 行追加するだけ。

まずはこれで再学習し、validation accuracy の差を見るのがよい。
