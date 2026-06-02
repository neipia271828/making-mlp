# PyTorch でデータ拡張を行う方法

このプロジェクトで FashionMNIST にデータ拡張を入れるときの基本をまとめる。

## 1. データ拡張とは何か

データ拡張は、学習用画像に小さな変形を加えて訓練データの見え方を増やす方法。

たとえば次のような効果を狙える。

1. 過学習を減らす
2. 少ないデータでも汎化しやすくする
3. 位置ずれや傾きに少し強くする

ただし、FashionMNIST のような 28x28 の小さい画像では、強すぎる変形を入れるとかえって精度が落ちることがある。

## 2. このプロジェクトで拡張を入れる場所

このプロジェクトでは `src/model.py` の `build_transform()` で前処理を定義している。

今はおおむね次の流れになっている。

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(flatten_tensor),
])
```

データ拡張を入れるときは、通常 `ToTensor()` の前に画像変形系の処理を追加する。

## 3. 学習用データにだけ適用する

重要なのは、データ拡張は `train_ds` にだけ入れ、`valid_ds` には入れないこと。

検証用データまで変形すると、評価がぶれる。

たとえば次のように学習用と検証用で transform を分ける。

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(flatten_tensor),
])

valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(flatten_tensor),
])
```

そのうえで、`datasets.FashionMNIST(...)` に別々に渡す。

```python
train_ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=train_transform,
)

valid_ds = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=valid_transform,
)
```

## 4. まず試しやすい拡張

FashionMNIST で最初に試しやすいのは次の 2 つ。

### 軽い回転

```python
transforms.RandomRotation(10)
```

`10` は最大で前後 10 度だけ回す設定。

服の画像は多少の傾きがあっても同じクラスなので、軽い回転は比較的入れやすい。

### 軽い平行移動

```python
transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
```

これは回転なしで、上下左右に最大 10% ずらす。

実データの位置ずれに少し強くしたいときに使いやすい。

### 左右反転を入れる方法

左右反転を入れるなら `RandomHorizontalFlip` を使う。

```python
transforms.RandomHorizontalFlip(p=0.5)
```

`p=0.5` は、50% の確率で左右反転するという意味。

学習用 transform に加えるなら次のようになる。

```python
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(flatten_tensor),
])
```

ただし、FashionMNIST では左右反転が常に有効とは限らない。

- T-shirt や Pullover のように左右対称に近いクラスには入りやすい
- Sandal や Ankle boot のように向きの情報が効く画像では不利になることがある

そのため、左右反転は「入れ方は簡単だが、精度改善は要検証」という位置づけで考えるのがよい。

## 5. このプロジェクト向けの実用例

MLP のまま試すなら、まずは次のくらいが無難。

```python
def build_train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(flatten_tensor),
    ])


def build_valid_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(flatten_tensor),
    ])
```

その後 `build_dataloaders()` の中で使い分ける。

```python
def build_dataloaders(device: torch.device) -> tuple[DataLoader, DataLoader]:
    train_transform = build_train_transform()
    valid_transform = build_valid_transform()

    train_ds = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=train_transform,
    )

    valid_ds = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=valid_transform,
    )
```

## 6. 入れすぎない方がよい拡張

FashionMNIST + MLP では、次のような強い拡張は最初は避けた方がよい。

1. 大きすぎる回転
2. 左右反転を無条件で強く入れること
3. 強すぎる切り抜き
4. 画像の意味を壊すノイズ

特に左右反転は、衣類画像の見え方を大きく変えることがあり、常に有効とは限らない。

もし試すなら、まずは `p=0.1` や `p=0.2` のような弱め設定から入れて、ベースラインと比較する方が安全。

また、MLP は CNN より画像の空間構造をうまく扱えないので、強い変形を入れても活かしにくいことがある。

## 7. 精度を比較するときの見方

データ拡張を入れたら、少なくとも次を比較する。

1. 最終 accuracy
2. 学習 loss の下がり方
3. 過学習が減ったか
4. 学習時間がどれだけ増えたか

データ拡張を入れると、学習 loss は少し下がりにくくなることがある。

その代わり validation accuracy が改善するなら、汎化には効いていると判断しやすい。

## 8. このプロジェクトでの進め方

まずは次の順で試すのがよい。

1. 拡張なしでベースラインを取る
2. `RandomRotation(10)` だけ追加する
3. `RandomAffine(... translate=(0.1, 0.1))` を追加する
4. 必要なら `RandomHorizontalFlip(p=0.1)` を追加する
5. accuracy と学習時間を比較する

一度に多くの拡張を入れると、どれが効いたのか分からなくなる。

## 9. まとめ

このプロジェクトでデータ拡張を行うなら、`train_ds` にだけ軽い回転や平行移動を入れるのが始めやすい。

まずは `build_transform()` を学習用と検証用に分け、弱い変形から試すのが実務的。

左右反転を入れる場合も、最初は弱い確率で追加し、本当に validation accuracy が上がるかを確認してから強める方がよい。
