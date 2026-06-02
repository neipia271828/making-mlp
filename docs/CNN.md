# PyTorch で CNN モデルを作る方法

このプロジェクトで FashionMNIST 用の CNN を作るときの基本と、`CIFAR10` のような複数チャネル画像へ広げる方法をまとめる。

## 1. CNN とは何か

CNN は `Convolution` 層を使って、画像の位置関係を保ったまま特徴を取り出すモデル。

MLP との大きな違いは、画像を最初から 1 次元に潰さず、`高さ x 幅` の構造を使えること。

FashionMNIST のような画像分類では、MLP より CNN の方が精度を出しやすいことが多い。

## 2. MLP と何が違うか

このプロジェクトの MLP では、入力画像を `784` 次元ベクトルへ平坦化している。

```python
transforms.Lambda(flatten_tensor)
```

CNN では通常、この平坦化を入力前処理では行わない。

理由は、畳み込み層が `1 x 28 x 28` のような画像テンソルをそのまま受け取るから。

つまり CNN 用の前処理では、まず次の形が基本になる。

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
```

## 3. FashionMNIST 用の最小 CNN

FashionMNIST はグレースケール画像なので、入力チャンネル数は `1`。

最小構成なら、畳み込みを 2 回行って最後に全結合層へつなげればよい。

```python
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
```

## 4. なぜ `64 * 7 * 7` になるのか

FashionMNIST の元画像は `28 x 28`。

上の構成では `MaxPool2d(2)` を 2 回使うので、サイズは次のように変わる。

1. `28 x 28`
2. `14 x 14`
3. `7 x 7`

最後の畳み込み出力チャンネル数が `64` なので、全結合層へ入る直前の特徴量数は `64 * 7 * 7` になる。

ここが CNN 実装でよく詰まりやすい点。

## 5. このプロジェクトで前処理をどう変えるか

今の MLP 向け前処理では平坦化が入っている。

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(flatten_tensor),
])
```

CNN ではこれをやめて、画像の形を保つ。

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
```

つまり、`MLP` と `CNN` を切り替えるなら、モデル本体だけでなく `transform` も切り替える必要がある。

## 6. 学習ループはほぼそのまま使える

CNN に変えても、学習ループ自体はほとんど同じ。

```python
for x_batch, y_batch in train_loader:
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    optimizer.zero_grad()
    logits = model(x_batch)
    loss = criterion(logits, y_batch)
    loss.backward()
    optimizer.step()
```

変わるのは主に次の 2 点。

1. `model = CNN().to(device)` にする
2. `transform` から平坦化を外す

## 7. このプロジェクト向けのファイル分割案

今の構成に合わせるなら、たとえば次のように置ける。

```text
src/model/CNN.py
```

中身は次のように書ける。

```python
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
```

## 8. `constants.py` の `MODEL` で切り替える考え方

`MODEL = "MLP"` や `MODEL = "CNN"` を切り替えたいなら、モデル生成を 1 か所に集めると扱いやすい。

例:

```python
from model.MLP import MLP
from model.CNN import CNN

def build_model(model_name: str):
    if model_name == "MLP":
        return MLP()
    if model_name == "CNN":
        return CNN()
    raise ValueError(f"Unknown MODEL: {model_name}")
```

そして `train.py` では次のように使う。

```python
from constants import MODEL

model = build_model(MODEL).to(device)
```

ただし、CNN では前処理も変わるので、将来的には `build_dataloaders()` 側も `MODEL` を受け取って分岐した方が自然。

## 9. 精度改善のために試しやすい要素

最小 CNN が動いたら、次を順に試せる。

1. 畳み込みチャンネル数を増やす
2. 全結合層の幅を増やす
3. `Dropout` を入れる
4. `batch_size` を見直す
5. 軽いデータ拡張を入れる

たとえば `Dropout` を入れるなら次のようになる。

```python
self.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(64 * 7 * 7, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 10),
)
```

## 10. CNN を作るときの注意点

このプロジェクトで詰まりやすい点は次の通り。

1. `flatten_tensor` を残したまま CNN に入れてしまう
2. `Linear` の入力次元 `64 * 7 * 7` を間違える
3. `MLP` 用の前処理をそのまま使う
4. モデル切り替えだけして `build_dataloaders()` を変えていない

特に 1 と 4 は起こりやすい。

CNN では入力が `[batch_size, 1, 28, 28]` であることを前提にするので、ここが崩れると shape error になりやすい。

## 11. まとめ

このプロジェクトで CNN を作るなら、まずは `Conv2d -> ReLU -> MaxPool2d` を 2 段重ねた最小構成で十分。

MLP から切り替えるときは、モデル本体だけでなく `transform` の平坦化を外すことが重要。

## 12. 複数チャネルのデータセット向けにするときの考え方

ここまでは `FashionMNIST` のような `1ch` 画像を前提にしていた。

ただし `CIFAR10` のようなカラー画像では、入力は `3 x 32 x 32` になる。

そのため、複数チャネル対応で最初に確認する点は次の 3 つ。

1. `Conv2d` の最初の `in_channels`
2. `Normalize()` の平均値と標準偏差の長さ
3. 畳み込みとプーリング後の空間サイズ

## 13. 最初の `Conv2d` の入力チャンネル数を合わせる

グレースケール画像なら `in_channels=1`、RGB 画像なら `in_channels=3` にする。

たとえば `CIFAR10` を想定した最初の層はこうなる。

```python
nn.Conv2d(3, 32, kernel_size=3, padding=1)
```

つまり、単一チャネル向け CNN を複数チャネル向けに変えるときは、まずここを合わせる。

## 14. `Normalize()` もチャンネル数に合わせる

`Normalize()` の `mean` と `std` は、チャンネルごとに値を持つ。

`1ch` ならこうなる。

```python
transforms.Normalize((0.5,), (0.5,))
```

`3ch` ならこうなる。

```python
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
```

長さが合っていないと、前処理の時点で shape error になりやすい。

## 15. `Linear` の入力次元は画像サイズに応じて変わる

`FashionMNIST` は `28 x 28` なので、`MaxPool2d(2)` を 2 回通すと `7 x 7` になった。

一方で `CIFAR10` は `32 x 32` なので、同じ構成なら次のように変わる。

1. `32 x 32`
2. `16 x 16`
3. `8 x 8`

最後の出力チャンネル数が `64` なら、全結合層の入力は `64 * 8 * 8` になる。

つまり、`1ch` か `3ch` かだけでなく、入力画像サイズも `Linear(...)` の次元に効く。

## 16. `CIFAR10` 向けの最小 CNN 例

複数チャネル画像向けの最小例としては、次のように書ける。

```python
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
```

この例では、`28 x 28` 前提の `64 * 7 * 7` をそのまま流用しないことが重要。

## 17. 今のリポジトリでは `in_channels` をデータセット設定から取るとよい

このリポジトリには `src/lib/dataset_builder.py` の `DATASET_CONFIGS` があり、データセットごとの `in_channels` を持てるようになっている。

たとえば次のような情報をまとめられる。

```python
DATASET_CONFIGS = {
    "FashionMNIST": DatasetConfig(..., in_channels=1, image_size=(28, 28), ...),
    "CIFAR10": DatasetConfig(..., in_channels=3, image_size=(32, 32), ...),
}
```

この形にしておくと、前処理側とモデル側でチャンネル数の前提がずれにくい。

## 18. モデルを複数データセット対応にしたいときの形

`in_channels` をコンストラクタで受け取ると、単一チャネル用と複数チャネル用を 1 つのクラスで扱いやすい。

```python
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, image_size: tuple[int, int]) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        height, width = image_size
        feature_height = height // 4
        feature_width = width // 4

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * feature_height * feature_width, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
```

こうしておくと、`FashionMNIST` なら `in_channels=1`、`CIFAR10` なら `in_channels=3` を渡すだけで済む。

## 19. このプロジェクトで複数チャネル対応するときの注意点

複数チャネル対応で詰まりやすいのは次の通り。

1. `Conv2d(1, ...)` のままで RGB 画像を入れてしまう
2. `Normalize((0.5,), (0.5,))` のままで 3ch 画像を使ってしまう
3. `64 * 7 * 7` を `32 x 32` 画像でもそのまま使ってしまう
4. モデルだけ `3ch` 対応にして、データセット設定や前処理を変えていない

複数チャネル対応は、モデル本体だけの修正では完結しない。

`dataset config`、`transform`、`Conv2d`、`Linear` の 4 か所をセットで見ると崩れにくい。
