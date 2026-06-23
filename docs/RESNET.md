# PyTorch で ResNet を実装する方法

このプロジェクトへ CIFAR-10 用の ResNet-18 相当モデルを追加する方法をまとめる。

この文書では `torchvision.models.resnet18` をそのまま呼び出さず、Residual Block から実装する。ResNet の仕組みと、入力サイズやチャンネル数が変わるときの処理を理解することが目的。

## 1. ResNet が解決する問題

CNN は層を深くすると複雑な特徴を学習できる一方、勾配が前の層まで伝わりにくくなり、深くしたのに学習しづらくなることがある。

ResNet は通常の変換結果 `F(x)` に、入力 `x` を足して次の層へ渡す。

```text
入力 x ───────────────┐
  │                   │
  └─ Conv → BN → ReLU → Conv → BN ─┤ + ├→ ReLU
                                    ↑
                              F(x) + x
```

数式では次の形になる。

```text
y = ReLU(F(x) + x)
```

この入力を直接流す経路を、skip connection または shortcut connection と呼ぶ。

## 2. 最初に追加するファイル

既存の動的ロード方式に合わせて、次の3ファイルを追加する。

```text
src/model/ResNet-v0/
├── constants.py
├── model.py
└── preprocessing.py
```

`src/lib/factory.py` は `CONSTANTS.MODEL` と同名のディレクトリから、モデル、前処理、学習設定を自動で読み込む。`ResNet-v0` のようにハイフン付きの名前を指定した場合、factory は `ResNet_v0`、続いて `ResNet` というクラス名も探索する。そのため、ディレクトリ名は `ResNet-v0`、モデルのクラス名は `ResNet` とすれば factory の変更は不要。

## 3. BasicBlock を実装する

ResNet-18 と ResNet-34 は、3x3 畳み込みを2回行う `BasicBlock` を使う。

`src/model/ResNet-v0/model.py` を、次の `BasicBlock` と後述する `ResNet` クラスで置き換える。既存の `CNN` クラスと `ConvBlockVGG` の import は残さない。

```python
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.residual(x)
        out = out + identity
        return self.relu(out)
```

### shortcut に変換が必要な理由

`out + identity` を計算するには、両方のテンソル形状が同じでなければならない。

たとえば特徴マップを `[64, 32, 32]` から `[128, 16, 16]` へ変えるブロックでは、元の `x` をそのまま加算できない。そこで shortcut 側でも `1x1 Conv` を使い、チャンネル数と画像サイズを合わせる。

```text
residual: [64, 32, 32] → [128, 16, 16]
shortcut: [64, 32, 32] → [128, 16, 16]
```

形状が変わらないブロックでは `nn.Identity()` が入力をそのまま返す。

## 4. CIFAR-10 用 ResNet-18 を組み立てる

同じ `model.py` に `ResNet` を追加する。

```python
class ResNet(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.in_channels = 64

        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(64, block_count=2, stride=1)
        self.layer2 = self._make_layer(128, block_count=2, stride=2)
        self.layer3 = self._make_layer(256, block_count=2, stride=2)
        self.layer4 = self._make_layer(512, block_count=2, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512 * BasicBlock.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(
        self,
        out_channels: int,
        block_count: int,
        stride: int,
    ) -> nn.Sequential:
        blocks = [BasicBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels * BasicBlock.expansion

        for _ in range(1, block_count):
            blocks.append(BasicBlock(self.in_channels, out_channels))

        return nn.Sequential(*blocks)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
```

### CIFAR-10 用に変更している点

一般的な ImageNet 用 ResNet-18 は、先頭に `7x7 Conv, stride=2` と MaxPool を使う。CIFAR-10 は `32x32` と小さいため、その構成をそのまま使うと序盤で画像を縮小しすぎる。

この実装では次の構成にする。

```text
入力                  [3, 32, 32]
stem                  [64, 32, 32]
layer1: 2 blocks      [64, 32, 32]
layer2: 2 blocks      [128, 16, 16]
layer3: 2 blocks      [256, 8, 8]
layer4: 2 blocks      [512, 4, 4]
AdaptiveAvgPool2d     [512, 1, 1]
Linear                [10]
```

`AdaptiveAvgPool2d((1, 1))` を使うため、`Linear` の入力に `512 * 4 * 4` のような画像サイズを直接書く必要はない。

## 5. 学習設定を追加する

`src/model/ResNet-v0/constants.py` を次の内容にする。

最初は既存の CNN-v4 と条件をそろえると、モデル構造だけによる差を比較しやすい。

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConstants:
    NUM_EPOCHS: int = 120
    BATCHSIZE: int = 256
    L_LATE: float = 3e-3
    WEIGHT_DECAY: float = 1e-4
    SCHEDULER_NAME: str = "CosineAnnealingLR"
    ETA_MIN: float = 1e-5


MODEL_CONSTANTS = ModelConstants()
```

現在の `src/train.py` は Adam を固定で使用するため、この設定も Adam 向け。ResNet では SGD + momentum もよく使われるが、最初の実装では同時にオプティマイザまで変更せず、モデルが正しく動くことを先に確認する。

## 6. 前処理を追加する

`src/model/ResNet-v0/preprocessing.py` を次の内容にする。

```python
from torchvision import transforms

from CONSTANTS import CONSTANTS
from lib.dataset_builder import (
    build_dataloaders as build_shared_dataloaders,
    get_dataset_config,
)
from .constants import MODEL_CONSTANTS


def build_transform(dataset_name: str, train: bool) -> transforms.Compose:
    dataset_config = get_dataset_config(dataset_name)
    transform_steps = []

    if train:
        transform_steps.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(dataset_config.mean, dataset_config.std),
        ]
    )

    return transforms.Compose(transform_steps)


def build_dataloaders(device):
    dataset_name = CONSTANTS.PROJECT
    return build_shared_dataloaders(
        dataset_name=dataset_name,
        batch_size=MODEL_CONSTANTS.BATCHSIZE,
        device=device,
        train_transform=build_transform(dataset_name, train=True),
        valid_transform=build_transform(dataset_name, train=False),
    )
```

まずは CIFAR-10 で標準的な `RandomCrop` と `RandomHorizontalFlip` だけを使う。CNN-v4 の強いデータ拡張をそのまま移すと、モデル変更と前処理変更のどちらが結果に影響したか判断しづらくなる。

## 7. ResNet へ切り替える

`src/CONSTANTS.py` の設定を次のように変更する。

```python
class MetaConstants:
    PROJECT: str = "CIFAR10"
    MODEL: str = "ResNet-v0"
```

これにより factory は自動的に次を読み込む。

```text
src/model/ResNet-v0/model.py
src/model/ResNet-v0/constants.py
src/model/ResNet-v0/preprocessing.py
```

## 8. 学習前に shape を確認する

いきなり120 epoch学習せず、まずダミー入力で出力形状を確認する。

プロジェクトルートから次を実行する。

```bash
uv run python -c '
import sys
import torch
sys.path.insert(0, "src")
from importlib import import_module

ResNet = import_module("model.ResNet-v0.model").ResNet

model = ResNet()
x = torch.randn(2, 3, 32, 32)
y = model(x)
print(y.shape)
'
```

成功時は次のようになる。

```text
torch.Size([2, 10])
```

パラメータ数も確認できる。

```bash
uv run python -c '
import sys
sys.path.insert(0, "src")
from importlib import import_module

ResNet = import_module("model.ResNet-v0.model").ResNet

model = ResNet()
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
'
```

この構成は約1,117万パラメータになる。GPUメモリ不足になる場合は、最初に `BATCHSIZE` を `256` から `128` または `64` へ下げる。

## 9. 1 epoch だけ動作確認する

shape 確認後、`constants.py` を一時的に次のようにする。

```python
NUM_EPOCHS: int = 1
```

そのうえで学習を実行する。

```bash
uv run python src/train.py
```

出力の先頭が `cuda` になり、最後まで例外なく完了すれば、モデル、データローダー、GPU転送、ログ保存が接続できている。

```text
cuda
ep=1, ...
```

確認後に `NUM_EPOCHS` を `120` へ戻して本学習を行う。

## 10. 実装時に起こりやすいエラー

### `out + identity` で shape error になる

原因は residual 側だけでチャンネル数または画像サイズを変えていること。

```python
if stride != 1 or in_channels != out_channels:
```

この条件で shortcut 側にも `1x1 Conv` が入っているか確認する。

### `Class ResNet was not found` と表示される

`src/model/ResNet-v0/model.py` 内のクラス名が、factory が探索する `ResNet` になっているか確認する。

```python
class ResNet(nn.Module):
```

### 入力チャンネル数が合わない

CIFAR-10 は3チャンネルなので、stem の先頭は `nn.Conv2d(3, 64, ...)` にする。FashionMNIST、MNIST、KMNISTへ移植する場合は `3` を `1` に変更する必要がある。

### GPUメモリが足りない

まず `BATCHSIZE` を下げる。モデルのチャンネル数を変える前に、次の順で試す。

```text
256 → 128 → 64
```

### 学習は動くが精度が伸びない

一度にモデル、データ拡張、optimizer、learning rateをすべて変更すると原因を切り分けられない。最初は既存条件に合わせ、次の順で比較する。

1. ResNet本体だけを導入する
2. データ拡張を調整する
3. learning rateを調整する
4. AdamとSGDを比較する

## 11. 実装完了の確認項目

次をすべて満たせば、最初のResNet実装は完了。

- `BasicBlock` が `F(x) + x` を計算している
- 形状が変わる場所で shortcut に `1x1 Conv` を使っている
- ダミー入力 `[2, 3, 32, 32]` の出力が `[2, 10]` になる
- `CONSTANTS.MODEL` が `ResNet-v0` になっている
- 1 epoch の学習が最後まで完了する
- GPUサーバー上で先頭に `cuda` と表示される
- 学習ログとモデルが `data/CIFAR10/` 配下へ保存される

## 12. 次の改善候補

最初の比較が終わった後は、1項目ずつ次を試せる。

1. optimizer を SGD + momentum に切り替える
2. CIFAR-10 の平均・標準偏差を使う
3. label smoothing を追加する
4. MixUp または CutMix を追加する
5. mixed precision でGPU学習を高速化する
6. BasicBlock の幅を小さくした軽量版と比較する

変更は一度に1種類にし、CNN-v4と同じ epoch 数、batch size、前処理条件で比較可能なログを残す。

## 参考

- Kaiming He ほか, [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- PyTorch, [ResNet implementation](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
