# PyTorch で scheduler を設定する方法

このプロジェクトで learning rate scheduler を追加する手順をまとめる。

ここでいう scheduler は、学習中に `learning_rate` を段階的または連続的に変化させる仕組み。

特に CNN 学習では、

1. 前半は大きめの learning rate で速く進める
2. 後半は小さめの learning rate で細かく詰める

という動きが欲しくなることが多い。

## 1. scheduler は何のために使うか

learning rate を固定すると、

- 前半は遅い
- 後半は振動しやすい
- ある精度から伸びにくい

ということがある。

scheduler を使うと、epoch が進むにつれて learning rate を落とせる。

たとえば `3e-4` 固定よりも、

- 最初は `3e-4`
- 途中から `1e-4`
- 最後は `3e-5`

のように下げる方が、validation accuracy が伸びやすいことがある。

## 2. このプロジェクトで scheduler を入れる場所

このリポジトリでは学習の本体は [src/train.py](/Users/suzukiakiramuki/projects/learning-llm/making-mlp/src/train.py:1) にある。

流れは大きく次の通り。

1. `optimizer` を作る
2. epoch ごとに学習する
3. epoch の最後で scheduler を進める

つまり、追加位置はこの 2 箇所になる。

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
)

scheduler = ...
```

と

```python
for epoch in range(num_epochs):
    ...
    scheduler.step()
```

## 3. まずは StepLR が一番簡単

最初に試すなら `torch.optim.lr_scheduler.StepLR` が分かりやすい。

例:

```python
from torch.optim.lr_scheduler import StepLR
```

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
)

scheduler = StepLR(
    optimizer,
    step_size=100,
    gamma=0.1,
)
```

これは、

- 100 epoch ごとに
- learning rate を `0.1` 倍する

という意味。

今の `NUM_EPOCHS=300` なら、

- epoch 1-100: `3e-4`
- epoch 101-200: `3e-5`
- epoch 201-300: `3e-6`

のように動く。

epoch の最後で `step()` する。

```python
for epoch in range(num_epochs):
    ...
    scheduler.step()
```

## 4. MultiStepLR はもう少し実用的

このプロジェクトでは `300 epoch` 前後を回すことがあるので、特定の節目だけ下げたいなら `MultiStepLR` の方が使いやすい。

```python
from torch.optim.lr_scheduler import MultiStepLR
```

```python
scheduler = MultiStepLR(
    optimizer,
    milestones=[150, 250],
    gamma=0.1,
)
```

これは、

- epoch 150 で 1 回下げる
- epoch 250 でさらに 1 回下げる

という意味。

`FashionMNIST + CNN` では、こちらの方が `StepLR(step_size=100)` より自然なことが多い。

## 5. CosineAnnealingLR も候補

後半を滑らかに詰めたいなら `CosineAnnealingLR` もよく使う。

```python
from torch.optim.lr_scheduler import CosineAnnealingLR
```

```python
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,
    eta_min=1e-5,
)
```

これは learning rate を急に落とさず、なめらかに下げていく方式。

今のように

- `lr=3e-4`
- `weight_decay=1e-5`
- 300 epoch 回す

という設定とは相性がよい。

最初に 1 本だけ試すなら、`MultiStepLR` か `CosineAnnealingLR` が有力。

## 6. 具体的な実装例

[src/train.py](/Users/suzukiakiramuki/projects/learning-llm/making-mlp/src/train.py:1) に `MultiStepLR` を入れる最小例はこうなる。

```python
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
```

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
)

scheduler = MultiStepLR(
    optimizer,
    milestones=[150, 250],
    gamma=0.1,
)
```

```python
for epoch in range(num_epochs):
    ...

    scheduler.step()
```

## 7. validation loss を見て下げたいなら ReduceLROnPlateau

epoch 数ではなく、`valid_loss` が改善しなくなったタイミングで下げたいなら `ReduceLROnPlateau` を使う。

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau
```

```python
scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.1,
    patience=10,
)
```

この scheduler だけは `scheduler.step()` の引数に監視値を渡す必要がある。

```python
for epoch in range(num_epochs):
    ...
    scheduler.step(valid_loss)
```

普通の `StepLR` や `MultiStepLR` と違って、`valid_loss` を見て動く点に注意。

## 8. ModelConstants に持たせる方法

このプロジェクトは [src/model/CNN/constants.py](/Users/suzukiakiramuki/projects/learning-llm/making-mlp/src/model/CNN/constants.py:1) のように、モデルごとの設定を dataclass で持っている。

そのため、scheduler 設定も定数側へ寄せると扱いやすい。

たとえばこうする。

```python
@dataclass(frozen=True)
class ModelConstants:
    NUM_EPOCHS: int = 300
    BATCHSIZE: int = 256
    L_LATE: float = 3 * 1e-4
    WEIGHT_DECAY: float = 1e-5
    SCHEDULER_NAME: str = "multistep"
    SCHEDULER_GAMMA: float = 0.1
    SCHEDULER_MILESTONES: tuple[int, ...] = (150, 250)
```

そして `train.py` 側で条件分岐する。

```python
if model_constants.SCHEDULER_NAME == "multistep":
    scheduler = MultiStepLR(
        optimizer,
        milestones=list(model_constants.SCHEDULER_MILESTONES),
        gamma=model_constants.SCHEDULER_GAMMA,
    )
else:
    scheduler = None
```

epoch の最後では `None` を考慮する。

```python
if scheduler is not None:
    scheduler.step()
```

この形にしておくと、`CNN` と `CNN-v2` で別設定を持たせやすい。

## 9. 何を最初に試すべきか

今のこのプロジェクトなら、最初は次の順がよい。

1. `MultiStepLR(milestones=[150, 250], gamma=0.1)`
2. `CosineAnnealingLR(T_max=num_epochs, eta_min=1e-5)`
3. `ReduceLROnPlateau(mode="min", factor=0.1, patience=10)`

理由は次の通り。

- `StepLR` は簡単だが節目が固定すぎる
- `MultiStepLR` は制御しやすい
- `CosineAnnealingLR` は滑らかで試しやすい
- `ReduceLROnPlateau` は便利だが、まずは挙動が読みやすいものから入れる方が切り分けしやすい

## 10. よくあるミス

1. `scheduler.step()` の位置をバッチごとにしてしまう
2. `ReduceLROnPlateau` なのに `valid_loss` を渡していない
3. learning rate を下げすぎて後半まったく進まなくなる
4. scheduler を入れたのに現在の learning rate を記録していない

基本は、まず epoch 単位 scheduler を 1 つ入れて挙動を見る方が安全。

## 11. まとめ

このプロジェクトで scheduler を入れるなら、

1. `optimizer` を作った直後に scheduler を作る
2. epoch の最後で `scheduler.step()` する
3. 最初は `MultiStepLR` か `CosineAnnealingLR` を試す
4. モデルごとの差を持たせたいなら `ModelConstants` に設定を寄せる

という流れで進めれば十分。
