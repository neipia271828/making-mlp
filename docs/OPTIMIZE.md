# PyTorch で学習速度を向上させる方法

このプロジェクトで MLP 学習を速くするときの基本をまとめる。

## 1. まず確認するべきこと

学習速度は大きく分けて次の 3 つで決まる。

1. モデル計算そのもの
2. データ読み込みの速さ
3. CPU から GPU への転送コスト

FashionMNIST のような小さい画像を MLP で学習する場合は、モデル計算よりも周辺のオーバーヘッドが効くことがある。

特に Apple Silicon の `mps` や小規模 GPU 学習では、GPU を使っても劇的に速くならないことがある。

## 2. device を正しく使う

PyTorch は自動で GPU を使わないので、まず `device` を明示する。

```python
import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
```

モデルとミニバッチを同じ `device` に載せる。

```python
model = MLP(784, 128, 10).to(device)

for x_batch, y_batch in train_loader:
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
```

CPU のまま学習していると、当然ここが最初のボトルネックになる。

## 3. DataLoader を調整する

小規模データでも `DataLoader` の設定で待ち時間が減ることがある。

```python
train_loader = DataLoader(
    train_ds,
    batch_size=128,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)

valid_loader = DataLoader(
    valid_ds,
    batch_size=128,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)
```

各引数の意味は次の通り。

- `batch_size`: 大きくすると 1 epoch あたりの反復回数が減る
- `num_workers`: データ準備を並列化する
- `pin_memory`: CPU から GPU への転送を効率化しやすい

ただし、`mps` 環境では `pin_memory=True` は現状サポートされておらず、警告が出ることがある。

そのため、`pin_memory` は `cuda` のときだけ有効にする方が安全。

```python
use_cuda = device.type == "cuda"

train_loader = DataLoader(
    train_ds,
    batch_size=128,
    shuffle=True,
    num_workers=2,
    pin_memory=use_cuda,
)
```

また、データセットが小さい場合は `num_workers` を増やしすぎるとかえって遅くなることもある。

最初は `0`, `2`, `4` を試し、速いものを選ぶのが実務的。

また、`num_workers > 0` を使う場合は、`DataLoader` に渡す前処理が pickle 可能である必要がある。

たとえば次のような `lambda` は環境によって worker 起動時に失敗することがある。

```python
transforms.Lambda(lambda x: x.view(-1))
```

その場合は、トップレベル関数に切り出す。

```python
def flatten_tensor(x):
    return x.view(-1)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(flatten_tensor),
])
```

さらに、学習処理全体は `if __name__ == "__main__":` の中から呼ぶ方がよい。

## 4. 転送時に non_blocking を使う

`pin_memory=True` を使うなら、テンソル転送側も合わせる。

```python
for x_batch, y_batch in train_loader:
    x_batch = x_batch.to(device, non_blocking=True)
    y_batch = y_batch.to(device, non_blocking=True)
```

これにより、環境によっては転送待ちを減らせる。

ただし、効果は主に `cuda` で出やすく、`mps` では差が小さいこともある。

## 5. batch_size を見直す

学習速度を上げたいとき、最初に試す価値が高いのは `batch_size` の調整。

今が `64` なら、次を試せる。

```python
batch_size=128
```

あるいは

```python
batch_size=256
```

バッチを大きくすると、1 epoch あたりのループ回数が減るので速くなりやすい。

ただし、注意点もある。

- メモリ使用量が増える
- 大きくしすぎると精度が落ちることがある
- 環境によっては最適値が違う

まずは `64`, `128`, `256` を比較すれば十分。

## 6. epoch 数を目的に合わせて減らす

学習速度そのものではないが、総学習時間を減らすには `num_epochs` の見直しも有効。

FashionMNIST の MLP では、後半の epoch で精度があまり伸びないことがある。

たとえば validation accuracy が早い段階で頭打ちなら、`50` epoch 固定にしなくてよい。

```python
num_epochs = 20
```

あるいは、途中で改善が止まったら打ち切る方法もある。

これは特に試行錯誤の段階で効く。

## 7. モデルを必要以上に大きくしない

大きいモデルは表現力が上がる一方で、計算時間も増える。

たとえば `hidden_dim` をむやみに大きくすると、速さは落ちやすい。

```python
model = MLP(784, 128, 10).to(device)
```

まずは小さめの構成で学習フローを固め、必要になったら `256` や `512` に広げる方が効率的。

## 8. 可視化は学習ループの外に置く

`matplotlib` の描画は学習後に 1 回だけ行う。

各 epoch ごとにグラフ描画や画像保存を入れると、その分だけ遅くなる。

今のように最後にまとめて `plt.show()` する形なら問題ない。

## 9. このプロジェクト向けの実用的な改善順

このプロジェクトでは、まず次の順で試すのがよい。

1. `batch_size=128` に上げる
2. `num_workers=2` を試す
3. `pin_memory=True` を付ける
4. `to(device, non_blocking=True)` を試す
5. `num_epochs` を減らしても十分か確認する

例:

```python
train_loader = DataLoader(
    train_ds,
    batch_size=128,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)

valid_loader = DataLoader(
    valid_ds,
    batch_size=128,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

for x_batch, y_batch in train_loader:
    x_batch = x_batch.to(device, non_blocking=True)
    y_batch = y_batch.to(device, non_blocking=True)
```

## 10. 覚えておくべきこと

FashionMNIST + 小規模 MLP では、学習速度の改善余地はあるが、劇的な差が出ないことも多い。

その場合は、実行時間の多くがモデル計算ではなく、起動コストやデータ供給側にある。

速さだけでなく、1 epoch あたりの時間と最終精度のバランスで判断するのが実務的。
