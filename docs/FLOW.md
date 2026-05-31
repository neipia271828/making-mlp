# PyTorch で MLP を構築する手順

このプロジェクトで MLP を作るときの基本フローをまとめる。

## 1. 環境を準備する

まず PyTorch を入れる。`pyproject.toml` にはまだ依存がないので、必要なら追加する。

```bash
uv add torch
```

GPU を使う場合は、環境に合った PyTorch のインストール方法を選ぶ。

### device を指定する

PyTorch は自動では GPU を使わないので、`device` を明示してモデルとテンソルを載せる。

```python
import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(device)
```

Apple Silicon Mac では `mps`、NVIDIA GPU 環境では `cuda`、どちらも使えなければ `cpu` を使う。

## 2. データを用意する

MLP は画像なら 1 次元ベクトル、表形式データなら数値特徴量を入力にする。

典型的には次を行う。

1. データを読み込む
2. 学習用と検証用に分ける
3. 正規化する
4. `Dataset` と `DataLoader` に載せる

```python
from torch.utils.data import DataLoader, TensorDataset

train_ds = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
```

### torchvision からデータセットを取得する

MNIST や FashionMNIST のような定番データセットは `torchvision.datasets` から取得できる。

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
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

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=64, shuffle=False)
```

`download=True` にすると、まだローカルに存在しないときだけ `data/` 配下へ自動で保存される。

画像系データセットで MLP を使うときは、`28 x 28` のような 2 次元画像を `784` 次元ベクトルへ平坦化してから `Linear` 層へ入れる。

## 3. モデルを定義する

MLP は `Linear` 層と活性化関数を並べる。

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)
```

入力テンソルは通常 `[batch_size, input_dim]` の形にする。

モデルを作ったら `device` へ移す。

```python
model = MLP(input_dim=784, hidden_dim=128, output_dim=10).to(device)
```

## 4. 損失関数と最適化手法を決める

分類なら `CrossEntropyLoss`、回帰なら `MSELoss` を使うことが多い。

```python
import torch.nn as nn
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
```

## 5. 学習ループを書く

学習の基本は以下の流れ。

1. `model.train()` を呼ぶ
2. 入力をモデルへ渡す
3. 損失を計算する
4. `zero_grad()` で勾配を消す
5. `backward()` で逆伝播する
6. `step()` で更新する

```python
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"epoch={epoch+1}, loss={total_loss:.4f}")
```

## 6. 評価する

検証時は `model.eval()` と `torch.no_grad()` を使う。

```python
correct = 0
total = 0

model.eval()
with torch.no_grad():
    for x_batch, y_batch in valid_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        logits = model(x_batch)
        pred = logits.argmax(dim=1)
        correct += (pred == y_batch).sum().item()
        total += y_batch.size(0)

accuracy = correct / total
print(f"accuracy={accuracy:.4f}")
```

`next(model.parameters()).device` を見ると、モデルが今どの `device` に載っているか確認できる。

## 7. 学習結果をグラフで表示する

loss や accuracy の推移を見ると、学習が進んでいるか、過学習していないかを確認しやすい。

まず各 epoch の値を記録する。

```python
train_losses = []
valid_accuracies = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_losses.append(total_loss)

    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in valid_loader:
            logits = model(x_batch)
            pred = logits.argmax(dim=1)
            correct += (pred == y_batch).sum().item()
            total += y_batch.size(0)

    valid_accuracies.append(correct / total)
```

次に `matplotlib` で可視化する。

```bash
uv add matplotlib
```

```python
import matplotlib.pyplot as plt

epochs = range(1, num_epochs + 1)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, marker="o")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(epochs, valid_accuracies, marker="o")
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.tight_layout()
plt.show()
```

`loss` が下がり、`accuracy` が上がっていれば、少なくとも学習は進んでいると判断しやすい。

## 8. 保存と再利用

重みは `state_dict` で保存する。

```python
torch.save(model.state_dict(), "mlp.pt")
```

読み込み側では同じ構造のモデルを作ってから重みを戻す。

```python
model = MLP(input_dim=784, hidden_dim=256, output_dim=10)
model.load_state_dict(torch.load("mlp.pt", map_location="cpu"))
model.eval()
```

## 9. 最小チェックリスト

- 入力の shape が `Linear` に合っているか
- `CrossEntropyLoss` のとき、ラベルは one-hot ではなくクラス ID か
- 学習時に `model.train()`、評価時に `model.eval()` を使っているか
- `optimizer.zero_grad()` を忘れていないか
- 保存した重みと同じ構造で読み込んでいるか
