# matplotlib でグラフを出力する方法

このプロジェクトで学習結果を `matplotlib` で可視化する基本をまとめる。

## 1. 何をグラフにするか

学習でまず見たいのは、各 epoch ごとの次の値。

1. `train_loss`
2. `valid_accuracy`

この 2 つを並べて見ると、学習が進んでいるか、過学習気味かを把握しやすい。

## 2. まず値をリストにためる

`matplotlib` は、描画前に `x` と `y` の配列を持っている必要がある。

このプロジェクトでは、学習中に次のようなリストへ記録する形が自然。

```python
train_losses = []
valid_accuracies = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for x_batch, y_batch in train_loader:
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

`x` 軸には通常 epoch 番号を使う。

```python
epochs = range(1, num_epochs + 1)
```

## 3. matplotlib を import する

基本形はこれ。

```python
import matplotlib.pyplot as plt
```

`pyplot` を `plt` という名前で使うのが定番。

## 4. 1 枚の図を作る

まず図全体のサイズを決める。

```python
plt.figure(figsize=(10, 4))
```

ここでは横 10、縦 4 の図を作っている。

## 5. 折れ線グラフを描く

学習 loss を描く最小例はこれ。

```python
plt.plot(epochs, train_losses, marker="o")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
```

`marker="o"` を付けると、各 epoch の点が丸で表示される。

validation accuracy も同じ考え方で描ける。

```python
plt.plot(epochs, valid_accuracies, marker="o")
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
```

## 6. 2 つのグラフを横に並べる

このプロジェクトでは、loss と accuracy を 1 枚の図に横並びで描いている。

```python
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
```

`plt.subplot(1, 2, 1)` は、

1. 1 行
2. 2 列
3. 1 番目の領域

を意味する。

`plt.subplot(1, 2, 2)` は右側のグラフ領域。

## 7. レイアウトを整える

ラベルやタイトルが重ならないようにするにはこれを入れる。

```python
plt.tight_layout()
```

複数グラフを並べるときは入れておく方がよい。

## 8. 画像として保存する

このプロジェクトでは、グラフを `svg` で保存している。

```python
from pathlib import Path

graph_dir = Path(f"data/{PROJECT}/logs") / timestamp
graph_dir.mkdir(parents=True, exist_ok=True)

graph_path = graph_dir / "train_graph.svg"
plt.savefig(graph_path)
```

これで保存先は次のようになる。

```text
data/FashionMNIST/logs/2026-06-01-17-30/train_graph.svg
```

`mkdir(parents=True, exist_ok=True)` を入れておくと、親ディレクトリがなくても保存できる。

## 9. 画面にも表示する

保存だけでなく、実行時に画面へ表示したいならこれを呼ぶ。

```python
plt.show()
```

ただし、実行環境によっては `plt.show()` がウィンドウ待ちになり、処理が止まって見えることがある。

自動実行だけを重視するなら、保存だけして `show()` を外す選択もある。

## 10. このプロジェクトの実装例

今の `src/lib/graph_drawer.py` に近い形は次の通り。

```python
import matplotlib.pyplot as plt
from pathlib import Path

def graph_drawer(PROJECT, epochs, timestamp, train_losses, valid_accuracies):
    graph_dir = Path(f"data/{PROJECT}/logs") / timestamp
    graph_dir.mkdir(parents=True, exist_ok=True)

    graph_path = graph_dir / "train_graph.svg"

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
    plt.savefig(graph_path)
    plt.show()
```

## 11. よくある詰まりどころ

この用途で起こりやすい問題は次の通り。

1. `epochs` と `train_losses` の長さが合っていない
2. 保存先ディレクトリが存在しない
3. `plt.show()` で処理が止まったように見える
4. `savefig()` の保存先拡張子が意図と違う

特に自動処理では、まず `savefig()` だけで保存確認する方が切り分けしやすい。

## 12. まとめ

このプロジェクトで `matplotlib` を使うなら、

1. epoch ごとの値をリストにためる
2. `plt.plot()` で描く
3. `plt.savefig()` で `svg` 保存する

の流れで十分。

学習結果の確認なら、`Training Loss` と `Validation Accuracy` を横並びで出す形が扱いやすい。

## 13. CSV のデータをグラフとして表示する

保存済みの CSV から後でグラフを描きたい場合は、`pandas` で読み込んで `matplotlib` に渡すのが簡単。

まず CSV を読む。

```python
import pandas as pd

df = pd.read_csv("data/FashionMNIST/logs/FashionMNIST.csv")
print(df.head())
print(df.columns)
```

`df.columns` を見ると、今の CSV にどの列が入っているか確認できる。

### accuracy を描く

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/FashionMNIST/logs/FashionMNIST.csv")

plt.plot(df["accuracy"], marker="o")
plt.title("Accuracy")
plt.xlabel("Row")
plt.ylabel("Accuracy")
plt.show()
```

### time_stamp を x 軸にして描く

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/FashionMNIST/logs/FashionMNIST.csv")

plt.figure(figsize=(10, 4))
plt.plot(df["time_stamp"], df["accuracy"], marker="o")
plt.title("Accuracy")
plt.xlabel("Time")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

ラベルが長いので、`plt.xticks(rotation=45)` で少し回すと見やすい。

### loss と accuracy を並べて描く

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/FashionMNIST/logs/FashionMNIST.csv")

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(df["time_stamp"], df["total_loss"], marker="o")
plt.title("Total Loss")
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
plt.plot(df["time_stamp"], df["accuracy"], marker="o")
plt.title("Accuracy")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
```

### 画像として保存する

画面表示だけでなく保存したいなら、`show()` の代わりに `savefig()` を使う。

```python
plt.savefig("graph.svg")
```

### このプロジェクトでの注意点

CSV の列構成は変更途中だと古い列が混ざることがある。

そのため、最初に次を確認してから描く方が安全。

```python
print(df.columns)
```

特に `accuracy`, `total_loss`, `time_stamp` の列名が期待どおりかは先に見た方がよい。

## 14. 同じフレームに複数の折れ線グラフを載せる

同じグラフ領域に複数の線を描きたいときは、同じ `Axes` に対して `plot()` を複数回呼べばよい。

```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
y1 = [0.5, 0.6, 0.7, 0.8]
y2 = [0.4, 0.55, 0.65, 0.75]

plt.plot(x, y1, marker="o", label="model A")
plt.plot(x, y2, marker="s", label="model B")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison")
plt.legend()
plt.show()
```

`label=` を付けて `plt.legend()` を呼ぶと、どの線が何を表すか分かりやすくなる。

### CSV の複数列を同じフレームへ描く

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/FashionMNIST/logs/FashionMNIST.csv")

plt.figure(figsize=(10, 4))
plt.plot(df["time_stamp"], df["accuracy"], marker="o", label="accuracy")
plt.plot(df["time_stamp"], df["total_loss"], marker="s", label="total_loss")

plt.xticks(rotation=45)
plt.title("Metrics")
plt.legend()
plt.tight_layout()
plt.show()
```

## 15. 値のスケールが違う場合は左右の軸を分ける

`accuracy` と `total_loss` のように値のスケールがかなり違う場合、同じ `y` 軸だと片方が見づらくなることがある。

その場合は `twinx()` を使って左右で軸を分ける。

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/FashionMNIST/logs/FashionMNIST.csv")

fig, ax1 = plt.subplots(figsize=(10, 4))
ax2 = ax1.twinx()

ax1.plot(df["time_stamp"], df["accuracy"], color="blue", marker="o", label="accuracy")
ax2.plot(df["time_stamp"], df["total_loss"], color="red", marker="s", label="total_loss")

ax1.set_xlabel("Time")
ax1.set_ylabel("Accuracy", color="blue")
ax2.set_ylabel("Loss", color="red")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

この方法なら、片方が `0.8` 前後でも、もう片方が `100` を超えるような値でも見やすく比較できる。
