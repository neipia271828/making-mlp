# AI の評価指標を記録する方法

このプロジェクトで学習結果を記録するときによく使う評価指標をまとめる。

対象は次の 6 つ。

1. `train_loss`
2. `validation_loss`
3. `accuracy`
4. `precision`
5. `recall`
6. `f1_score`

`F1-score` という表記もよく使うが、コードや CSV の列名では `f1_score` のように英数字と `_` でそろえる方が扱いやすい。

## 1. 何をどこへ記録するか

このプロジェクトでは、記録先を 2 つに分けると扱いやすい。

1. 各 epoch の推移を保存する `train_logs.csv`
2. 学習 1 回ごとの最終結果を保存する `data/{PROJECT}/logs/{PROJECT}.csv`

使い分けは次の通り。

1. `train_logs.csv`
   各 epoch の `train_loss`, `validation_loss`, `accuracy`, `precision`, `recall`, `f1_score`
2. summary CSV
   最終 epoch の値、総学習時間、設定値、モデル情報

## 2. 各指標の意味

### `train_loss`

学習データに対する損失。
各 batch の loss を足し上げるか、平均して記録する。

### `validation_loss`

検証データに対する損失。
学習中の更新はせず、`model.eval()` と `torch.no_grad()` の中で計算する。

### `accuracy`

全予測のうち、正解した割合。

```python
accuracy = correct / total
```

### `precision`

陽性と予測したもののうち、実際に正しかった割合。

```python
precision = tp / (tp + fp)
```

### `recall`

実際に陽性だったもののうち、正しく陽性と判定できた割合。

```python
recall = tp / (tp + fn)
```

### `f1_score`

`precision` と `recall` の調和平均。

```python
f1_score = 2 * precision * recall / (precision + recall)
```

## 3. まず validation 側で必要な値を集める

今の `src/train.py` は validation で `accuracy` だけを計算している。
追加指標を記録するには、validation ループで次を集める。

1. `validation_loss` 用の loss 合計
2. 正解ラベル `y_batch`
3. 予測ラベル `pred`

単一ラベル多クラス分類なら、まずは micro 平均で記録するのが簡単。

```python
valid_loss_total = 0.0
correct = 0
total = 0
tp = 0
fp = 0
fn = 0

model.eval()
with torch.no_grad():
    for x_batch, y_batch in valid_loader:
        x_batch = x_batch.to(device, non_blocking=use_cuda)
        y_batch = y_batch.to(device, non_blocking=use_cuda)

        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        pred = logits.argmax(dim=1)

        valid_loss_total += loss.item()
        correct += (pred == y_batch).sum().item()
        total += y_batch.size(0)

        # 単一ラベル分類で micro 平均を取るなら、
        # 各サンプルについて pred == y が TP、
        # pred != y は誤分類として数える。
        tp += (pred == y_batch).sum().item()
        fp += (pred != y_batch).sum().item()
        fn += (pred != y_batch).sum().item()
```

この micro 平均では、単一ラベル多クラス分類なら `precision` と `recall` と `f1_score` は `accuracy` と同じ値になる。

そのため、FashionMNIST のような単一ラベル多クラス分類で各指標を分けて見たい場合は、macro 平均を使う方が意味がある。

## 4. macro 平均で `precision` / `recall` / `f1_score` を出す

クラスごとの差を見たいなら、各クラスごとに `tp`, `fp`, `fn` を数えて平均する。

```python
num_classes = 10
tp = torch.zeros(num_classes, dtype=torch.float32)
fp = torch.zeros(num_classes, dtype=torch.float32)
fn = torch.zeros(num_classes, dtype=torch.float32)

model.eval()
with torch.no_grad():
    for x_batch, y_batch in valid_loader:
        x_batch = x_batch.to(device, non_blocking=use_cuda)
        y_batch = y_batch.to(device, non_blocking=use_cuda)

        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        pred = logits.argmax(dim=1)

        valid_loss_total += loss.item()
        correct += (pred == y_batch).sum().item()
        total += y_batch.size(0)

        for class_index in range(num_classes):
            pred_is_class = pred == class_index
            true_is_class = y_batch == class_index

            tp[class_index] += (pred_is_class & true_is_class).sum().item()
            fp[class_index] += (pred_is_class & ~true_is_class).sum().item()
            fn[class_index] += (~pred_is_class & true_is_class).sum().item()
```

集計後に各指標を計算する。

```python
validation_loss = valid_loss_total / len(valid_loader)
accuracy = correct / total

precision_per_class = tp / (tp + fp).clamp(min=1)
recall_per_class = tp / (tp + fn).clamp(min=1)
f1_per_class = 2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class).clamp(min=1e-12)

precision = precision_per_class.mean().item()
recall = recall_per_class.mean().item()
f1_score = f1_per_class.mean().item()
```

`clamp(min=1)` や `clamp(min=1e-12)` は、0 除算を避けるために入れている。

## 5. epoch ごとのログへ保存する

今の `src/lib/log_maker.py` では `train_logs.csv` のヘッダが次になっている。

```python
["ep_time", "epoch", "train_loss", "valid_accuracy"]
```

指標を増やすなら、たとえば次の形に広げる。

```python
[
    "ep_time",
    "epoch",
    "train_loss",
    "validation_loss",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
]
```

`src/train.py` 側の `logs.append(...)` も同じ順にそろえる。

```python
logs.append([
    ep_time,
    epoch + 1,
    total_loss,
    validation_loss,
    accuracy,
    precision,
    recall,
    f1_score,
])
```

この形なら、後で `docs/GRAPH.md` の手順で各列をそのまま可視化しやすい。

## 6. summary CSV に最終値を保存する

`src/train.py` では最後に `make_summalize_csv(...)` へ metrics を渡している。
ここへ最終 epoch の評価値を追加すれば、学習 1 回ごとの結果比較に使える。

```python
make_summalize_csv(
    meta_constants.PROJECT,
    meta_constants,
    model_constants,
    {
        "time_stamp": timestamp,
        "req_time": req_time,
        "ep_time": time_per_ep,
        "train_loss": total_loss,
        "validation_loss": validation_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    },
)
```

`make_summalize_csv()` は動的に列を追加できるので、新しいキーを増やせば summary CSV の列も拡張できる。

## 7. 実装イメージ

学習ループ全体では、validation 部分を次のように整理すると分かりやすい。

```python
train_losses = []
validation_losses = []
valid_accuracies = []
valid_precisions = []
valid_recalls = []
valid_f1_scores = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device, non_blocking=use_cuda)
        y_batch = y_batch.to(device, non_blocking=use_cuda)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)
    train_losses.append(train_loss)

    valid_loss_total = 0.0
    correct = 0
    total = 0
    num_classes = 10
    tp = torch.zeros(num_classes, dtype=torch.float32)
    fp = torch.zeros(num_classes, dtype=torch.float32)
    fn = torch.zeros(num_classes, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in valid_loader:
            x_batch = x_batch.to(device, non_blocking=use_cuda)
            y_batch = y_batch.to(device, non_blocking=use_cuda)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            pred = logits.argmax(dim=1)

            valid_loss_total += loss.item()
            correct += (pred == y_batch).sum().item()
            total += y_batch.size(0)

            for class_index in range(num_classes):
                pred_is_class = pred == class_index
                true_is_class = y_batch == class_index
                tp[class_index] += (pred_is_class & true_is_class).sum().item()
                fp[class_index] += (pred_is_class & ~true_is_class).sum().item()
                fn[class_index] += (~pred_is_class & true_is_class).sum().item()

    validation_loss = valid_loss_total / len(valid_loader)
    accuracy = correct / total
    precision_per_class = tp / (tp + fp).clamp(min=1)
    recall_per_class = tp / (tp + fn).clamp(min=1)
    f1_per_class = 2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class).clamp(min=1e-12)

    precision = precision_per_class.mean().item()
    recall = recall_per_class.mean().item()
    f1_score = f1_per_class.mean().item()

    validation_losses.append(validation_loss)
    valid_accuracies.append(accuracy)
    valid_precisions.append(precision)
    valid_recalls.append(recall)
    valid_f1_scores.append(f1_score)
```

## 8. 記録時の注意

### `train_loss` は合計より平均の方が比較しやすい

今の `src/train.py` では `total_loss` をそのまま記録している。
ただし batch 数が変わると値のスケールも変わるので、比較目的なら次の形の方が見やすい。

```python
train_loss = total_loss / len(train_loader)
validation_loss = valid_loss_total / len(valid_loader)
```

### `accuracy` だけでは偏りを見落とす

クラス不均衡があるデータでは、`accuracy` が高くても一部クラスをほとんど当てていないことがある。
そのときは `precision`、`recall`、`f1_score` も一緒に見る。

### 列名は固定する

CSV の列名は途中で `F1-score` と `f1_score` が混ざると扱いづらい。
最初から次のように固定した方がよい。

1. `train_loss`
2. `validation_loss`
3. `accuracy`
4. `precision`
5. `recall`
6. `f1_score`

## 9. まとめ

このプロジェクトで評価指標を記録するなら、次の流れにすると整理しやすい。

1. 学習中に `train_loss` を記録する
2. validation で `validation_loss`, `accuracy`, `precision`, `recall`, `f1_score` を計算する
3. 各 epoch の値を `train_logs.csv` に保存する
4. 最終値を summary CSV に保存する
5. `docs/GRAPH.md` の方法で可視化する

まずは `train_loss`、`validation_loss`、`accuracy` を安定して取り、その後に `precision`、`recall`、`f1_score` を足すと実装しやすい。
