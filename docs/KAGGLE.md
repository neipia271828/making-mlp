# Kaggle コンペティション参加ガイド

## 1. アカウント作成

1. [kaggle.com](https://www.kaggle.com) にアクセス
2. "Register" → Google/メールで登録
3. プロフィールページで **電話番号認証**（コンペ参加・データDLに必要）

---

## 2. コンペティションの探し方

[kaggle.com/competitions](https://www.kaggle.com/competitions) にアクセス。

| フィルター | 内容 |
|---|---|
| "Getting Started" | 初心者向け常設コンペ（Titanic, MNIST など） |
| "Featured" | 賞金付きの主要コンペ |
| "Research" | 研究目的 |
| "Playground" | 練習用、評価期間あり |

初参加は **Getting Started** の **Digit Recognizer**（MNIST の手書き数字）がおすすめ。このプロジェクトのモデルをそのまま活用できる。

---

## 3. Kaggle API のセットアップ

コマンドラインからデータのダウンロードや提出ができる。

### インストール

```bash
pip install kaggle
```

### 認証トークンの取得

1. kaggle.com → 右上アイコン → **Account**
2. **API** セクション → "Create New Token" をクリック
3. `kaggle.json` がダウンロードされる
4. 以下の場所に配置：

```bash
# macOS / Linux
mkdir -p ~/.config/kaggle
mv ~/Downloads/kaggle.json ~/.config/kaggle/kaggle.json
chmod 600 ~/.config/kaggle/kaggle.json
```

### 動作確認

```bash
kaggle competitions list
```

---

## 4. データのダウンロード

```bash
# コンペのデータ一覧を確認
kaggle competitions files digit-recognizer

# ダウンロード（カレントディレクトリに展開）
kaggle competitions download -c digit-recognizer -p data/Kaggle/digit-recognizer
unzip data/Kaggle/digit-recognizer/digit-recognizer.zip -d data/Kaggle/digit-recognizer
```

データ構成（例：Digit Recognizer）:
```
data/Kaggle/digit-recognizer/
├── train.csv      ← 学習データ（label + pixel 値）
├── test.csv       ← 提出用テストデータ（label なし）
└── sample_submission.csv  ← 提出フォーマットのサンプル
```

---

## 5. 提出ファイルの作り方

### 提出フォーマットの確認

```python
import pandas as pd
sample = pd.read_csv("data/Kaggle/digit-recognizer/sample_submission.csv")
print(sample.head())
# ImageId,Label
# 1,0
# 2,0
```

### 推論 → 提出ファイル生成の流れ

```python
import torch
import pandas as pd

# テストデータ読み込み（CSV → テンソル変換）
test_df = pd.read_csv("data/Kaggle/digit-recognizer/test.csv")
test_pixels = test_df.values  # shape: (28000, 784)
x_test = torch.tensor(test_pixels, dtype=torch.float32) / 255.0
x_test = x_test.view(-1, 1, 28, 28)  # CNN 用に reshape

# モデルをロードして推論
model.eval()
with torch.no_grad():
    logits = model(x_test)
    preds = logits.argmax(dim=1).numpy()

# 提出ファイル作成
submission = pd.DataFrame({
    "ImageId": range(1, len(preds) + 1),
    "Label": preds
})
submission.to_csv("submission.csv", index=False)
```

---

## 6. 提出方法

### Web UI から提出

1. コンペページ → **Submit Predictions**
2. `submission.csv` をアップロード
3. スコアが表示される（Public Leaderboard）

### API から提出

```bash
kaggle competitions submit -c digit-recognizer -f submission.csv -m "CNN-v3 baseline"
```

### 提出履歴の確認

```bash
kaggle competitions submissions digit-recognizer
```

---

## 7. スコアの見方

| 項目 | 説明 |
|---|---|
| **Public LB** | テストデータの一部で評価（コンペ期間中に見える） |
| **Private LB** | テストデータの残りで評価（コンペ終了後に確定） |

Public LB に過学習しないよう注意（"shake-up" と呼ばれる順位変動が起きやすい）。

---

## 8. このプロジェクトのモデルを使う場合

Digit Recognizer（MNIST）には **MLP** か **CNN** モデルをそのまま使える。

```
CONSTANTS.py の PROJECT を "Kaggle-DigitRecognizer" などに変更し、
preprocessing.py でデータ読み込みを CSV 対応に修正する。
```

CIFAR10 コンペには **CNN-v3 / CNN-v4** が活用できる。

---

## 参考リンク

- [Kaggle Learn（無料チュートリアル）](https://www.kaggle.com/learn)
- [Digit Recognizer コンペ](https://www.kaggle.com/c/digit-recognizer)
- [Kaggle API ドキュメント](https://github.com/Kaggle/kaggle-api)
