"""Tiny-ImageNet-200 を手動準備するための CLI エントリ。

学習時は train.py が ensure_dataset() で自動準備するため通常は不要だが、
事前に DL/整形だけ済ませたい場合（例: サーバー側で先に用意）に使う。

使い方:
    python scripts/prepare_tiny_imagenet.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lib.dataset_builder import ensure_dataset  # noqa: E402


if __name__ == "__main__":
    ensure_dataset("TinyImageNet")
