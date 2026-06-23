"""Tiny-ImageNet-200 をダウンロードし、val を ImageFolder 形式へ再配置する。

使い方:
    python scripts/prepare_tiny_imagenet.py

実行後の構造:
    data/tiny-imagenet-200/train/<wnid>/images/*.JPEG   (DL時点の構造のまま)
    data/tiny-imagenet-200/val_organized/<wnid>/*.JPEG  (本スクリプトが生成)
"""

import os
import shutil
import urllib.request
import zipfile

URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
DATA_DIR = "data"
ROOT = os.path.join(DATA_DIR, "tiny-imagenet-200")
ZIP_PATH = os.path.join(DATA_DIR, "tiny-imagenet-200.zip")


def download_and_extract() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(ROOT):
        print(f"{ROOT} already exists, skip download")
        return
    if not os.path.exists(ZIP_PATH):
        print("downloading tiny-imagenet-200.zip ...")
        urllib.request.urlretrieve(URL, ZIP_PATH)
    print("extracting ...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(DATA_DIR)


def reorganize_val() -> None:
    val_dir = os.path.join(ROOT, "val")
    out_dir = os.path.join(ROOT, "val_organized")
    if os.path.exists(out_dir):
        print(f"{out_dir} already exists, skip reorganize")
        return

    ann = os.path.join(val_dir, "val_annotations.txt")
    with open(ann) as f:
        mapping = {line.split("\t")[0]: line.split("\t")[1] for line in f}

    for fname, wnid in mapping.items():
        cls_dir = os.path.join(out_dir, wnid)
        os.makedirs(cls_dir, exist_ok=True)
        shutil.copy(
            os.path.join(val_dir, "images", fname),
            os.path.join(cls_dir, fname),
        )
    print(f"reorganized {len(mapping)} val images into {out_dir}")


if __name__ == "__main__":
    download_and_extract()
    reorganize_val()
