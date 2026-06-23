import os
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torchvision import datasets


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    root: str
    num_classes: int
    in_channels: int
    image_size: tuple[int, int]
    mean: tuple[float, ...]
    std: tuple[float, ...]


DATASET_CONFIGS = {
    "FashionMNIST": DatasetConfig("FashionMNIST", "data", 10, 1, (28, 28), (0.5,), (0.5,)),
    "MNIST": DatasetConfig("MNIST", "data", 10, 1, (28, 28), (0.5,), (0.5,)),
    "KMNIST": DatasetConfig("KMNIST", "data", 10, 1, (28, 28), (0.5,), (0.5,)),
    "CIFAR10": DatasetConfig("CIFAR10", "data", 10, 3, (32, 32), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    "TinyImageNet": DatasetConfig(
        "TinyImageNet", "data/tiny-imagenet-200", 200, 3, (64, 64),
        (0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821),
    ),
}


def _build_torchvision(dataset_cls):
    def builder(config: DatasetConfig, train: bool, transform):
        return dataset_cls(
            root=config.root,
            train=train,
            download=True,
            transform=transform,
        )
    return builder


def _build_imagefolder(config: DatasetConfig, train: bool, transform):
    split = "train" if train else "val_organized"
    return datasets.ImageFolder(
        os.path.join(config.root, split),
        transform=transform,
    )


# 「名前 → 作り方(builder)」の登録テーブル。新データセットはここに1行足すだけ。
DATASET_BUILDERS = {
    "FashionMNIST": _build_torchvision(datasets.FashionMNIST),
    "MNIST": _build_torchvision(datasets.MNIST),
    "KMNIST": _build_torchvision(datasets.KMNIST),
    "CIFAR10": _build_torchvision(datasets.CIFAR10),
    "TinyImageNet": _build_imagefolder,
}


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    try:
        return DATASET_CONFIGS[dataset_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset: {dataset_name}") from exc


def build_dataset(dataset_name: str, train: bool, transform):
    dataset_config = get_dataset_config(dataset_name)

    try:
        builder = DATASET_BUILDERS[dataset_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset: {dataset_name}") from exc

    return builder(dataset_config, train, transform)


def build_loader(dataset, batch_size: int, train: bool, device: torch.device) -> DataLoader:
    use_cuda = device.type == "cuda"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=2,
        pin_memory=use_cuda,
    )


def build_dataloaders(
    dataset_name: str,
    batch_size: int,
    device: torch.device,
    train_transform,
    valid_transform,
) -> tuple[DataLoader, DataLoader]:
    train_ds = build_dataset(dataset_name, train=True, transform=train_transform)
    valid_ds = build_dataset(dataset_name, train=False, transform=valid_transform)
    return (
        build_loader(train_ds, batch_size, train=True, device=device),
        build_loader(valid_ds, batch_size, train=False, device=device),
    )


TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def _prepare_tiny_imagenet(config: DatasetConfig) -> None:
    """Tiny-ImageNet を DL・展開し、val を ImageFolder 形式へ再配置する（冪等）。"""
    root = config.root                          # data/tiny-imagenet-200
    data_dir = os.path.dirname(root) or "."     # data
    train_dir = os.path.join(root, "train")
    val_organized = os.path.join(root, "val_organized")

    if os.path.isdir(train_dir) and os.path.isdir(val_organized):
        return  # 既に準備済み

    os.makedirs(data_dir, exist_ok=True)

    if not os.path.isdir(root):
        zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")
        if not os.path.exists(zip_path):
            print("downloading tiny-imagenet-200.zip ...")
            urllib.request.urlretrieve(TINY_IMAGENET_URL, zip_path)
        print("extracting ...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(data_dir)

    if not os.path.isdir(val_organized):
        val_dir = os.path.join(root, "val")
        ann = os.path.join(val_dir, "val_annotations.txt")
        with open(ann) as f:
            mapping = {line.split("\t")[0]: line.split("\t")[1] for line in f}
        for fname, wnid in mapping.items():
            cls_dir = os.path.join(val_organized, wnid)
            os.makedirs(cls_dir, exist_ok=True)
            shutil.copy(
                os.path.join(val_dir, "images", fname),
                os.path.join(cls_dir, fname),
            )
        print(f"reorganized {len(mapping)} val images into {val_organized}")


# 「名前 → 準備処理」の登録テーブル。torchvision 系は download=True が担うので未登録。
DATASET_PREPARERS = {
    "TinyImageNet": _prepare_tiny_imagenet,
}


def ensure_dataset(dataset_name: str) -> None:
    """学習前にデータが無ければ取得・整形する。準備不要なデータセットでは何もしない。"""
    config = get_dataset_config(dataset_name)
    preparer = DATASET_PREPARERS.get(dataset_name)
    if preparer is not None:
        preparer(config)
