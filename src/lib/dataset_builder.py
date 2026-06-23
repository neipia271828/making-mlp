import os
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
