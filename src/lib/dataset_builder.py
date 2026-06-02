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
}

DATASET_CLASSES = {
    "FashionMNIST": datasets.FashionMNIST,
    "MNIST": datasets.MNIST,
    "KMNIST": datasets.KMNIST,
    "CIFAR10": datasets.CIFAR10,
}


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    try:
        return DATASET_CONFIGS[dataset_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset: {dataset_name}") from exc


def build_dataset(dataset_name: str, train: bool, transform):
    dataset_config = get_dataset_config(dataset_name)

    try:
        dataset_cls = DATASET_CLASSES[dataset_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset: {dataset_name}") from exc

    return dataset_cls(
        root=dataset_config.root,
        train=train,
        download=True,
        transform=transform,
    )


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
