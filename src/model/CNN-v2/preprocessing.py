
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .constants import MODEL_CONSTANTS

@dataclass(frozen=True)
class DatasetConfig:
    name: str
    root: str
    num_classes: int
    in_channels: int
    image_size: tuple[int, int]
    mean: tuple
    std: tuple


DATASET_CONFIGS = {
    "FashionMNIST": DatasetConfig(
        name="FashionMNIST",
        root="data",
        num_classes=10,
        in_channels=1,
        image_size=(28, 28),
        mean=(0.5,),
        std=(0.5,),
    ),
    "MNIST": DatasetConfig(
        name="MNIST",
        root="data",
        num_classes=10,
        in_channels=1,
        image_size=(28, 28),
        mean=(0.5,),
        std=(0.5,),
    ),
    "CIFAR10": DatasetConfig(
        name="CIFAR10",
        root="data",
        num_classes=10,
        in_channels=3,
        image_size=(32, 32),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
}


def build_dataset(dataset_name: str, train: bool, transform):
    dataset_map = {
        "FashionMNIST": datasets.FashionMNIST,
        "MNIST": datasets.MNIST,
        "CIFAR10": datasets.CIFAR10,
    }
    dataset_cls = dataset_map[dataset_name]
    dataset_config = DATASET_CONFIGS[dataset_name]
    return dataset_cls(
        root=dataset_config.root,
        train=train,
        download=True,
        transform=transform,
    )


def build_loader(dataset, batch_size: int, train: bool, device: torch.device):
    use_cuda = device.type == "cuda"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=2,
        pin_memory=use_cuda,
    )


def build_transform(dataset_name: str, train: bool) -> transforms.Compose:
    dataset_config = DATASET_CONFIGS[dataset_name]
    transform_steps = []

    if train:
        transform_steps.extend(
            [
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            ]
        )

    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(dataset_config.mean, dataset_config.std),
        ]
    )

    return transforms.Compose(transform_steps)


def build_common_dataloaders(dataset_name: str, batch_size: int, device: torch.device):
    train_transform = build_transform(dataset_name, train=True)
    valid_transform = build_transform(dataset_name, train=False)
    train_ds = build_dataset(dataset_name, True, train_transform)
    valid_ds = build_dataset(dataset_name, False, valid_transform)
    return (
        build_loader(train_ds, batch_size, True, device),
        build_loader(valid_ds, batch_size, False, device),
    )

def build_dataloaders(device):
    return build_common_dataloaders(
        dataset_name="FashionMNIST",
        batch_size=MODEL_CONSTANTS.BATCHSIZE,
        device=device,
    )
