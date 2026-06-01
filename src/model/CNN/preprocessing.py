
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch

from .constants import MODEL_CONSTANTS

def build_transform_with_augumantation() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

def build_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

def build_dataloaders(device: torch.device) -> tuple[DataLoader, DataLoader]:
    transform_with_augumantation = build_transform_with_augumantation()
    transform = build_transform()

    train_ds = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=transform_with_augumantation,
    )

    valid_ds = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )

    use_cuda = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=MODEL_CONSTANTS.BATCHSIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=use_cuda,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=MODEL_CONSTANTS.BATCHSIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=use_cuda,
    )

    return train_loader, valid_loader
