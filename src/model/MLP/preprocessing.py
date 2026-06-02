
import torch
from torchvision import transforms

from CONSTANTS import CONSTANTS
from .constants import MODEL_CONSTANTS
from lib.dataset_builder import build_dataloaders as build_shared_dataloaders, get_dataset_config

def flatten_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.view(-1)


def build_transform(dataset_name: str, train: bool) -> transforms.Compose:
    dataset_config = get_dataset_config(dataset_name)
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
            transforms.Lambda(flatten_tensor),
        ]
    )

    return transforms.Compose(transform_steps)


def build_dataloaders(device: torch.device):
    dataset_name = CONSTANTS.PROJECT
    return build_shared_dataloaders(
        dataset_name=dataset_name,
        batch_size=MODEL_CONSTANTS.BATCHSIZE,
        device=device,
        train_transform=build_transform(dataset_name, train=True),
        valid_transform=build_transform(dataset_name, train=False),
    )
