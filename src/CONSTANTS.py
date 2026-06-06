from dataclasses import dataclass
from lib.device_getter import get_device
@dataclass(frozen=True)
class MetaConstants:
    PROJECT: str = "CIFAR10"
    MODEL: str = "CNN-v4"
    WRITE_TRAIN_LOG: bool = True
    WRITE_SUMMARY_LOG: bool = True
    DRAW_TRAIN_GRAPH: bool = False
    BACKUP_BOUNDARY: float = 0.92
    DEVICE = get_device()
    USE_CUDA: bool = DEVICE.type == "cuda"


CONSTANTS = MetaConstants()
