from dataclasses import dataclass

@dataclass(frozen=True)
class MetaConstants:
    PROJECT: str = "CIFAR10"
    MODEL: str = "CNN-v3"
    WRITE_TRAIN_LOG: bool = True
    WRITE_SUMMARY_LOG: bool = True
    DRAW_TRAIN_GRAPH: bool = True
    BACKUP_BOUNDARY: float = 0.92


CONSTANTS = MetaConstants()
