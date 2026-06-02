from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConstants:
    NUM_EPOCHS: int = 1
    BATCHSIZE: int = 256
    L_LATE: float = 3e-3
    WEIGHT_DECAY: float = 5e-5
    SCHEDULER_NAME: str = "CosineAnnealingLR"
    ETA_MIN: float = 1e-5


MODEL_CONSTANTS = ModelConstants()
