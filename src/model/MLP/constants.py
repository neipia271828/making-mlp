from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConstants:
    NUM_EPOCHS: int = 1
    BATCHSIZE: int = 256
    L_LATE: float = 1e-3
    WEIGHT_DECAY: float = 0.0


MODEL_CONSTANTS = ModelConstants()
