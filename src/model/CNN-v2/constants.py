from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConstants:
    NUM_EPOCHS: int = 1
    BATCHSIZE: int = 256
    L_LATE: float = 3 * 1e-4
    WEIGHT_DECAY: float = 1e-5


MODEL_CONSTANTS = ModelConstants()
