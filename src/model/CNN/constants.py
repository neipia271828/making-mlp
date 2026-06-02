from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConstants:
    NUM_EPOCHS: int = 100
    BATCHSIZE: int = 256
    L_LATE: float = 3 * 1e-4
    WEIGHT_DECAY: float = 5e-5


MODEL_CONSTANTS = ModelConstants()
