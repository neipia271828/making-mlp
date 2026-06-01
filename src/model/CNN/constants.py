from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConstants:
    NUM_EPOCHS: int = 300
    BATCHSIZE: int = 256
    L_LATE: float = 3 * 1e-4


MODEL_CONSTANTS = ModelConstants()
