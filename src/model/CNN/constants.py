from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConstants:
    NUM_EPOCHS: int = 200
    BATCHSIZE: int = 256
    L_LATE: float = 1e-3


MODEL_CONSTANTS = ModelConstants()
