from importlib import import_module
from pathlib import Path
from types import ModuleType


MODEL_ROOT = Path(__file__).resolve().parent.parent / "model"


def _load_module(model_name: str, module_name: str) -> ModuleType:
    module_path = MODEL_ROOT / model_name / f"{module_name}.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Missing module file: {module_path}")
    return import_module(f"model.{model_name}.{module_name}")


def build_model(model_name: str):
    model_module = _load_module(model_name, "model")
    candidate_names = [
        model_name,
        model_name.replace("-", "_"),
        model_name.split("-", 1)[0],
    ]

    model_class = None
    for candidate_name in candidate_names:
        model_class = getattr(model_module, candidate_name, None)
        if model_class is not None:
            break

    if model_class is None:
        raise AttributeError(f"Class {model_name} was not found in {model_name}/model.py")
    return model_class()


def build_dataloaders(model_name: str, device):
    preprocessing_module = _load_module(model_name, "preprocessing")
    loader_builder = getattr(preprocessing_module, "build_dataloaders", None)
    if loader_builder is None:
        raise AttributeError(f"build_dataloaders was not found in {model_name}/preprocessing.py")
    return loader_builder(device)


def load_model_constants(model_name: str):
    constants_module = _load_module(model_name, "constants")
    model_constants = getattr(constants_module, "MODEL_CONSTANTS", None)
    if model_constants is None:
        raise AttributeError(f"MODEL_CONSTANTS was not found in {model_name}/constants.py")
    return model_constants
