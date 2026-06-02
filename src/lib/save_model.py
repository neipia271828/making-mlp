import hashlib
import subprocess
import torch
from pathlib import Path

from lib.commithash_getter import get_commit_hash

MODEL_SOURCE_ROOT = Path("src/model")


def _iter_model_source_files() -> list[Path]:
    return sorted(
        path for path in MODEL_SOURCE_ROOT.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )


def _calculate_model_source_hash() -> str:
    digest = hashlib.sha256()
    for path in _iter_model_source_files():
        digest.update(str(path.relative_to(MODEL_SOURCE_ROOT.parent)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _auto_commit_model_sources(source_hash: str) -> str:
    subprocess.run(["git", "add", "src/model"], check=True)

    diff_result = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--", "src/model"],
        check=False,
    )
    if diff_result.returncode == 0:
        return get_commit_hash()
    if diff_result.returncode != 1:
        raise subprocess.CalledProcessError(diff_result.returncode, diff_result.args)

    subprocess.run(
        ["git", "commit", "-m", f"model source update: {source_hash[:12]}"],
        check=True,
    )
    return get_commit_hash()


def _ensure_model_dir(project: str, timestamp: str) -> Path:
    model_dir = Path(f"data/{project}/models") / timestamp
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def save_best_checkpoint(project: str, timestamp: str, model, epoch: int, valid_loss: float, valid_accuracy: float) -> None:
    model_dir = _ensure_model_dir(project, timestamp)
    checkpoint_path = model_dir / "best_checkpoint.pt"
    torch.save(model.state_dict(), checkpoint_path)

    commit_hash = get_commit_hash()
    metadata_path = model_dir / "best_checkpoint.txt"
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write(f"epoch={epoch}\n")
        f.write(f"valid_loss={valid_loss}\n")
        f.write(f"valid_accuracy={valid_accuracy}\n")
        f.write(f"commit_hash={commit_hash}\n")


def save_model(project, timestamp, model):
    models_root = Path(f"data/{project}/models")
    models_root.mkdir(parents=True, exist_ok=True)

    source_hash_path = models_root / "model_source_hash.txt"
    current_source_hash = _calculate_model_source_hash()
    previous_source_hash = source_hash_path.read_text(encoding="utf-8").strip() if source_hash_path.exists() else None

    if current_source_hash == previous_source_hash:
        return

    model_dir = _ensure_model_dir(project, timestamp)

    model_path = model_dir / "model.pt"
    torch.save(model.state_dict(), model_path)

    source_hash_path.write_text(current_source_hash, encoding="utf-8")
    commit_hash = _auto_commit_model_sources(current_source_hash)

    with open(model_dir / "model.txt", "w", encoding="utf-8") as f:
        f.write(commit_hash)
