from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import mimetypes
import os
import re
import shlex
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = Path(__file__).resolve().parent / "static"
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
EDITABLE_META_FIELDS = {
    "PROJECT",
    "MODEL",
    "WRITE_TRAIN_LOG",
    "WRITE_SUMMARY_LOG",
    "DRAW_TRAIN_GRAPH",
    "BACKUP_BOUNDARY",
}
EDITABLE_MODEL_FIELDS = {
    "NUM_EPOCHS",
    "BATCHSIZE",
    "L_LATE",
    "WEIGHT_DECAY",
    "SCHEDULER_NAME",
    "ETA_MIN",
}
SUMMARY_ALIASES = {
    "model": "MODEL",
    "epochs": "NUM_EPOCHS",
}
DEFAULT_SSH_HOST = "172.16.51.202"
DEFAULT_SSH_PORT = 2222
DEFAULT_SSH_USER = "student222"
DEFAULT_SSH_KEY = "~/.ssh/id_ed25519_kmc_gpu"


class ApiError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass(frozen=True)
class ConfigDocument:
    values: dict[str, object]
    version: str


def _safe_eval(node: ast.AST) -> object:
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, bool, int, float, type(None))):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _safe_eval(node.operand)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Only numeric unary expressions are supported")
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if isinstance(left, bool) or isinstance(right, bool):
            raise ValueError("Boolean arithmetic is not supported")
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            raise ValueError("Only numeric expressions are supported")
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        return left / right
    raise ValueError("Expression is not an editable literal")


def _config_assignments(source: str, class_name: str) -> dict[str, ast.AnnAssign]:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                item.target.id: item
                for item in node.body
                if isinstance(item, ast.AnnAssign)
                and isinstance(item.target, ast.Name)
                and item.value is not None
            }
    raise ValueError(f"Class {class_name} was not found")


def read_config(path: Path, class_name: str, editable_fields: set[str]) -> ConfigDocument:
    source = path.read_text(encoding="utf-8")
    assignments = _config_assignments(source, class_name)
    values: dict[str, object] = {}
    for name, assignment in assignments.items():
        if name not in editable_fields:
            continue
        try:
            values[name] = _safe_eval(assignment.value)
        except ValueError:
            continue
    version = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    return ConfigDocument(values=values, version=version)


def _render_value(value: object) -> str:
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    raise ValueError(f"Unsupported value type: {type(value).__name__}")


def _coerce_value(name: str, value: object, current: object) -> object:
    if isinstance(current, bool):
        if not isinstance(value, bool):
            raise ApiError(HTTPStatus.BAD_REQUEST, f"{name} must be a boolean")
        coerced = value
    elif isinstance(current, int) and not isinstance(current, bool):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ApiError(HTTPStatus.BAD_REQUEST, f"{name} must be an integer")
        coerced = value
    elif isinstance(current, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ApiError(HTTPStatus.BAD_REQUEST, f"{name} must be a number")
        coerced = float(value)
    elif isinstance(current, str):
        if not isinstance(value, str) or not value.strip():
            raise ApiError(HTTPStatus.BAD_REQUEST, f"{name} must be a non-empty string")
        coerced = value.strip()
    else:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"{name} is not editable")

    if name in {"NUM_EPOCHS", "BATCHSIZE"} and coerced <= 0:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"{name} must be greater than zero")
    if name in {"L_LATE", "WEIGHT_DECAY", "ETA_MIN"} and coerced < 0:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"{name} must not be negative")
    if name == "L_LATE" and coerced == 0:
        raise ApiError(HTTPStatus.BAD_REQUEST, "L_LATE must be greater than zero")
    if name == "BACKUP_BOUNDARY" and not 0 <= coerced <= 1:
        raise ApiError(HTTPStatus.BAD_REQUEST, "BACKUP_BOUNDARY must be between 0 and 1")
    return coerced


def update_config(
    path: Path,
    class_name: str,
    editable_fields: set[str],
    updates: dict[str, object],
    expected_version: str,
) -> ConfigDocument:
    source = path.read_text(encoding="utf-8")
    current_version = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    if expected_version != current_version:
        raise ApiError(
            HTTPStatus.CONFLICT,
            "設定ファイルが別の処理で変更されました。再読み込みしてください。",
        )
    assignments = _config_assignments(source, class_name)
    current = read_config(path, class_name, editable_fields)
    if not updates:
        raise ApiError(HTTPStatus.BAD_REQUEST, "No values were supplied")

    replacements: list[tuple[int, int, str]] = []
    for name, value in updates.items():
        if name not in editable_fields or name not in current.values or name not in assignments:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"{name} is not editable")
        coerced = _coerce_value(name, value, current.values[name])
        if coerced == current.values[name]:
            continue
        value_node = assignments[name].value
        start = _offset_for(source, value_node.lineno, value_node.col_offset)
        end = _offset_for(source, value_node.end_lineno, value_node.end_col_offset)
        replacements.append((start, end, _render_value(coerced)))

    if not replacements:
        return current

    updated_source = source
    for start, end, replacement in sorted(replacements, reverse=True):
        updated_source = updated_source[:start] + replacement + updated_source[end:]
    ast.parse(updated_source)

    file_mode = path.stat().st_mode
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as temporary:
        temporary.write(updated_source)
        temporary_path = Path(temporary.name)
    os.chmod(temporary_path, file_mode)
    os.replace(temporary_path, path)
    return read_config(path, class_name, editable_fields)


def _offset_for(source: str, line: int, column: int) -> int:
    lines = source.splitlines(keepends=True)
    return sum(len(item) for item in lines[: line - 1]) + column


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _normalise_summary(row: dict[str, str]) -> dict[str, str]:
    result = dict(row)
    for old_name, new_name in SUMMARY_ALIASES.items():
        if old_name in result and new_name not in result:
            result[new_name] = result[old_name]
    return result


def _require_safe_name(value: str, label: str) -> str:
    if not value or not SAFE_NAME.fullmatch(value):
        raise ApiError(HTTPStatus.BAD_REQUEST, f"Invalid {label}")
    return value


class AppState:
    def __init__(self, repo_root: Path = REPO_ROOT) -> None:
        self.repo_root = repo_root.resolve()
        self.static_root = STATIC_ROOT.resolve()
        self.write_lock = threading.Lock()
        self.git_lock = threading.Lock()

    @property
    def model_root(self) -> Path:
        return self.repo_root / "src" / "model"

    @property
    def data_root(self) -> Path:
        return self.repo_root / "data"

    def model_names(self) -> list[str]:
        if not self.model_root.exists():
            return []
        return sorted(
            path.name
            for path in self.model_root.iterdir()
            if path.is_dir()
            and (path / "model.py").exists()
            and (path / "constants.py").exists()
            and (path / "preprocessing.py").exists()
        )

    def project_names(self) -> list[str]:
        projects = {
            path.name
            for path in self.data_root.iterdir()
            if path.is_dir() and (path / "logs").is_dir()
        } if self.data_root.exists() else set()
        projects.update({"CIFAR10", "FashionMNIST", "KMNIST", "MNIST"})
        return sorted(projects)

    def configs(self, selected_model: str | None = None) -> dict[str, object]:
        meta = read_config(
            self.repo_root / "src" / "CONSTANTS.py",
            "MetaConstants",
            EDITABLE_META_FIELDS,
        )
        model_name = selected_model or str(meta.values.get("MODEL", ""))
        _require_safe_name(model_name, "model")
        model_path = self.model_root / model_name / "constants.py"
        if not model_path.exists():
            raise ApiError(HTTPStatus.NOT_FOUND, f"Model constants not found: {model_name}")
        model = read_config(model_path, "ModelConstants", EDITABLE_MODEL_FIELDS)
        return {
            "meta": {"values": meta.values, "version": meta.version},
            "model": {
                "name": model_name,
                "values": model.values,
                "version": model.version,
            },
            "models": self.model_names(),
            "projects": self.project_names(),
        }

    def update_configs(self, payload: dict[str, object]) -> dict[str, object]:
        scope = payload.get("scope")
        values = payload.get("values")
        version = payload.get("version")
        if not isinstance(values, dict) or not isinstance(version, str):
            raise ApiError(HTTPStatus.BAD_REQUEST, "values and version are required")

        with self.write_lock:
            if scope == "meta":
                if "MODEL" in values and values["MODEL"] not in self.model_names():
                    raise ApiError(HTTPStatus.BAD_REQUEST, "Unknown model")
                if "PROJECT" in values and values["PROJECT"] not in self.project_names():
                    raise ApiError(HTTPStatus.BAD_REQUEST, "Unknown project")
                document = update_config(
                    self.repo_root / "src" / "CONSTANTS.py",
                    "MetaConstants",
                    EDITABLE_META_FIELDS,
                    values,
                    version,
                )
                return {"scope": "meta", "values": document.values, "version": document.version}

            if scope == "model":
                model_name = payload.get("model")
                if not isinstance(model_name, str) or model_name not in self.model_names():
                    raise ApiError(HTTPStatus.BAD_REQUEST, "Unknown model")
                document = update_config(
                    self.model_root / model_name / "constants.py",
                    "ModelConstants",
                    EDITABLE_MODEL_FIELDS,
                    values,
                    version,
                )
                return {
                    "scope": "model",
                    "model": model_name,
                    "values": document.values,
                    "version": document.version,
                }
        raise ApiError(HTTPStatus.BAD_REQUEST, "scope must be meta or model")

    def runs(self, project: str) -> list[dict[str, object]]:
        project = _require_safe_name(project, "project")
        logs_root = self.data_root / project / "logs"
        summary_rows = {
            row.get("time_stamp", ""): _normalise_summary(row)
            for row in _read_csv(logs_root / f"{project}.csv")
            if row.get("time_stamp")
        }
        run_names = set(summary_rows)
        if logs_root.exists():
            run_names.update(
                path.name
                for path in logs_root.iterdir()
                if path.is_dir() and SAFE_NAME.fullmatch(path.name)
            )

        runs: list[dict[str, object]] = []
        for run_name in sorted(run_names, reverse=True):
            summary = summary_rows.get(run_name, {})
            run_dir = logs_root / run_name
            epoch_rows = _read_csv(run_dir / "train_logs.csv")
            latest = epoch_rows[-1] if epoch_rows else {}
            runs.append(
                {
                    "timestamp": run_name,
                    "model": summary.get("MODEL", ""),
                    "epochs": summary.get("NUM_EPOCHS", latest.get("epoch", "")),
                    "valid_accuracy": summary.get(
                        "valid_accuracy_last10_avg",
                        summary.get("valid_accuracy", latest.get("valid_accuracy", "")),
                    ),
                    "valid_loss": summary.get(
                        "valid_loss_last10_avg",
                        summary.get("valid_loss", latest.get("valid_loss", "")),
                    ),
                    "ep_time": summary.get("ep_time", latest.get("ep_time", "")),
                    "has_graph": (run_dir / "train_graph.svg").exists(),
                    "has_epochs": bool(epoch_rows),
                    "summary": summary,
                }
            )
        return runs

    def run_detail(self, project: str, run_name: str) -> dict[str, object]:
        project = _require_safe_name(project, "project")
        run_name = _require_safe_name(run_name, "run")
        rows = _read_csv(self.data_root / project / "logs" / run_name / "train_logs.csv")
        if not rows:
            raise ApiError(HTTPStatus.NOT_FOUND, "Epoch log was not found")
        return {
            "project": project,
            "timestamp": run_name,
            "epochs": rows,
            "graph_url": f"/api/graph?project={project}&run={run_name}",
        }

    def graph_path(self, project: str, run_name: str) -> Path:
        project = _require_safe_name(project, "project")
        run_name = _require_safe_name(run_name, "run")
        path = self.data_root / project / "logs" / run_name / "train_graph.svg"
        if not path.exists():
            raise ApiError(HTTPStatus.NOT_FOUND, "Graph was not found")
        return path

    def _git(self, *arguments: str, timeout: int = 15) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                ["git", *arguments],
                cwd=self.repo_root,
                capture_output=True,
                stdin=subprocess.DEVNULL,
                text=True,
                timeout=timeout,
                check=False,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            )
        except FileNotFoundError:
            raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "git command was not found") from None
        except subprocess.TimeoutExpired:
            raise ApiError(HTTPStatus.GATEWAY_TIMEOUT, "git command timed out") from None

    def _require_git_success(
        self,
        result: subprocess.CompletedProcess[str],
        fallback_message: str,
        status: int = HTTPStatus.INTERNAL_SERVER_ERROR,
    ) -> str:
        if result.returncode != 0:
            message = (result.stderr or result.stdout or fallback_message).strip()
            raise ApiError(status, message[-4_000:])
        return result.stdout.strip()

    def repository_status(self) -> dict[str, object]:
        inside = self._git("rev-parse", "--is-inside-work-tree")
        self._require_git_success(inside, "This directory is not a Git repository")
        branch = self._require_git_success(
            self._git("branch", "--show-current"),
            "Could not determine current branch",
        )
        commit = self._require_git_success(
            self._git("rev-parse", "--short", "HEAD"),
            "Could not determine current commit",
        )
        subject = self._require_git_success(
            self._git("log", "-1", "--format=%s"),
            "Could not read latest commit",
        )
        changes_output = self._require_git_success(
            self._git("status", "--porcelain"),
            "Could not read working tree status",
        )
        changes = [line for line in changes_output.splitlines() if line]

        remote_result = self._git("remote", "get-url", "origin")
        remote = remote_result.stdout.strip() if remote_result.returncode == 0 else ""
        upstream_result = self._git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}")
        upstream = upstream_result.stdout.strip() if upstream_result.returncode == 0 else ""
        ahead = 0
        behind = 0
        if upstream:
            counts_result = self._git("rev-list", "--left-right", "--count", f"HEAD...{upstream}")
            if counts_result.returncode == 0:
                parts = counts_result.stdout.split()
                if len(parts) == 2:
                    ahead, behind = int(parts[0]), int(parts[1])

        return {
            "branch": branch or "detached HEAD",
            "commit": commit,
            "subject": subject,
            "remote": remote,
            "upstream": upstream,
            "ahead": ahead,
            "behind": behind,
            "clean": not changes,
            "change_count": len(changes),
            "changes": changes[:50],
        }

    def pull_repository(self) -> dict[str, object]:
        if not self.git_lock.acquire(blocking=False):
            raise ApiError(HTTPStatus.CONFLICT, "Another pull is already running")
        try:
            before = self._require_git_success(
                self._git("rev-parse", "HEAD"),
                "Could not determine current commit",
            )
            pull_result = self._git("pull", "--ff-only", timeout=120)
            output = "\n".join(
                part.strip() for part in (pull_result.stdout, pull_result.stderr) if part.strip()
            )
            if pull_result.returncode != 0:
                raise ApiError(
                    HTTPStatus.CONFLICT,
                    output[-4_000:] or "git pull --ff-only failed",
                )
            after = self._require_git_success(
                self._git("rev-parse", "HEAD"),
                "Could not determine updated commit",
            )
            changed_files: list[str] = []
            if before != after:
                changed_output = self._require_git_success(
                    self._git("diff", "--name-only", f"{before}..{after}"),
                    "Could not list pulled files",
                )
                changed_files = [line for line in changed_output.splitlines() if line]
            return {
                "updated": before != after,
                "message": output or "Already up to date.",
                "changed_files": changed_files,
                "restart_required": any(
                    path == "mlp-web" or path.startswith("webapp/") for path in changed_files
                ),
                "repository": self.repository_status(),
            }
        finally:
            self.git_lock.release()


class AppHandler(BaseHTTPRequestHandler):
    server_version = "MakingMLPWeb/1.0"

    @property
    def state(self) -> AppState:
        return self.server.app_state  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if parsed.path == "/api/health":
                self._send_json({"status": "ok"})
            elif parsed.path == "/api/config":
                self._send_json(self.state.configs(_query_value(query, "model", required=False)))
            elif parsed.path == "/api/runs":
                self._send_json({"runs": self.state.runs(_query_value(query, "project"))})
            elif parsed.path == "/api/run":
                self._send_json(
                    self.state.run_detail(
                        _query_value(query, "project"),
                        _query_value(query, "run"),
                    )
                )
            elif parsed.path == "/api/repository":
                self._send_json(self.state.repository_status())
            elif parsed.path == "/api/graph":
                self._send_file(
                    self.state.graph_path(
                        _query_value(query, "project"),
                        _query_value(query, "run"),
                    ),
                    "image/svg+xml",
                )
            elif parsed.path in {"/", "/index.html"}:
                self._send_file(self.state.static_root / "index.html", "text/html; charset=utf-8")
            elif parsed.path.startswith("/static/"):
                name = parsed.path.removeprefix("/static/")
                if "/" in name or not SAFE_NAME.fullmatch(name):
                    raise ApiError(HTTPStatus.NOT_FOUND, "Static file not found")
                path = self.state.static_root / name
                self._send_file(path, mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            else:
                raise ApiError(HTTPStatus.NOT_FOUND, "Not found")
        except ApiError as error:
            self._send_json({"error": error.message}, error.status)
        except Exception as error:
            self._send_json({"error": f"Internal server error: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        try:
            parsed = urlparse(self.path)
            self._require_same_origin()
            if parsed.path == "/api/repository/pull":
                self._send_json(self.state.pull_repository())
                return
            if parsed.path != "/api/config":
                raise ApiError(HTTPStatus.NOT_FOUND, "Not found")
            if self.headers.get_content_type() != "application/json":
                raise ApiError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Content-Type must be application/json")
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 64_000:
                raise ApiError(HTTPStatus.BAD_REQUEST, "Invalid request body size")
            try:
                payload = json.loads(self.rfile.read(content_length))
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise ApiError(HTTPStatus.BAD_REQUEST, "Invalid JSON") from None
            if not isinstance(payload, dict):
                raise ApiError(HTTPStatus.BAD_REQUEST, "JSON body must be an object")
            self._send_json(self.state.update_configs(payload))
        except ApiError as error:
            self._send_json({"error": error.message}, error.status)
        except Exception as error:
            self._send_json({"error": f"Internal server error: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _require_same_origin(self) -> None:
        origin = self.headers.get("Origin")
        host = self.headers.get("Host")
        if not origin:
            return
        parsed_origin = urlparse(origin)
        if parsed_origin.scheme not in {"http", "https"} or parsed_origin.netloc != host:
            raise ApiError(HTTPStatus.FORBIDDEN, "Cross-origin writes are not allowed")

    def _send_json(self, payload: object, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.exists() or not path.is_file():
            raise ApiError(HTTPStatus.NOT_FOUND, "File not found")
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(body)


def _query_value(query: dict[str, list[str]], name: str, required: bool = True) -> str | None:
    values = query.get(name)
    if not values or not values[0]:
        if required:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"Missing query parameter: {name}")
        return None
    return values[0]


def create_server(host: str, port: int, repo_root: Path = REPO_ROOT) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), AppHandler)
    server.app_state = AppState(repo_root)  # type: ignore[attr-defined]
    return server


def format_tunnel_guide(
    remote_port: int,
    local_port: int,
    ssh_host: str,
    ssh_port: int,
    ssh_user: str,
    ssh_key: str,
) -> str:
    destination = shlex.quote(f"{ssh_user}@{ssh_host}")
    key_argument = (
        ssh_key
        if re.fullmatch(r"[A-Za-z0-9_./~+-]+", ssh_key)
        else shlex.quote(ssh_key)
    )
    command = " ".join(
        [
            "ssh",
            "-p",
            str(ssh_port),
            "-N",
            "-L",
            shlex.quote(f"{local_port}:127.0.0.1:{remote_port}"),
            "-i",
            key_argument,
            destination,
        ]
    )
    return "\n".join(
        [
            "",
            "=== MacからGPUサーバーへ接続する方法 ===",
            "1. このGPUサーバー側のターミナルは、アプリを起動したままにします。",
            "2. Macで別のターミナルを開き、次を実行します。",
            "",
            f"   {command}",
            "",
            "3. SSHコマンドを実行したターミナルも開いたままにします。",
            "4. Macのブラウザで次を開きます。",
            "",
            f"   http://127.0.0.1:{local_port}",
            "",
            "終了するときは、アプリ側とSSHトンネル側で Ctrl+C を押します。",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Making MLP experiment log viewer")
    parser.add_argument(
        "command",
        nargs="?",
        choices=("serve", "tunnel"),
        default="serve",
        help="serve: Webアプリを起動 / tunnel: SSHトンネル手順だけを表示",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--local-port", type=int, default=None)
    parser.add_argument("--ssh-host", default=DEFAULT_SSH_HOST)
    parser.add_argument("--ssh-port", type=int, default=DEFAULT_SSH_PORT)
    parser.add_argument("--ssh-user", default=DEFAULT_SSH_USER)
    parser.add_argument("--ssh-key", default=DEFAULT_SSH_KEY)
    args = parser.parse_args()

    if args.command == "tunnel":
        print(
            format_tunnel_guide(
                remote_port=args.port,
                local_port=args.local_port or args.port,
                ssh_host=args.ssh_host,
                ssh_port=args.ssh_port,
                ssh_user=args.ssh_user,
                ssh_key=args.ssh_key,
            )
        )
        return

    server = create_server(args.host, args.port)
    print(f"Making MLP web app: http://{args.host}:{server.server_port}")
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print("WARNING: This server has no authentication. Prefer SSH port forwarding.")
    print(
        format_tunnel_guide(
            remote_port=server.server_port,
            local_port=args.local_port or server.server_port,
            ssh_host=args.ssh_host,
            ssh_port=args.ssh_port,
            ssh_user=args.ssh_user,
            ssh_key=args.ssh_key,
        )
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
