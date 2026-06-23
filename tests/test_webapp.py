from __future__ import annotations

import json
import subprocess
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from webapp.server import (
    ApiError,
    AppState,
    EDITABLE_META_FIELDS,
    create_server,
    read_config,
    update_config,
)


META_SOURCE = '''from dataclasses import dataclass

@dataclass(frozen=True)
class MetaConstants:
    PROJECT: str = "CIFAR10"
    MODEL: str = "TestModel"
    WRITE_TRAIN_LOG: bool = True
    WRITE_SUMMARY_LOG: bool = True
    DRAW_TRAIN_GRAPH: bool = False
    BACKUP_BOUNDARY: float = 0.92
    DEVICE = object()
'''

MODEL_SOURCE = '''from dataclasses import dataclass

@dataclass(frozen=True)
class ModelConstants:
    NUM_EPOCHS: int = 120
    BATCHSIZE: int = 256
    L_LATE: float = 3 * 1e-3
    WEIGHT_DECAY: float = 1e-4
'''


class WebAppTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "src" / "model" / "TestModel").mkdir(parents=True)
        (self.root / "src" / "CONSTANTS.py").write_text(META_SOURCE, encoding="utf-8")
        model_root = self.root / "src" / "model" / "TestModel"
        (model_root / "constants.py").write_text(MODEL_SOURCE, encoding="utf-8")
        (model_root / "model.py").write_text("class TestModel: pass\n", encoding="utf-8")
        (model_root / "preprocessing.py").write_text("def build_dataloaders(): pass\n", encoding="utf-8")
        logs = self.root / "data" / "CIFAR10" / "logs"
        run = logs / "2026-06-23-12-00"
        run.mkdir(parents=True)
        (logs / "CIFAR10.csv").write_text(
            "PROJECT,MODEL,NUM_EPOCHS,time_stamp,valid_accuracy_last10_avg,ep_time\n"
            "CIFAR10,TestModel,2,2026-06-23-12-00,0.85,1.25\n",
            encoding="utf-8",
        )
        (run / "train_logs.csv").write_text(
            "ep_time,epoch,train_loss,train_accuracy,valid_loss,valid_accuracy\n"
            "1.3,1,1.0,0.6,0.8,0.7\n"
            "1.2,2,0.7,0.8,0.5,0.85\n",
            encoding="utf-8",
        )
        (run / "train_graph.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_reads_arithmetic_constants(self) -> None:
        document = read_config(
            self.root / "src" / "model" / "TestModel" / "constants.py",
            "ModelConstants",
            {"NUM_EPOCHS", "BATCHSIZE", "L_LATE", "WEIGHT_DECAY"},
        )
        self.assertEqual(document.values["L_LATE"], 0.003)
        self.assertEqual(document.values["NUM_EPOCHS"], 120)

    def test_updates_only_allowlisted_constants(self) -> None:
        path = self.root / "src" / "CONSTANTS.py"
        document = read_config(path, "MetaConstants", EDITABLE_META_FIELDS)
        updated = update_config(
            path,
            "MetaConstants",
            EDITABLE_META_FIELDS,
            {"BACKUP_BOUNDARY": 0.95, "DRAW_TRAIN_GRAPH": True},
            document.version,
        )
        self.assertEqual(updated.values["BACKUP_BOUNDARY"], 0.95)
        self.assertTrue(updated.values["DRAW_TRAIN_GRAPH"])
        self.assertIn("DEVICE = object()", path.read_text(encoding="utf-8"))

    def test_rejects_stale_version_and_invalid_values(self) -> None:
        path = self.root / "src" / "CONSTANTS.py"
        document = read_config(path, "MetaConstants", EDITABLE_META_FIELDS)
        with self.assertRaises(ApiError) as context:
            update_config(
                path,
                "MetaConstants",
                EDITABLE_META_FIELDS,
                {"BACKUP_BOUNDARY": 2.0},
                document.version,
            )
        self.assertEqual(context.exception.status, 400)
        with self.assertRaises(ApiError) as context:
            update_config(
                path,
                "MetaConstants",
                EDITABLE_META_FIELDS,
                {"BACKUP_BOUNDARY": 0.8},
                "stale-version",
            )
        self.assertEqual(context.exception.status, 409)

    def test_same_value_does_not_rewrite_expression(self) -> None:
        path = self.root / "src" / "model" / "TestModel" / "constants.py"
        before = path.read_text(encoding="utf-8")
        document = read_config(
            path,
            "ModelConstants",
            {"NUM_EPOCHS", "BATCHSIZE", "L_LATE", "WEIGHT_DECAY"},
        )
        updated = update_config(
            path,
            "ModelConstants",
            {"NUM_EPOCHS", "BATCHSIZE", "L_LATE", "WEIGHT_DECAY"},
            {"L_LATE": 0.003},
            document.version,
        )
        self.assertEqual(path.read_text(encoding="utf-8"), before)
        self.assertEqual(updated.version, document.version)

    def test_discovers_runs_and_epoch_details(self) -> None:
        state = AppState(self.root)
        runs = state.runs("CIFAR10")
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["model"], "TestModel")
        self.assertEqual(runs[0]["valid_accuracy"], "0.85")
        detail = state.run_detail("CIFAR10", "2026-06-23-12-00")
        self.assertEqual(len(detail["epochs"]), 2)
        self.assertEqual(detail["epochs"][-1]["valid_accuracy"], "0.85")

    def test_http_api_and_config_write(self) -> None:
        server = create_server("127.0.0.1", 0, self.root)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{server.server_port}"
        try:
            with urllib.request.urlopen(f"{base_url}/api/config") as response:
                config = json.load(response)
            self.assertEqual(config["meta"]["values"]["MODEL"], "TestModel")

            request = urllib.request.Request(
                f"{base_url}/api/config",
                data=json.dumps(
                    {
                        "scope": "meta",
                        "version": config["meta"]["version"],
                        "values": {"BACKUP_BOUNDARY": 0.9},
                    }
                ).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request) as response:
                result = json.load(response)
            self.assertEqual(result["values"]["BACKUP_BOUNDARY"], 0.9)

            cross_origin_request = urllib.request.Request(
                f"{base_url}/api/config",
                data=b"{}",
                headers={"Content-Type": "application/json", "Origin": "https://example.com"},
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as context:
                urllib.request.urlopen(cross_origin_request)
            self.assertEqual(context.exception.code, 403)
            context.exception.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


class GitFeatureTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.local = self.root / "local"
        self.remote = self.root / "remote.git"
        self.local.mkdir()
        self._run("git", "init", "--bare", self.remote)
        self._run("git", "init", "-b", "main", cwd=self.local)
        self._configure_author(self.local)
        (self.local / "README.md").write_text("initial\n", encoding="utf-8")
        self._run("git", "add", "README.md", cwd=self.local)
        self._run("git", "commit", "-m", "initial", cwd=self.local)
        self._run("git", "remote", "add", "origin", self.remote, cwd=self.local)
        self._run("git", "push", "-u", "origin", "main", cwd=self.local)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _run(self, *command, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(part) for part in command],
            cwd=cwd or self.root,
            capture_output=True,
            text=True,
            check=True,
        )

    def _configure_author(self, repository: Path) -> None:
        self._run("git", "config", "user.name", "Web App Test", cwd=repository)
        self._run("git", "config", "user.email", "webapp@example.test", cwd=repository)

    def test_repository_status_reports_local_changes(self) -> None:
        state = AppState(self.local)
        clean_status = state.repository_status()
        self.assertEqual(clean_status["branch"], "main")
        self.assertTrue(clean_status["clean"])
        self.assertEqual(clean_status["change_count"], 0)

        (self.local / "README.md").write_text("changed\n", encoding="utf-8")
        dirty_status = state.repository_status()
        self.assertFalse(dirty_status["clean"])
        self.assertEqual(dirty_status["change_count"], 1)

    def test_pull_repository_fast_forwards_from_remote(self) -> None:
        contributor = self.root / "contributor"
        self._run("git", "clone", self.remote, contributor)
        self._run("git", "switch", "main", cwd=contributor)
        self._configure_author(contributor)
        (contributor / "REMOTE.md").write_text("from remote\n", encoding="utf-8")
        self._run("git", "add", "REMOTE.md", cwd=contributor)
        self._run("git", "commit", "-m", "remote change", cwd=contributor)
        self._run("git", "push", "origin", "main", cwd=contributor)

        result = AppState(self.local).pull_repository()
        self.assertTrue(result["updated"])
        self.assertIn("REMOTE.md", result["changed_files"])
        self.assertFalse(result["restart_required"])
        self.assertTrue((self.local / "REMOTE.md").exists())
        self.assertEqual(result["repository"]["behind"], 0)


if __name__ == "__main__":
    unittest.main()
