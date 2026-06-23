from __future__ import annotations

import subprocess
import select
import unittest
import urllib.request
from pathlib import Path


class CliTestCase(unittest.TestCase):
    def test_help_works_outside_repository_directory(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [repo_root / "mlp-web", "--help"],
            cwd="/tmp",
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Making MLP experiment log viewer", result.stdout)
        self.assertIn("--host", result.stdout)
        self.assertIn("--port", result.stdout)

    def test_cli_starts_health_endpoint(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        process = subprocess.Popen(
            [repo_root / "mlp-web", "--port", "0"],
            cwd="/tmp",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            ready, _, _ = select.select([process.stdout], [], [], 5)
            self.assertTrue(ready, "CLI did not print its URL within five seconds")
            line = process.stdout.readline().strip()
            self.assertTrue(line.startswith("Making MLP web app: "), line)
            base_url = line.removeprefix("Making MLP web app: ")
            with urllib.request.urlopen(f"{base_url}/api/health", timeout=5) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(response.read(), b'{"status": "ok"}')
        finally:
            process.terminate()
            process.wait(timeout=5)
            process.stdout.close()
            process.stderr.close()


if __name__ == "__main__":
    unittest.main()
