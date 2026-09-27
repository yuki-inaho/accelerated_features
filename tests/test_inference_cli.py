"""End-to-end test for the ``inference.py`` command-line entry point."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_cli_writes_match_visualization(tmp_path: Path) -> None:
    output = tmp_path / "matches.png"
    completed = subprocess.run(
        [
            sys.executable,
            "inference.py",
            "--method",
            "xfeat",
            "--top-k",
            "1024",
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert output.is_file()
    assert output.stat().st_size > 0
    assert "matches=" in completed.stdout
    assert "inliers=" in completed.stdout
