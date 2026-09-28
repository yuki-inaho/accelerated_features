"""Execute and save the paper-style notebook using this Python interpreter."""
from __future__ import annotations
import datetime
import hashlib
import json
import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks/raco_visualization.ipynb"
    notebook = nbformat.read(path, as_version=4)
    km = KernelManager(kernel_name="python3")
    km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        NotebookClient(notebook, km=km, timeout=600, resources={"metadata": {"path": str(root)}}, record_timing=True).execute()
    finally:
        if km.has_kernel:
            km.shutdown_kernel(now=True)
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    errors = [output for cell in notebook.cells if cell.cell_type == "code" for output in cell.get("outputs", []) if output.output_type == "error"]
    if errors:
        raise RuntimeError("Notebook contains error outputs")
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    counts = [cell.execution_count for cell in code_cells]
    if counts != list(range(1, len(code_cells) + 1)):
        raise RuntimeError("Not every code cell was executed in order")
    result = {"started_utc": started, "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "python_executable": sys.executable, "notebook": str(path.relative_to(root)),
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "code_cells": len(code_cells),
              "execution_counts": counts, "error_outputs": len(errors), "all_cells_executed": True}
    report_path = root / "outputs/raco-paper-visualization/notebook_execution.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
