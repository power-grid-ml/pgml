"""The preparation benchmark is independent of external repos and solver extras."""

import json
import os
from pathlib import Path
import subprocess
import sys


def test_standalone_example_without_reference_libraries_or_git(tmp_path):
    root = Path(__file__).resolve().parents[2]
    example = root / "run/examples/pgml/benchmark_harmonic_preparation.py"
    output = tmp_path / "timings.json"
    env = dict(
        os.environ, PYTHONPATH=str(root / "src"), PATH="", PYTHONDONTWRITEBYTECODE="1"
    )
    # Refuse optional reference solvers even when the development environment
    # has them installed; run outside a checkout with no Git executable on PATH.
    bootstrap = """
import builtins, runpy, sys
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'pandapower', 'opendssdirect', 'power_grid_model'}:
        raise AssertionError('Optional reference solver imported: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            bootstrap,
            str(example),
            "--out",
            str(output),
            "--nodes",
            "3",
            "--batches",
            "1",
            "2",
            "--repeats",
            "1",
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(output.read_text())
    assert len(data["rows"]) == 6
    assert all(row["changed_max_complex_error_v"] < 1e-8 for row in data["rows"])
    assert set(data) == {
        "platform",
        "torch",
        "threads",
        "cuda_available",
        "orders",
        "repeats",
        "rows",
    }
