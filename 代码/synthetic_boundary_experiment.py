"""Entry point for the synthetic mechanism and boundary experiments."""
from pathlib import Path
import os
import runpy
import sys

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parent
RESOURCES = HERE / "复现资料"
INTERNAL_CODE = RESOURCES / "code"
os.environ.setdefault("MPC_DATA_ROOT", str(REPOSITORY / "data"))
os.environ.setdefault("MPLCONFIGDIR", str(HERE / "运行结果/matplotlib-cache"))
sys.path.insert(0, str(INTERNAL_CODE))
remaining = sys.argv[1:]
if not any(a == "--out" or a.startswith("--out=") for a in remaining):
    remaining += ["--out", str(HERE / "运行结果/synthetic")]
script = INTERNAL_CODE / "synthetic_boundary_experiment.py"
sys.argv = [str(script), *remaining]
runpy.run_path(str(script), run_name="__main__")
