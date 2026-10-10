"""Entry point for real experiments, unseen-pair experiments and verification."""
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
import argparse

parser = argparse.ArgumentParser(description=__doc__, add_help=False)
parser.add_argument("--mode", choices=["real", "unseen", "verify", "figures"], default="real")
args, remaining = parser.parse_known_args()
if args.mode == "real":
    script = INTERNAL_CODE / "metapath_coupling_experiment.py"
    if not any(a == "--out" or a.startswith("--out=") for a in remaining):
        remaining += ["--out", str(HERE / "运行结果/real")]
elif args.mode == "unseen":
    if remaining:
        parser.error("unseen mode takes no extra flags; use the recorded full protocol")
    os.environ.setdefault("MPC_UNSEEN_OUT", str(HERE / "运行结果/unseen_pairs"))
    script = INTERNAL_CODE / "unseen_pairs/run_experiment.py"
elif args.mode == "verify":
    if remaining:
        parser.error("verify mode takes no extra flags")
    script = INTERNAL_CODE / "run_verification.py"
else:
    if remaining:
        parser.error("figures mode takes no extra flags")
    script = INTERNAL_CODE / "generate_mpc_publication_figures.py"
sys.argv = [str(script), *remaining]
runpy.run_path(str(script), run_name="__main__")
