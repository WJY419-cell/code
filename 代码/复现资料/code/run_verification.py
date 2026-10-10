"""Run the four release checks from any working directory."""
from pathlib import Path
import os
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
(root / 'audit').mkdir(exist_ok=True)
env = os.environ.copy()
env.pop('MPC_FIT_AUDIT', None)
env.pop('MPC_UNSEEN_OUT', None)
for script in ['verify_recorded_results.py', 'verify_unseen_results.py', 'validate_solver.py', 'unseen_pairs/validate_protocol.py']:
    print(f'Checking {script}', flush=True)
    subprocess.run([sys.executable, str(root / 'code' / script)], cwd=root, env=env, check=True)
print('All four release checks passed.', flush=True)
