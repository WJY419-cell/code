"""Recompute published summaries and paired tests from the recorded runs.

This check requires no raw source data and does not rerun model fitting.
Run from any working directory: python code/verify_recorded_results.py
"""
from pathlib import Path
import csv
import json
import math
import metapath_coupling_experiment as real
import synthetic_boundary_experiment as synthetic

ROOT = Path(__file__).resolve().parents[1]

def read(name):
    with (ROOT / 'results' / name).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for k, v in row.items():
            if v == '':
                continue
            try:
                x = float(v)
                row[k] = int(x) if x.is_integer() else x
            except ValueError:
                pass
    return rows

def compare(actual, expected):
    assert len(actual) == len(expected), (len(actual), len(expected))
    for i, (a, b) in enumerate(zip(actual, expected)):
        assert set(a) == set(b), (i, set(a), set(b))
        for k, v in a.items():
            w = b[k]
            if isinstance(v, (float, int)) and isinstance(w, (float, int)):
                assert math.isclose(v, w, rel_tol=1e-9, abs_tol=1e-11), (i, k, v, w)
            else:
                assert v == w, (i, k, v, w)

checks = [
    ('real_per_seed.csv', real.summarize, 'real_summary.csv'),
    ('real_per_seed.csv', real.significance, 'paired_significance.csv'),
    ('support_sweep_per_seed.csv', real.summarize_support, 'support_sweep_summary.csv'),
    ('support_sweep_per_seed.csv', real.support_significance, 'support_sweep_significance.csv'),
    ('long_tail_per_seed.csv', real.summarize_long_tail, 'long_tail_summary.csv'),
    ('long_tail_per_seed.csv', real.long_tail_significance, 'long_tail_significance.csv'),
    ('synthetic_per_seed.csv', synthetic.summarize, 'synthetic_summary.csv'),
    ('synthetic_per_seed.csv', synthetic.paired_tests, 'synthetic_significance.csv'),
]
report = {}
for source, function, target in checks:
    compare(function(read(source)), read(target))
    report[target] = 'PASS'
print(json.dumps(report, indent=2))
(ROOT / 'verification_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
