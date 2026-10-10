"""Verify masked-pair summaries, paired effects and saved masks without raw data."""
from pathlib import Path
from collections import defaultdict
import csv
import json
import math
import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / 'supplement/unseen_pairs'

def read(name):
    with (P / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def close(actual, expected):
    assert math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-11), (actual, expected)

groups = defaultdict(list)
for row in read('per_seed.csv'):
    groups[(row['dataset'], row['mode'], float(row['fraction']), row['method'])].append(row)
assert len(groups) == 108 and sum(map(len, groups.values())) == 1296
for rows in groups.values():
    assert {int(r['seed']) for r in rows} == set(range(12)) and len(rows) == 12
for row in read('summary.csv'):
    rows = groups[(row['dataset'], row['mode'], float(row['fraction']), row['method'])]
    assert int(row['n_runs']) == 12
    for metric in ['test_nll', 'accuracy']:
        x = np.array([float(r[metric]) for r in rows])
        for suffix, value in [('mean', x.mean()), ('std', x.std(ddof=1)), ('ci95', 1.96*x.std(ddof=1)/np.sqrt(12))]:
            close(row[f'{metric}_{suffix}'], value)
effects = read('paired_effects.csv')
assert len(effects) == 90
expected = []
rng = np.random.default_rng(20261005)
for (ds, mode, fraction, method), rows in sorted(groups.items()):
    if method != 'MPC-Rank':
        continue
    values = lambda rr: np.array([float(r['test_nll']) for r in sorted(rr, key=lambda r:int(r['seed']))])
    x = values(rows)
    for baseline in ['Pooled-BT', 'Layerwise-BT', 'Uncoupled-Joint', 'RankCentrality', 'Borda']:
        y = values(groups[(ds, mode, fraction, baseline)])
        d = y-x
        dstat = d.copy()
        dstat[np.abs(dstat) < 1e-8] = 0
        ps = 1. if not np.any(dstat) else float(wilcoxon(dstat, alternative='greater').pvalue)
        boot = d[rng.integers(0, 12, (5000, 12))].mean(axis=1)
        lo, hi = np.quantile(boot, [.025, .975])
        expected.append(dict(dataset=ds, mode=mode, fraction=fraction, baseline=baseline, mean_nll_gain=d.mean(), relative_nll_reduction=100*(y.mean()-x.mean())/y.mean(), wins=np.sum(dstat>0), ties=np.sum(dstat==0), losses=np.sum(dstat<0), bootstrap_low=lo, bootstrap_high=hi, p_one_sided=ps))
running = 0.
for rank, index in enumerate(sorted(range(90), key=lambda t:expected[t]['p_one_sided'])):
    running = max(running, (90-rank)*expected[index]['p_one_sided'])
    expected[index]['p_holm_all90'] = min(1., running)
for a, b in zip(effects, expected):
    assert a.keys() == b.keys()
    for key, value in b.items():
        if isinstance(value, (str,)):
            assert a[key] == value
        else:
            close(a[key], value)
audit = read('split_audit.csv')
assert len(audit) == 216 and all(int(r['leakage_violations']) == 0 for r in audit)
assert all(float(r['donor_coverage_fraction']) == 0 for r in audit if r['mode'] == 'global')
with np.load(P / 'pair_masks.npz') as masks:
    assert len(masks.files) == 432
    for row in audit:
        prefix = f"{row['dataset']}_{int(row['seed'])}_{row['mode']}_{float(row['fraction'])}"
        outer, inner = masks[prefix+'_outer'], masks[prefix+'_inner']
        assert outer.dtype == bool and inner.dtype == bool and outer.shape == inner.shape
        assert not np.any(outer & inner)
        if row['mode'] == 'global':
            assert np.all(outer == outer[:1]) and np.all(inner == inner[:1])
        if float(row['fraction']) != .3:
            next_fraction = .2 if float(row['fraction']) == .1 else .3
            larger = masks[f"{row['dataset']}_{int(row['seed'])}_{row['mode']}_{next_fraction}_outer"]
            assert np.all(~outer | larger)
report = {'summary_groups':108, 'run_rows':1296, 'paired_effects':90, 'mask_scenarios':216, 'mask_arrays':432, 'all_checks':'PASS'}
(ROOT / 'verification_unseen_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
