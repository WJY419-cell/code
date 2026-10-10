# Protocol and provenance

| Experiment | Script | Final settings | Archived outputs |
|---|---|---|---|
| Standard, support sweep, sparse targets | code/metapath_coupling_experiment.py | 12 seeds, steps 750; support caps and target counts recorded in results/run_manifest.json | results/real_*, support_sweep_*, long_tail_* |
| Controlled simulations | code/synthetic_boundary_experiment.py | 30 seeds, steps 300; 30 items, 8 layers, 17 scenarios | results/synthetic_* |
| Within-layer/global masked pairs | code/unseen_pairs/run_experiment.py | 12 seeds; 10/20/30% masks; primary rate 20%; six methods | supplement/unseen_pairs/ |

The three optimizer archives contain 6226 real-data fits, 8100 simulation fits and 4392 masked-pair fits, respectively. Every recorded fit passed the gradient infinity-norm tolerance of 2e-6. Solver verification independently compares finite differences and an SLSQP constrained fit on a small generated instance. Recorded-result checks validate summaries; they do not substitute for rerunning from raw data.

The main MovieLens protocol is transductive: catalog selection and non-tied comparison coverage use availability before the record split. The graph excludes win/loss direction but depends on rating equality and includes held-out-record coverage. The masked-pair protocol rebuilds the graph using training item co-occurrence, including ties without using rating values, and excludes masked pairs from coverage. The latter still retains the fixed catalog. Protocol changes are not solely pair removal.

All ranking records from the same original unit stay in one partition. Masked pairs are excluded from their specified fitting scope; global masks exclude a pair from all layers. Inner validation uses separate records and disjoint masks. pair_masks.npz stores 432 outer/inner arrays for 216 settings. The saved split audit reports zero contamination; validate_protocol.py also intentionally inserts a forbidden pair and checks that the invariant rejects it.

The independent optimizer archives and protocol/completion records are historical outputs. The archived core-solver hash is unchanged. The run wrapper has only standalone/output-path portability edits. The final manuscript has language edits after these experiments; historical manuscript hashes refer to the run-time version. Current file hashes are in release_manifest.json.

Split-based confidence intervals and tests are descriptive and conditional on the observed datasets. Overlapping resplits are not independent population replications. Mean metric rankings and small p-values do not establish practical dominance. No missing pair is given a pseudo-label during fitting.
