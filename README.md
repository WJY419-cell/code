# MPC-Rank

Code for *A Structured Cross-Layer Aggregation Method for Sparse Multi-Relational Ranking*.

The repository has three top-level entries:

```text
代码/
  metapath_coupling_experiment.py
  synthetic_boundary_experiment.py
  复现资料/                         # Original implementation, recorded results and checks
README.md
requirements.txt
```

## Install and verify

Use Python 3.12. From the repository root:

```sh
python -m pip install -r requirements.txt
python "代码/metapath_coupling_experiment.py" --mode verify
```

Verification requires no raw data. It checks the recorded summaries, paired effects, masks, analytic gradients, constrained solver and isolation rules. Do not run Python with `-O`.

## Run experiments

Obtain [Breakfast](https://preflib.org/dataset/00035), [SUSHI3](https://www.kamishima.net/sushi/) and [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) from their original providers. Place them in:

```text
data/breakfast/00035-00000001.csv
data/sushi3-2016/sushi3a.5000.10.order
data/sushi3-2016/sushi3.udata
data/ml-100k/u.data
data/ml-100k/u.user
```

Alternatively set MPC_DATA_ROOT to a directory with that layout. Raw data are not included.

```sh
python "代码/metapath_coupling_experiment.py" --seeds 12 --steps 750
python "代码/synthetic_boundary_experiment.py" --seeds 30 --steps 300
python "代码/metapath_coupling_experiment.py" --mode unseen
```

New experiment outputs go to `代码/运行结果/`, separately from the archived paper results. `--out` can override the real/synthetic output folder; MPC_UNSEEN_OUT overrides the masked-pair output folder. To summarize a full masked-pair rerun, set MPC_UNSEEN_OUT to its output directory and run `python "代码/复现资料/code/unseen_pairs/analyze_results.py"`. Use `--mode figures` to regenerate the English manuscript figures from the archived results and raw data; figures are written to `代码/复现资料/figures/`.

The two visible scripts are runnable entry points. The tested original implementation, results, vector figures, optimizer records and provenance are kept together in `代码/复现资料/`; the model and archived numerical values are unchanged. Protocol details and software citation metadata are under `代码/复现资料/说明与校验/`. No software reuse license has been selected.

The masked-pair tests concern unobserved pairs in a fixed catalog, not new items. Improvements depend on the comparator and support regime; this study does not establish uniform superiority over all baselines. Details are in `代码/复现资料/说明与校验/REPRODUCIBILITY.md`.

For GitHub, upload these three top-level entries to your repository root. This package includes no raw datasets, credentials, cover letter, manuscript review notes, or GitHub Actions configuration.
