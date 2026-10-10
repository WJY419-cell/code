"""Generate publication figures for the MPC-Rank manuscript.

Every quantitative panel is reconstructed from the experiment CSV files or
from the same public-data loaders used by ``metapath_coupling_experiment.py``.
No values are entered manually and no unreported experiment is implied.
"""

import csv
import importlib.util
import math
import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/matplotlib-cache"))
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np


RESULTS = ROOT / "results"
OUT = ROOT / "figures"
SOURCE = ROOT / "code/metapath_coupling_experiment.py"


def load_experiment_module():
    spec = importlib.util.spec_from_file_location("mpc_exp", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def font(size=9, bold=False):
    return FontProperties(family="DejaVu Sans", size=size, weight="bold" if bold else "normal")


FP = font(9)
FP_SMALL = font(7.5)
FP_TITLE = font(9.3, True)
COLORS = {
    "Borda": "#D8B65A",
    "RankCentrality": "#70AFC6",
    "Layerwise-BT": "#D88B6A",
    "Pooled-BT": "#87AE78",
    "MPC-Rank": "#8E6BBE",
}
METHODS = ["Borda", "RankCentrality", "Pooled-BT", "MPC-Rank"]
PROB_METHODS = ["Layerwise-BT", "Pooled-BT", "MPC-Rank"]
MARKERS = {"Borda": "o", "RankCentrality": "s", "Layerwise-BT": "D",
           "Pooled-BT": "^", "MPC-Rank": "o"}
LINESTYLES = {"Layerwise-BT": "--", "Pooled-BT": "-.", "MPC-Rank": "-"}
DATASETS = ["Breakfast-6Context", "SUSHI-RegionPaths", "MovieLens-OccupationPaths"]
DATASET_SHORT = {DATASETS[0]: "Breakfast", DATASETS[1]: "SUSHI", DATASETS[2]: "MovieLens"}

plt.rcParams.update({
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def save(fig, stem: str):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(ax, xy, w, h, text, fc="#F4F7FA", ec="#315B7D", size=8):
    p = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                       facecolor=fc, edgecolor=ec, linewidth=1.2)
    ax.add_patch(p)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontproperties=font(size), color="#1E2A35")
    return p


def arrow(ax, p1, p2, color="#6C8295"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=10,
                                linewidth=1.1, color=color))


def plot_framework():
    fig, ax = plt.subplots(figsize=(11.2, 3.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.94, "(a) Items, relations, and contexts", fontproperties=FP_TITLE)
    ax.text(0.355, 0.94, "(b) Relation-induced sparse layers", fontproperties=FP_TITLE)
    ax.text(0.69, 0.94, "(c) Cross-layer coupling and outputs", fontproperties=FP_TITLE)
    ax.plot([0.33, 0.33], [0.08, 0.9], color="#D7DEE5", lw=1)
    ax.plot([0.665, 0.665], [0.08, 0.9], color="#D7DEE5", lw=1)

    # Heterogeneous network: objects, users/types, and typed paths.
    obj = [(0.06, 0.68), (0.19, 0.72), (0.10, 0.28), (0.25, 0.31)]
    ctx = [(0.15, 0.50), (0.275, 0.57)]
    for idx, (x, y) in enumerate(obj, 1):
        ax.add_patch(Rectangle((x - 0.026, y - 0.026), 0.052, 0.052,
                               fc="#DCEAF5", ec="#24577A", lw=1.2))
        ax.text(x, y, f"v{idx}", ha="center", va="center", fontsize=8)
    for idx, (x, y) in enumerate(ctx, 1):
        ax.add_patch(Circle((x, y), 0.03, fc="#F7E2B8", ec="#A16A19", lw=1.2))
        ax.text(x, y, f"c{idx}", ha="center", va="center", fontsize=8)
    edges = [(obj[0], ctx[0], "#4C92C3"), (ctx[0], obj[1], "#4C92C3"),
             (obj[2], ctx[0], "#4C92C3"), (obj[1], ctx[1], "#D88B3C"),
             (ctx[1], obj[3], "#D88B3C"), (obj[0], ctx[1], "#D88B3C")]
    for a, b, c in edges:
        ax.plot([a[0], b[0]], [a[1], b[1]], color=c, lw=1.4, alpha=0.9, zorder=0)
    # Layer panels.
    layer_y = [0.70, 0.47, 0.24]
    layer_c = ["#4C92C3", "#D88B3C", "#6A9F58"]
    # Same item set in each layer; different observed pairs illustrate missingness.
    layer_edges = [[(0, 1), (1, 2)],
                   [(0, 2), (2, 3)],
                   [(0, 3), (1, 2)]]
    for k, (y, c) in enumerate(zip(layer_y, layer_c), 1):
        box(ax, (0.37, y - 0.085), 0.25, 0.15, "", fc="#F8FAFC", ec=c, size=8)
        ax.text(0.385, y + 0.031, f"Layer $G^{{({k})}}$", ha="left", va="center",
                fontproperties=font(7.0), color="#27343E")
        ax.text(0.607, y + 0.031, "Local, incomplete", ha="right", va="center",
                fontproperties=font(6.0), color="#5C6770")
        xs = np.linspace(0.405, 0.585, 4)
        ys = y + np.array([-0.042, -0.006, -0.042, -0.006])
        for a, b in layer_edges[k-1]:
            ax.plot([xs[a], xs[b]], [ys[a], ys[b]], color=c, lw=1.2, zorder=1)
        for idx, (x, node_y) in enumerate(zip(xs, ys), 1):
            ax.add_patch(Circle((x, node_y), 0.012, fc="white", ec=c, lw=1, zorder=2))
            ax.text(x, node_y, f"v{idx}", ha="center", va="center", fontsize=4.8, zorder=3)

    # Coupling graph and model.
    centers = [(0.73, 0.69), (0.82, 0.69), (0.775, 0.53)]
    for i, (x, y) in enumerate(centers, 1):
        ax.add_patch(Circle((x, y), 0.035, fc="#E4F0E1", ec="#4E8052", lw=1.2))
        ax.text(x, y, f"$b^{{({i})}}$", ha="center", va="center", fontsize=8)
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        ax.plot([centers[a][0], centers[b][0]], [centers[a][1], centers[b][1]],
                color="#79A56D", lw=2, alpha=0.75)
    ax.text(0.775, 0.79, r"$S_{k\ell}$", ha="center", va="center", fontsize=9, color="#3A6840")
    box(ax, (0.69, 0.25), 0.17, 0.15,
        r"$u$: shared scores" "\n" r"$B$: layer offsets" "\n" r"$\mathrm{tr}(BL_SB^\top)$: coupling",
        fc="#EEF5FA", ec="#24577A", size=6.7)
    arrow(ax, (0.775, 0.48), (0.775, 0.41))
    box(ax, (0.88, 0.43), 0.095, 0.39,
        "Overall\nranking\n\nWithin-layer\nprobabilities\n\nPredicted\nlayer margins",
        fc="#FFF8E8", ec="#A16A19", size=6.3)
    arrow(ax, (0.86, 0.325), (0.905, 0.47))
    save(fig, "framework_metapath_coupling")


def plot_layer_structure(exp):
    datasets = [exp.breakfast_dataset(), exp.sushi_dataset(), exp.movielens_dataset()]
    fig = plt.figure(figsize=(11.2, 6.25))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.7, 2.4], hspace=0.18, wspace=0.28)
    for col, ds in enumerate(datasets):
        counts = Counter(r.layer for r in ds.records)
        vals = np.array([counts[k] for k in range(len(ds.layer_names))])
        axb = fig.add_subplot(gs[0, col])
        axb.bar(np.arange(1, len(vals) + 1), vals, color="#6D93AF", width=0.78)
        bar_ticks = np.arange(1, len(vals) + 1) if len(vals) <= 10 else np.array([1, 6, 11, 16, 21])
        axb.set_xticks(bar_ticks)
        axb.set_title(f"({chr(97 + col)}) {DATASET_SHORT[ds.name]}", loc="left", fontproperties=FP_TITLE)
        axb.set_ylabel("Records per layer", fontproperties=FP_SMALL)
        axb.tick_params(labelsize=7)
        for side in ("top", "right"):
            axb.spines[side].set_visible(False)
        axh = fig.add_subplot(gs[1, col])
        s = np.asarray(ds.similarity, float).copy()
        if np.max(s) > 0:
            s /= np.max(s)
        im = axh.imshow(s, cmap="Blues", vmin=0, vmax=1, interpolation="nearest", aspect="auto")
        axh.set_xlabel("Relation layer", fontproperties=FP_SMALL)
        axh.set_ylabel("Relation layer", fontproperties=FP_SMALL)
        if len(ds.layer_names) <= 10:
            labels = [str(i + 1) for i in range(len(ds.layer_names))]
            axh.set_xticks(range(len(labels)))
            axh.set_xticklabels(labels, fontsize=7)
            axh.set_yticks(range(len(labels)))
            axh.set_yticklabels(labels, fontsize=7)
        else:
            ticks = [0, 5, 10, 15, 20]
            axh.set_xticks(ticks)
            axh.set_xticklabels([str(x + 1) for x in ticks], fontsize=7)
            axh.set_yticks(ticks)
            axh.set_yticklabels([str(x + 1) for x in ticks], fontsize=7)
        if col == 2:
            cbar = fig.colorbar(im, ax=axh, fraction=0.046, pad=0.04)
            cbar.set_label("Normalized layer similarity", fontproperties=FP_SMALL)
            cbar.ax.tick_params(labelsize=7)
    save(fig, "layer_structure")


def plot_real_benchmark():
    rows = read_csv(RESULTS / "real_summary.csv")
    lookup = {(r["dataset"], r["method"]): r for r in rows}
    metrics = [("kendall_tau", "Kendall $\\tau$", True),
               ("pair_accuracy", "Pairwise accuracy", True),
               ("test_nll", "Test NLL", False)]
    fig, axes = plt.subplots(3, 3, figsize=(11.2, 7.2))
    x = np.arange(len(METHODS))
    for i, ds in enumerate(DATASETS):
        for j, (metric, title, _higher) in enumerate(metrics):
            ax = axes[i, j]
            means = [float(lookup[(ds, m)][f"{metric}_mean"]) for m in METHODS]
            cis = [float(lookup[(ds, m)][f"{metric}_ci95"]) for m in METHODS]
            for xi, m, y, e in zip(x, METHODS, means, cis):
                ax.errorbar(xi, y, yerr=e, fmt="o", ms=5.2, capsize=2.5,
                            color=COLORS[m], ecolor=COLORS[m], lw=1.2)
            ax.plot(x, means, color="#B9C3CC", lw=0.8, zorder=0)
            ax.grid(axis="y", color="#E7EBEF", lw=0.7)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(axis="y", labelsize=7)
            if i == 0:
                ax.set_title(title, fontproperties=FP_TITLE)
            if j == 0:
                ax.set_ylabel(f"({chr(97+i)}) {DATASET_SHORT[ds]}", fontproperties=FP_TITLE)
            if i == 2:
                ax.set_xticks(x)
                ax.set_xticklabels(["Borda", "RC", "Pooled", "MPC"], rotation=25, ha="right", fontsize=7)
            else:
                ax.set_xticks(x, [])
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=COLORS[m], label=m) for m in METHODS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.94], pad=0.8, h_pad=0.8, w_pad=0.8)
    save(fig, "real_benchmark_panel")


def bootstrap_ci(values, seed=20260826, n_boot=5000):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return values.mean(), lo, hi


def plot_paired_effects():
    rows = read_csv(RESULTS / "real_per_seed.csv")
    sig = read_csv(RESULTS / "paired_significance.csv")
    pvals = {(r["dataset"], r["metric"]): float(r["wilcoxon_one_sided_p"])
             for r in sig if r["baseline"] == "Pooled-BT"}
    raw = {}
    for r in rows:
        raw[(r["dataset"], int(r["seed"]), r["method"])] = r
    metrics = [("test_nll", "Relative NLL reduction (%)", 100.0),
               ("pair_accuracy", "Accuracy improvement (pp)", 100.0),
               ("kendall_tau", r"Kendall $\tau$ improvement ($\times10^{-3}$)", 1000.0)]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.45), constrained_layout=True)
    for j, (metric, title, scale) in enumerate(metrics):
        ax = axes[j]
        for i, ds in enumerate(DATASETS):
            diffs = []
            for seed in range(12):
                a = float(raw[(ds, seed, "MPC-Rank")][metric])
                b = float(raw[(ds, seed, "Pooled-BT")][metric])
                raw_diff = (b - a) if metric == "test_nll" else (a - b)
                diffs.append(scale * raw_diff)
            if metric == "test_nll":
                # Paired bootstrap of the same split-level observations.
                # Recompute the ratio of means, not the mean of ratios.
                base = np.array([float(raw[(ds, seed, "Pooled-BT")][metric]) for seed in range(12)])
                mpc = np.array([float(raw[(ds, seed, "MPC-Rank")][metric]) for seed in range(12)])
                mean = 100.0 * (base.mean() - mpc.mean()) / base.mean()
                rng = np.random.default_rng(20260826 + 31 * j + i)
                indices = rng.integers(0, len(base), size=(5000, len(base)))
                boot_base = base[indices].mean(axis=1)
                boot_mpc = mpc[indices].mean(axis=1)
                boot = 100.0 * (boot_base - boot_mpc) / boot_base
                lo, hi = np.quantile(boot, [0.025, 0.975])
            else:
                mean, lo, hi = bootstrap_ci(diffs, seed=20260826 + 31 * j + i)
            ax.errorbar(mean, i, xerr=[[mean - lo], [hi - mean]], fmt="o", ms=6,
                        capsize=3, color="#184E77", ecolor="#5E88A4", lw=1.4)
            p = pvals[(ds, metric)]
            p_label = "$p<0.001$" if p < 0.001 else f"$p={p:.3f}$"
            ax.text(max(hi, mean) + max(abs(hi-lo) * 0.12, [0.08, 0.10, 0.55][j]), i, p_label,
                    va="center", fontproperties=font(6.8), color="#4A5660")
        ax.axvline(0, color="#9AA5AE", lw=1, ls="--")
        ax.set_yticks(range(3))
        ax.set_yticklabels([DATASET_SHORT[d] for d in DATASETS], fontproperties=FP_SMALL)
        ax.invert_yaxis()
        ax.set_title(f"({chr(97+j)}) {title}", loc="left", fontproperties=FP_TITLE)
        ax.grid(axis="x", color="#E7EBEF", lw=0.7)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="x", labelsize=7)
    save(fig, "paired_effects_pooled")


def plot_synthetic_mechanism():
    rows = read_csv(RESULTS / "synthetic_summary.csv")
    support = [r for r in rows if r["family"] == "support"]
    boundary = [r for r in rows if r["family"] == "boundary"]
    methods = ["Pooled-BT", "Layerwise-BT", "Uncoupled-Joint", "MPC-Rank"]
    colors = {"Pooled-BT": "#87AE78", "Layerwise-BT": "#D88B6A",
              "Uncoupled-Joint": "#6C8295", "MPC-Rank": "#8E6BBE"}
    markers = {"Pooled-BT": "^", "Layerwise-BT": "D",
               "Uncoupled-Joint": "s", "MPC-Rank": "o"}
    labels = {"Pooled-BT": "Pooled-BT", "Layerwise-BT": "Layerwise-BT",
              "Uncoupled-Joint": "Uncoupled-Joint", "MPC-Rank": "MPC-Rank"}
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5),
                             gridspec_kw={"width_ratios": [1.25, 1.25, 1.0]},
                             constrained_layout=True)
    supports = sorted({int(r["target_support"]) for r in support})
    xpos = np.arange(len(supports))
    for ax, metric, ylabel, panel in (
        (axes[0], "target_test_nll", "Target-layer test NLL", "(a) Low-support prediction"),
        (axes[1], "target_theta_rmse", "Target-layer score RMSE", "(b) Parameter recovery"),
    ):
        for method in methods:
            vals = sorted((r for r in support if r["method"] == method),
                          key=lambda r: int(r["target_support"]))
            mean = np.array([float(r[f"{metric}_mean"]) for r in vals])
            ci = np.array([float(r[f"{metric}_ci95"]) for r in vals])
            ax.plot(xpos, mean, color=colors[method], marker=markers[method],
                    lw=1.45, ms=4.2, label=labels[method])
            ax.fill_between(xpos, mean - ci, mean + ci, color=colors[method],
                            alpha=0.10, linewidth=0)
        ax.set_xticks(xpos, [str(x) for x in supports], fontsize=7)
        ax.set_xlabel("Target-layer training comparisons", fontproperties=FP_SMALL)
        ax.set_ylabel(ylabel, fontproperties=FP_SMALL)
        ax.set_title(panel, loc="left", fontproperties=FP_TITLE)
        ax.grid(axis="y", color="#E7EBEF", lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=7)

    hs = [0.0, 0.3, 0.6, 1.0]
    qs = [0.0, 0.5, 1.0]
    heat = np.zeros((len(hs), len(qs)))
    for i, h in enumerate(hs):
        for j, q in enumerate(qs):
            cell = [r for r in boundary if math.isclose(float(r["heterogeneity"]), h)
                    and math.isclose(float(r["graph_corruption"]), q)]
            mpc = float(next(r for r in cell if r["method"] == "MPC-Rank")["target_test_nll_mean"])
            best = min(float(r["target_test_nll_mean"]) for r in cell if r["method"] != "MPC-Rank")
            heat[i, j] = 1000.0 * (best - mpc)
    vmax = max(abs(heat.min()), abs(heat.max()), 1e-6)
    im = axes[2].imshow(heat, cmap="RdBu", aspect="auto",
                        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax))
    axes[2].set_xticks(range(len(qs)), [str(x) for x in qs], fontsize=7)
    axes[2].set_yticks(range(len(hs)), [str(x) for x in hs], fontsize=7)
    axes[2].set_xlabel("Layer-graph misspecification $q$", fontproperties=FP_SMALL)
    axes[2].set_ylabel(r"Interlayer heterogeneity $\delta$", fontproperties=FP_SMALL)
    axes[2].set_title("(c) Applicability boundary", loc="left", fontproperties=FP_TITLE)
    for i in range(len(hs)):
        for j in range(len(qs)):
            axes[2].text(j, i, f"{heat[i, j]:+.1f}", ha="center", va="center",
                         fontsize=7, color="white" if abs(heat[i, j]) > 0.62 * vmax else "#24313B")
    cb = fig.colorbar(im, ax=axes[2], fraction=0.055, pad=0.04)
    cb.set_label(r"NLL gain over best baseline ($\times10^{-3}$)", fontproperties=font(6.8))
    cb.ax.tick_params(labelsize=6.5)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 1.07), frameon=False, prop=font(6.8))
    save(fig, "synthetic_mechanism_boundary")


def plot_support_curves(csv_name, x_key, stem):
    rows = read_csv(RESULTS / csv_name)
    lookup = {(r["dataset"], int(r[x_key]), r["method"]): r for r in rows}
    levels = {ds: sorted({int(r[x_key]) for r in rows if r["dataset"] == ds})
              for ds in DATASETS}
    metrics = [("test_nll", "Test NLL"), ("pair_accuracy", "Pairwise accuracy")]
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 5.7), sharex="col")
    for col, ds in enumerate(DATASETS):
        xs = levels[ds]
        for row, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            for method in PROB_METHODS:
                means = np.array([float(lookup[(ds, x, method)][f"{metric}_mean"]) for x in xs])
                cis = np.array([float(lookup[(ds, x, method)][f"{metric}_ci95"]) for x in xs])
                ax.plot(xs, means, color=COLORS[method], marker=MARKERS[method],
                        ls=LINESTYLES[method], lw=1.5, ms=4.5, label=method)
                ax.fill_between(xs, means - cis, means + cis, color=COLORS[method],
                                alpha=0.12, linewidth=0)
            ax.grid(axis="y", color="#E7EBEF", lw=0.7)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(labelsize=7)
            ax.set_ylabel(ylabel, fontproperties=FP_SMALL)
            if row == 0:
                ax.set_title(f"({chr(97 + col)}) {DATASET_SHORT[ds]}", loc="left",
                             fontproperties=FP_TITLE)
            if row == 1:
                xlabel = "Training rankings per target layer" if x_key == "target_support" else "Training rankings per layer"
                ax.set_xlabel(xlabel, fontproperties=FP_SMALL)
                ax.set_xticks(xs)
    handles = [plt.Line2D([0], [0], color=COLORS[m], marker=MARKERS[m],
                          ls=LINESTYLES[m], lw=1.5, label=m) for m in PROB_METHODS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=3, frameon=False, prop=FP_SMALL)
    fig.tight_layout(rect=[0, 0, 1, 0.94], pad=0.8, h_pad=0.9, w_pad=0.9)
    save(fig, stem)


def main():
    exp = load_experiment_module()
    plot_framework()
    plot_layer_structure(exp)
    plot_support_curves("support_sweep_summary.csv", "records_per_layer", "real_support_sweep")
    plot_support_curves("long_tail_summary.csv", "target_support", "long_tail_target_layers")
    plot_real_benchmark()
    plot_paired_effects()
    plot_synthetic_mechanism()
    print(f"Wrote publication figures to {OUT}")


if __name__ == "__main__":
    main()
