"""Figures for the report. Matplotlib only, headless."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve


def reliability(results: dict, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name, r in results.items():
        y, p = r["eval_y"], r["eval_p"]
        try:
            fy, fx = calibration_curve(y, p, n_bins=6, strategy="quantile")
            ax.plot(fx, fy, "o-", label=f"{name} (ECE {r['metrics']['ece']:.3f})")
        except ValueError:
            pass
    ax.set_xlabel("predicted P(failure)")
    ax.set_ylabel("observed failure rate")
    ax.set_title("Reliability (calibrated, out-of-sample)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def roc(results: dict, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    for name, r in results.items():
        if len(set(r["eval_y"])) < 2:
            continue
        fpr, tpr, _ = roc_curve(r["eval_y"], r["eval_p"])
        ax.plot(fpr, tpr, label=f"{name} (AUROC {r['metrics']['auroc']:.3f})")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC (out-of-sample)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def threshold_sweep(sweep, chosen: float, path, title="Threshold sweep"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    a = axes[0]
    a.plot(sweep.threshold, sweep.recall, label="recall (turn)")
    a.plot(sweep.threshold, sweep.precision, label="precision (turn)")
    a.plot(sweep.threshold, sweep.flag_rate, label="flag rate")
    a.axvline(chosen, color="k", ls=":", label=f"chosen {chosen:.2f}")
    a.set_xlabel("threshold")
    a.set_title(title)
    a.legend(fontsize=8)
    b = axes[1]
    b.plot(sweep.threshold, sweep.conv_recall, label="failing convs flagged (>=1 turn)")
    b.plot(sweep.threshold, sweep.conv_false_alarm, label="succeeding convs flagged (false alarm)")
    b.axvline(chosen, color="k", ls=":")
    b.set_xlabel("threshold")
    b.set_title("Conversation level")
    b.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def bars(labels, values, path, title, ylabel="AUROC", errs=None, ref=0.5):
    fig, ax = plt.subplots(figsize=(max(5, 0.8 * len(labels) + 2), 4))
    ax.bar(range(len(labels)), values, yerr=errs, color="#4c78a8", capsize=3)
    ax.axhline(ref, color="k", ls="--", lw=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(min(0.3, np.nanmin(values) - 0.05), 1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def line(x, ys: dict, path, title, xlabel, ylabel="AUROC"):
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, v in ys.items():
        ax.plot(x, v, "o-", label=k)
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
