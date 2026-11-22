"""Metrics: discrimination (AUROC / PR-AUC), calibration (Brier / ECE), CIs and by-turn breakdown."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def auroc(y, p) -> float:
    y = np.asarray(y)
    return float("nan") if len(set(y)) < 2 else float(roc_auc_score(y, p))


def prauc(y, p) -> float:
    y = np.asarray(y)
    return float("nan") if len(set(y)) < 2 else float(average_precision_score(y, p))


def ece(y, p, bins: int = 10) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    out = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            out += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(out)


def summarize(y, p) -> dict:
    y, p = np.asarray(y), np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return {"auroc": auroc(y, p), "pr_auc": prauc(y, p), "brier": float(brier_score_loss(y, p)),
            "ece": ece(y, p), "log_loss": float(log_loss(y, p, labels=[0, 1])), "base_rate": float(y.mean())}


def bootstrap_ci(y, p, groups, fn=auroc, n: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Conversation-level bootstrap (rows of one conversation are never split apart)."""
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    rng = np.random.default_rng(seed)
    idx_by = {g: np.where(groups == g)[0] for g in np.unique(groups)}
    keys = list(idx_by)
    vals = []
    for _ in range(n):
        pick = rng.choice(len(keys), len(keys), replace=True)
        idx = np.concatenate([idx_by[keys[i]] for i in pick])
        v = fn(y[idx], p[idx])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))


TURN_BINS = [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 5, "4-5"), (6, 99, "6+")]


def auroc_by_turn(turns, y, p) -> pd.DataFrame:
    """Does the score still separate outcomes EARLY in the conversation? (leading-indicator check)"""
    turns, y, p = np.asarray(turns), np.asarray(y), np.asarray(p)
    rows = []
    for lo, hi, lab in TURN_BINS:
        m = (turns >= lo) & (turns <= hi)
        rows.append({"turn": lab, "rows": int(m.sum()), "failed_share": float(y[m].mean()) if m.any() else np.nan,
                     "auroc": auroc(y[m], p[m]) if m.any() else np.nan})
    return pd.DataFrame(rows)


def within_group_auroc(y, p, cids, groups, min_conv_per_class: int = 3) -> dict:
    """AUROC computed separately inside each group (e.g. each task source), for groups where BOTH outcomes
    occur. This removes the 'which benchmark is it?' shortcut: if every conversation of a group failed, a model
    can look good pooled just by recognising that group.

    Returns {"per_group": {group: auroc}, "macro": mean or nan}.
    """
    y, p, cids, groups = map(np.asarray, (y, p, cids, groups))
    out = {}
    for g in np.unique(groups):
        mk = groups == g
        conv_y = pd.Series(y[mk]).groupby(cids[mk]).first()
        if min(int(conv_y.sum()), int((1 - conv_y).sum())) >= min_conv_per_class:
            out[str(g)] = auroc(y[mk], p[mk])
    return {"per_group": out, "macro": float(np.mean(list(out.values()))) if out else float("nan")}


def bootstrap_diff_ci(y, p_a, p_b, groups, fn=auroc, n: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Paired bootstrap of fn(p_a) - fn(p_b), resampling whole groups (tasks). Returns (diff, lo, hi).
    Use it to ask 'does adding the signals improve on source alone?' without eyeballing overlapping intervals."""
    y, p_a, p_b, groups = map(np.asarray, (y, p_a, p_b, groups))
    rng = np.random.default_rng(seed)
    idx_by = {g: np.where(groups == g)[0] for g in np.unique(groups)}
    keys = list(idx_by)
    diffs = []
    for _ in range(n):
        pick = rng.choice(len(keys), len(keys), replace=True)
        idx = np.concatenate([idx_by[keys[i]] for i in pick])
        a, b = fn(y[idx], p_a[idx]), fn(y[idx], p_b[idx])
        if not (np.isnan(a) or np.isnan(b)):
            diffs.append(a - b)
    point = fn(y, p_a) - fn(y, p_b)
    if not diffs:
        return float(point), float("nan"), float("nan")
    return float(point), float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))


def _task_pairs(tasks, runs, y, p):
    """Run-level mean score per run, then every (failed run, succeeded run) pair of the SAME task."""
    d = pd.DataFrame({"task": tasks, "run": runs, "y": y, "p": p}).groupby(["task", "run"], as_index=False) \
        .agg(y=("y", "first"), p=("p", "mean"))
    out = {}
    for t, g in d.groupby("task"):
        f, s = g[g.y == 1].p.to_numpy(), g[g.y == 0].p.to_numpy()
        if len(f) and len(s):
            out[t] = (f[:, None] > s[None, :]).astype(float) + 0.5 * (f[:, None] == s[None, :])
    return out


def within_task_auroc(tasks, runs, y, p, n_boot: int = 500, seed: int = 0) -> dict:
    """AUROC using ONLY pairs of runs of the same task that ended differently (e.g. two models on one task).
    Task difficulty cancels out, so this is the cleanest check that a score tracks what went wrong in the run.
    Returns {"auroc", "ci_low", "ci_high", "n_tasks", "n_pairs"} (NaN when no task has both outcomes)."""
    pairs = _task_pairs(tasks, runs, y, p)
    if not pairs:
        return {"auroc": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_tasks": 0, "n_pairs": 0}
    keys = list(pairs)
    point = float(np.concatenate([m.ravel() for m in pairs.values()]).mean())
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), len(keys), replace=True)
        boots.append(np.concatenate([pairs[keys[i]].ravel() for i in pick]).mean())
    return {"auroc": point, "ci_low": float(np.quantile(boots, 0.025)), "ci_high": float(np.quantile(boots, 0.975)),
            "n_tasks": len(keys), "n_pairs": int(sum(m.size for m in pairs.values()))}
