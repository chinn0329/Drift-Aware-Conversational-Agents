"""Diagnostics that answer: do the signals carry information BEYOND task source / model identity?

All functions take the canonical table from data.prepare(). Nothing here changes the trained model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cv import cross_val_oof
from .data import CID, GROUP, MODEL, TASK, TURN, Y
from .metrics import auroc, bootstrap_ci, bootstrap_diff_ci, within_group_auroc
from .models import fit_model

# which way each signal SHOULD point if it detects drift (+1: higher value -> more likely to fail)
EXPECTED = {"self_consistency_variance": +1, "contradiction_score": +1, "confidence": -1,
            "confidence_delta": -1, "silent_commitment_ratio": +1}


def _onehot(df: pd.DataFrame, col: str, prefix: str):
    d2, cols = df.copy(), []
    for v in sorted(d2[col].unique()):
        c = f"{prefix}_{v}"
        d2[c] = (d2[col] == v).astype(float)
        cols.append(c)
    return d2, cols


def incremental_value(df, feats, name, cfg, params=None, repeats=None) -> tuple[pd.DataFrame, dict]:
    """Source only  vs  source + signals  vs  signals only (out-of-fold, same folds).
    The paired CI on (source+signals) - (source only) is THE test of whether the signals add anything."""
    d2, src = _onehot(df, GROUP, "src")
    y, tasks = d2[Y].to_numpy(), d2[TASK].to_numpy()
    oof = {}
    for label, cols in (("source only", src), ("signals only", feats), ("source + signals", src + feats)):
        oof[label] = np.nanmean(cross_val_oof(d2, cols, name, cfg, params=params, repeats=repeats), axis=0)
    rows = []
    for label, p in oof.items():
        wg = within_group_auroc(y, p, d2[CID].to_numpy(), d2[GROUP].to_numpy())["macro"]
        lo, hi = bootstrap_ci(y, p, tasks, n=300, seed=cfg["seed"])
        rows.append({"model_inputs": label, "auroc_pooled": auroc(y, p), "ci_low": lo, "ci_high": hi,
                     "auroc_within_group": wg})
    diff, dlo, dhi = bootstrap_diff_ci(y, oof["source + signals"], oof["source only"], tasks, n=300, seed=cfg["seed"])
    return pd.DataFrame(rows), {"gain_over_source_only": diff, "ci_low": dlo, "ci_high": dhi}


def signal_by_source(df, feats, min_runs: int = 5, n_boot: int = 300, seed: int = 0) -> pd.DataFrame:
    """AUROC of each RAW signal (no model) inside each task source that has both outcomes.
    >0.5 means higher value -> more failures. `as_expected` checks the direction a drift signal should have and
    that the interval excludes 0.5. This is the audit table for Person A."""
    rows = []
    for g, part in df.groupby(GROUP):
        runs = part.groupby(CID)[Y].first()
        nf, ns = int(runs.sum()), int((1 - runs).sum())
        if min(nf, ns) < min_runs:
            continue
        for f in feats:
            x = part[f].to_numpy(dtype=float)
            x = np.where(np.isnan(x), np.nanmedian(x), x)
            a = auroc(part[Y].to_numpy(), x)
            lo, hi = bootstrap_ci(part[Y].to_numpy(), x, part[TASK].to_numpy(), n=n_boot, seed=seed)
            exp = EXPECTED.get(f, 0)
            ok = bool(exp) and ((exp > 0 and lo > 0.5) or (exp < 0 and hi < 0.5))
            rows.append({"source": g, "signal": f, "runs_failed": nf, "runs_succeeded": ns, "auroc": a,
                         "ci_low": lo, "ci_high": hi, "expected_direction": "higher=fail" if exp > 0 else "lower=fail",
                         "as_expected": ok})
    return pd.DataFrame(rows)


def within_task_paired(df, feats, n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    """For tasks where one run failed and another succeeded: mean signal of the failed run minus the succeeded run
    (run-level averages), averaged over tasks. Task difficulty cancels. Caution: the two runs usually come from
    different MODELS, so a difference may reflect the model, not drift."""
    run = df.groupby([TASK, CID], as_index=False).agg(y=(Y, "first"), model=(MODEL, "first"),
                                                       **{f: (f, "mean") for f in feats})
    diffs, n_cross = {}, 0
    for t, g in run.groupby(TASK):
        if g.y.nunique() < 2:
            continue
        diffs[t] = {f: g[g.y == 1][f].mean() - g[g.y == 0][f].mean() for f in feats}
        n_cross += int(g[g.y == 1].model.iloc[0] != g[g.y == 0].model.iloc[0])
    if not diffs:
        return pd.DataFrame(columns=["signal", "n_tasks", "mean_fail_minus_success", "ci_low", "ci_high"])
    keys, rng, rows = list(diffs), np.random.default_rng(seed), []
    for f in feats:
        v = np.array([diffs[k][f] for k in keys], dtype=float)
        boots = [np.nanmean(v[rng.choice(len(v), len(v), replace=True)]) for _ in range(n_boot)]
        rows.append({"signal": f, "n_tasks": len(keys), "tasks_where_models_differ": n_cross,
                     "mean_fail_minus_success": float(np.nanmean(v)), "ci_low": float(np.quantile(boots, 0.025)),
                     "ci_high": float(np.quantile(boots, 0.975))})
    return pd.DataFrame(rows)


def leave_one_group_out(df, feats, name, cfg, params, col: str, min_runs: int = 5,
                        unseen_tasks: bool | None = None) -> pd.DataFrame:
    """Train on every other value of `col` (model or task source), test on the held-out one.

    unseen_tasks=True  : also drop training runs of any TASK the held-out group touched (default for models).
                         Strict: the test tasks are genuinely new. If every model ran every task, nothing is
                         left to train on and the row says so.
    unseen_tasks=False : keep them. Measures transfer to a new model on tasks another model already attempted;
                         task difficulty can leak, so read it as optimistic.
    """
    if unseen_tasks is None:
        unseen_tasks = col == MODEL
    rows = []
    for g in sorted(df[col].unique()):
        te_mask = df[col] == g
        te = df[te_mask]
        tr = df[~te_mask & ~df[TASK].isin(te[TASK])] if unseen_tasks else df[~te_mask]
        runs = te.groupby(CID)[Y].first()
        nf, ns = int(runs.sum()), int((1 - runs).sum())
        row = {col: g, "protocol": "test tasks unseen" if unseen_tasks else "test tasks seen via other groups",
               "test_runs": len(runs), "test_failed": nf, "test_succeeded": ns, "train_runs": tr[CID].nunique(),
               "auroc": np.nan, "note": ""}
        if min(nf, ns) < min_runs:
            row["note"] = f"needs >= {min_runs} runs of each outcome to score"
        elif len(tr) == 0:
            row["note"] = "no training runs left: every task of this group was also run by the others"
        elif tr[Y].nunique() < 2:
            row["note"] = "training data has one outcome"
        else:
            est, _ = fit_model(name, tr[feats], tr[Y].to_numpy(), tr[TASK].to_numpy(), cfg, params=params,
                               seed=cfg["seed"])
            row["auroc"] = auroc(te[Y], est.predict_proba(te[feats])[:, 1])
        rows.append(row)
    return pd.DataFrame(rows)
