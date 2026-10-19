"""Cross-validation engine: repeated stratified GROUP k-fold (a conversation is never split across folds)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .data import CID, Y
from .models import fit_model


def n_folds(df: pd.DataFrame, want: int) -> int:
    conv = df.groupby(CID)[Y].first()
    return max(2, min(want, int(conv.sum()), int((1 - conv).sum())))


def cross_val_oof(df, feats, name, cfg, repeats=None, folds=None, params=None, tune=False,
                  train_fraction=1.0, seed=None):
    """Out-of-fold P(failure) for every row, `repeats` times with different fold assignments.

    Returns array of shape (repeats, n_rows). Each row's score always comes from a model that never saw
    its conversation. `train_fraction` < 1 subsamples training conversations (learning curves).
    """
    ev = cfg["evaluation"]
    repeats = repeats or ev["cv_repeats"]
    seed = cfg["seed"] if seed is None else seed
    k = n_folds(df, folds or ev["cv_folds"])
    X, y, g = df[feats], df[Y].to_numpy(), df[CID].to_numpy()
    oof = np.full((repeats, len(df)), np.nan)
    for r in range(repeats):
        rng = np.random.default_rng(seed + r)
        skf = StratifiedGroupKFold(k, shuffle=True, random_state=seed + r)
        for tr, te in skf.split(X, y, g):
            if train_fraction < 1.0:
                convs = np.unique(g[tr])
                keep = rng.choice(convs, max(4, int(round(len(convs) * train_fraction))), replace=False)
                tr = tr[np.isin(g[tr], keep)]
            if len(set(y[tr])) < 2:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est, _ = fit_model(name, X.iloc[tr], y[tr], g[tr], cfg, tune=tune, params=params, seed=seed + r)
                oof[r, te] = est.predict_proba(X.iloc[te])[:, 1]
    return oof


def official_split_eval(df, feats, name, cfg, params=None, tune=False):
    """Fit on the given train split, score the given val/test splits (the plan's original protocol)."""
    tr = df[df["split"] == "train"]
    est, used = fit_model(name, tr[feats], tr[Y].to_numpy(), tr[CID].to_numpy(), cfg, tune=tune, params=params,
                          seed=cfg["seed"])
    out = {"params": used}
    for s in ("val", "test"):
        part = df[df["split"] == s]
        if len(part):
            out[s] = (part, est.predict_proba(part[feats])[:, 1])
    return est, out
