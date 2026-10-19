"""Probability calibration + threshold sweep/selection (plan Sec. 6-7)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Calibrator:
    """Maps raw model scores to calibrated P(failure). Pickle-safe, no sklearn-version coupling."""

    def __init__(self, method: str = "platt"):
        self.method = method
        self.m = None

    def fit(self, p, y):
        p, y = np.asarray(p, float), np.asarray(y)
        if self.method == "none" or len(set(y)) < 2:
            self.m = None
        elif self.method == "platt":
            self.m = LogisticRegression(C=1e4, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
        elif self.method == "isotonic":
            self.m = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
        else:
            raise ValueError(f"Unknown calibration method {self.method}")
        return self

    def transform(self, p):
        p = np.asarray(p, float)
        if self.m is None:
            return np.clip(p, 0.0, 1.0)
        if self.method == "platt":
            return np.clip(self.m.predict_proba(_logit(p).reshape(-1, 1))[:, 1], 0.0, 1.0)
        return np.clip(self.m.predict(p), 0.0, 1.0)


def cross_fit_calibrate(p, y, groups, method="platt", folds=5, seed=0):
    """Calibrated version of out-of-fold scores WITHOUT using a row's own conversation to fit its calibrator."""
    p, y, groups = np.asarray(p, float), np.asarray(y), np.asarray(groups)
    out = np.full(len(p), np.nan)
    k = max(2, min(folds, len(set(groups[y == 1])), len(set(groups[y == 0]))))
    for tr, te in StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(p, y, groups):
        out[te] = Calibrator(method).fit(p[tr], y[tr]).transform(p[te])
    return out


def sweep_thresholds(meta: pd.DataFrame, y, p, step: float = 0.01) -> pd.DataFrame:
    """For each threshold report the trade-off the plan asks for.

    turn-level : precision / recall(TPR) / FPR / F1 / flag_rate (share of turns that would trigger an intervention)
    conversation-level : conv_recall (failing conversations flagged at least once),
                         conv_false_alarm (succeeding conversations flagged at least once = "unnecessary intervention"),
                         first_flag_turn (mean first flagged turn among caught failing conversations).
    `meta` needs columns conversation_id and turn_index, aligned with y and p.
    """
    y, p = np.asarray(y), np.asarray(p, float)
    cid = meta["conversation_id"].to_numpy()
    turn = meta["turn_index"].to_numpy()
    conv_ids, inv = np.unique(cid, return_inverse=True)
    conv_y = np.zeros(len(conv_ids), int)
    conv_y[inv] = y
    rows = []
    for t in np.arange(step, 1.0, step):
        f = p >= t
        tp, fp = int((f & (y == 1)).sum()), int((f & (y == 0)).sum())
        fn, tn = int((~f & (y == 1)).sum()), int((~f & (y == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else np.nan
        rec = tp / (tp + fn) if tp + fn else np.nan
        fpr = fp / (fp + tn) if fp + tn else np.nan
        f1 = 2 * prec * rec / (prec + rec) if prec and rec and not np.isnan(prec + rec) else 0.0
        ever = np.zeros(len(conv_ids), bool)
        np.logical_or.at(ever, inv, f)
        first = np.full(len(conv_ids), np.inf)
        np.minimum.at(first, inv[f], turn[f])
        caught = ever & (conv_y == 1)
        rows.append({
            "threshold": round(float(t), 4), "precision": prec, "recall": rec, "fpr": fpr, "f1": f1,
            "flag_rate": float(f.mean()),
            "conv_recall": float(caught.sum() / max(1, (conv_y == 1).sum())),
            "conv_false_alarm": float((ever & (conv_y == 0)).sum() / max(1, (conv_y == 0).sum())),
            "first_flag_turn": float(first[caught].mean()) if caught.any() else np.nan,
        })
    return pd.DataFrame(rows)


def select_threshold(sweep: pd.DataFrame, cfg: dict) -> tuple[float, dict]:
    c = cfg["threshold"]
    s = sweep.copy()
    strat = c["strategy"]
    if strat == "youden":
        s["score"] = s["recall"] - s["fpr"]
        row = s.loc[s["score"].idxmax()]
    elif strat == "fbeta":
        b2 = c.get("beta", 1.0) ** 2
        s["score"] = (1 + b2) * s["precision"] * s["recall"] / (b2 * s["precision"] + s["recall"])
        row = s.loc[s["score"].fillna(-1).idxmax()]
    elif strat == "target_recall":
        ok = s[s["recall"] >= c["target_recall"]]
        row = ok.iloc[-1] if len(ok) else s.loc[s["recall"].idxmax()]  # highest threshold still meeting the target
    elif strat == "max_flag_rate":
        ok = s[s["flag_rate"] <= c["max_flag_rate"]]
        row = ok.loc[ok["recall"].idxmax()] if len(ok) else s.loc[s["flag_rate"].idxmin()]
    else:
        raise ValueError(f"Unknown threshold strategy {strat}")
    row = row.to_dict()
    return float(row["threshold"]), row
