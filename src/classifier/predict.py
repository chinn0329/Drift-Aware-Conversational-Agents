"""Runtime interface used by Person C's orchestrator / intervention policy.

    from src.classifier.predict import predict_drift_score, explain, should_intervene

    features = {"self_consistency_variance": 0.4, "contradiction_score": 0.99, "confidence": 0.5,
                "confidence_delta": -0.3, "silent_commitment_ratio": 0.0}
    predict_drift_score(features)        # -> float in [0, 1] = calibrated P(conversation fails)
    should_intervene(features)           # -> bool (score >= calibrated threshold)
    explain(features)["dominant_signal"] # -> "self_consistency" | "contradiction" | "confidence" | ...

Features may be a dict (preferred), a pandas Series/DataFrame row, or a list in the agreed 5-feature order.
Unknown / missing features are tolerated (missing -> NaN -> imputed), so the contract survives schema changes.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import REPO_ROOT

_CACHE: dict = {}


def _models_dir(models_dir=None) -> Path:
    d = models_dir or os.environ.get("DRIFT_MODELS_DIR") or "models"
    d = Path(d)
    return d if d.is_absolute() else REPO_ROOT / d


def load_bundle(models_dir=None, model: str | None = None) -> dict:
    """Load the latest primary bundle (or a specific model of the latest run). Cached."""
    mdir = _models_dir(models_dir)
    key = (str(mdir), model)
    if key in _CACHE:
        return _CACHE[key]
    latest = json.loads((mdir / "latest.json").read_text(encoding="utf-8"))
    fname = "primary.joblib" if model is None else f"{model}.joblib"
    bundle = joblib.load(mdir / latest["run_id"] / fname)
    _CACHE[key] = bundle
    return bundle


def clear_cache():
    _CACHE.clear()


class TrajectoryTracker:
    """Per-conversation helper for models trained with data.trajectory_features: true.

        tracker = TrajectoryTracker()                       # one per conversation
        feats = tracker.update(raw_turn_features)           # adds <signal>__cummax / __cummean / __dfirst
        score = predict_drift_score(feats)
        tracker.truncate(k)                                 # after a ROLLBACK to turn k: forget turns > k

    For a model WITHOUT trajectory features it is a harmless pass-through, so callers can always use it.
    """

    def __init__(self, models_dir=None, base=None, stats=None):
        if base is None:
            tr = load_bundle(models_dir)["meta"].get("trajectory") or {}
            base, stats = tr.get("base", []), tr.get("stats", [])
        self.base, self.stats = list(base), list(stats or [])
        self.history: list[dict] = []

    def update(self, features) -> dict:
        feats = dict(features)
        self.history.append({f: float(feats.get(f, np.nan)) for f in self.base})
        out = dict(feats)
        for f in self.base:
            v = np.array([h[f] for h in self.history], dtype=float)
            ok = v[~np.isnan(v)]
            if "cummax" in self.stats:
                out[f"{f}__cummax"] = float(ok.max()) if len(ok) else np.nan
            if "cummean" in self.stats:
                out[f"{f}__cummean"] = float(ok.mean()) if len(ok) else np.nan
            if "dfirst" in self.stats:
                out[f"{f}__dfirst"] = float(v[-1] - ok[0]) if len(ok) else np.nan
        return out

    def truncate(self, keep_turns: int):
        self.history = self.history[:keep_turns]


def _frame(features, bundle: dict) -> pd.DataFrame:
    if isinstance(features, pd.DataFrame):
        df = features.copy()
    elif isinstance(features, pd.Series):
        df = features.to_frame().T
    elif isinstance(features, Mapping):
        df = pd.DataFrame([dict(features)])
    else:  # sequence in the agreed contract order
        order = bundle["input_order"]
        vals = list(features)
        if len(vals) != len(order):
            raise ValueError(f"Expected {len(order)} features in order {order}, got {len(vals)}")
        df = pd.DataFrame([dict(zip(order, vals))])
    if bundle["meta"].get("trajectory"):
        missing = [c for c in bundle["features"] if "__" in c and c not in df.columns]
        if missing:
            raise ValueError("This model was trained with trajectory features; pass the dict returned by "
                             f"TrajectoryTracker.update(...) (missing {missing[:3]}...).")
    for c in bundle["features"]:
        if c not in df.columns:
            df[c] = np.nan
    return df[bundle["features"]].apply(pd.to_numeric, errors="coerce")


def _score(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    raw = bundle["model"].predict_proba(X)[:, 1]
    return np.clip(bundle["calibrator"].transform(raw), 0.0, 1.0)


def predict_drift_scores(features_df: pd.DataFrame, models_dir=None) -> np.ndarray:
    b = load_bundle(models_dir)
    return _score(b, _frame(features_df, b))


def predict_drift_score(features, models_dir=None) -> float:
    """THE interface agreed in Week 1: features -> float in [0, 1]."""
    b = load_bundle(models_dir)
    return float(_score(b, _frame(features, b))[0])


def get_threshold(models_dir=None) -> float:
    return float(load_bundle(models_dir)["threshold"])


def should_intervene(features, models_dir=None) -> bool:
    return predict_drift_score(features, models_dir) >= get_threshold(models_dir)


def explain(features, models_dir=None) -> dict:
    """Which signal group is pushing the score up the most?

    Model-agnostic occlusion: replace one signal group by its value on HEALTHY (successful) conversations and
    see how far the drift score drops. contribution > 0 means "this signal is raising the score".
    Works identically for logistic regression, XGBoost and the MLP.
    """
    b = load_bundle(models_dir)
    X = _frame(features, b)
    full = float(_score(b, X)[0])
    contrib = {}
    for gname, cols in b["signal_groups"].items():
        base = set(cols)                          # a group also owns its derived <signal>__<stat> columns
        cols = [c for c in b["features"] if c.split("__")[0] in base]
        if not cols:
            contrib[gname] = 0.0
            continue
        Xr = X.copy()
        for c in cols:
            Xr[c] = b["healthy_ref"][c]
        contrib[gname] = full - float(_score(b, Xr)[0])
    dom = max(contrib, key=contrib.get)
    return {"drift_score": full, "threshold": float(b["threshold"]), "intervene": full >= b["threshold"],
            "dominant_signal": dom if contrib[dom] > 0 else None, "contributions": contrib}
