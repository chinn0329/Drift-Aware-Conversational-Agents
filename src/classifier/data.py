"""Load, validate and standardise the turn-level table written by Person A.

Canonical columns after loading (independent of the file's own column names):
    conversation_id, turn_index, <features...>, y, split, group
where y = 1 means the conversation FAILED (that is what drift_score predicts).

Run `python -m src.classifier.data` for a data-quality report.
"""
from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, resolve

CID, TURN, Y, SPLIT, GROUP = "conversation_id", "turn_index", "y", "split", "group"
_TRUE = {"true", "1", "yes", "pass", "passed", "success", "t"}
_FALSE = {"false", "0", "no", "fail", "failed", "failure", "f"}


def _to_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float) > 0.5
    low = s.astype(str).str.strip().str.lower()
    bad = ~low.isin(_TRUE | _FALSE)
    if bad.any():
        raise ValueError(f"Cannot parse label values: {sorted(low[bad].unique())[:5]}")
    return low.isin(_TRUE)


def hash_bucket(cid: str, seed: int = 0) -> float:
    """Stable value in [0,1) for a conversation id. Same id -> same bucket, forever."""
    h = hashlib.md5(f"{seed}:{cid}".encode()).hexdigest()
    return (int(h, 16) % 10_000) / 10_000


def assign_hash_split(cids: pd.Series, fractions: dict, seed: int) -> pd.Series:
    cut_train = fractions["train"]
    cut_val = cut_train + fractions["val"]
    uniq = {c: hash_bucket(c, seed) for c in cids.unique()}
    b = cids.map(uniq)
    return pd.Series(np.where(b < cut_train, "train", np.where(b < cut_val, "val", "test")), index=cids.index)


def _read_raw(cfg: dict) -> pd.DataFrame:
    d = cfg["data"]
    files = sorted(glob.glob(str(resolve(cfg, d["raw_logs_glob"]))))
    if not files:
        raise FileNotFoundError(f"No files match {d['raw_logs_glob']}")
    frames = [pd.read_json(f, lines=True) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df[SPLIT] = assign_hash_split(df[d["id_col"]].astype(str), d["hash_split_fractions"], d["hash_seed"])
    return df


def _read_splits(cfg: dict) -> pd.DataFrame:
    d = cfg["data"]
    sdir = resolve(cfg, d["splits_dir"])
    frames = []
    for name in ("train", "val", "test"):
        p = sdir / f"{name}.csv"
        if p.exists():
            frames.append(pd.read_csv(p).assign(**{SPLIT: name}))
    if not frames:
        raise FileNotFoundError(f"No train/val/test csv in {sdir}")
    return pd.concat(frames, ignore_index=True)


def load_table(cfg: dict) -> pd.DataFrame:
    d = cfg["data"]
    raw = _read_raw(cfg) if d["source"] == "raw_logs" else _read_splits(cfg)

    feats = list(d["features"]) + list(d.get("extra_features", []))
    need = [d["id_col"], d["label_col"]] + [f for f in feats if f != d["turn_col"]]
    if d["turn_col"] in feats or d["turn_col"]:
        need.append(d["turn_col"])
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Found {raw.columns.tolist()}. Fix data.* in the config.")

    df = raw.copy()
    df[CID] = df[d["id_col"]].astype(str)
    df[TURN] = df[d["turn_col"]].astype(int)
    ok = _to_bool(df[d["label_col"]])
    df[Y] = (~ok if d["label_is_success"] else ok).astype(int)
    df[GROUP] = df[d["group_col"]].astype(str) if d.get("group_col") in df.columns else "all"
    for f in feats:
        df[f] = pd.to_numeric(df[f], errors="coerce")

    if d.get("duplicate_policy", "keep") == "mean":
        agg = {f: "mean" for f in feats if f != TURN}
        agg.update({Y: "first", SPLIT: "first", GROUP: "first"})
        df = df.groupby([CID, TURN], as_index=False).agg(agg)
    cols = [CID, TURN] + [f for f in feats if f != TURN] + [Y, SPLIT, GROUP]
    return df[cols].sort_values([CID, TURN]).reset_index(drop=True)


def validate(df: pd.DataFrame, cfg: dict) -> dict:
    """Hard errors for things that would silently corrupt results; warnings for the rest."""
    d = cfg["data"]
    feats = list(d["features"]) + list(d.get("extra_features", []))
    feats = [f for f in feats if f in df.columns]
    rep: dict = {"warnings": []}

    per_conv = df.groupby(CID)[Y].nunique()
    if (per_conv > 1).any():
        raise ValueError(f"Label differs within a conversation: {per_conv[per_conv > 1].index[:5].tolist()}")
    leak = df.groupby(CID)[SPLIT].nunique()
    if (leak > 1).any():
        raise ValueError(f"Conversation appears in several splits (leakage): {leak[leak > 1].index[:5].tolist()}")

    conv = df.groupby(CID).agg(y=(Y, "first"), split=(SPLIT, "first"), group=(GROUP, "first"), turns=(TURN, "nunique"))
    rep["n_rows"] = int(len(df))
    rep["n_conversations"] = int(len(conv))
    rep["conversation_failure_rate"] = float(conv.y.mean())
    rep["per_split"] = {
        s: {"conversations": int(len(g)), "rows": int((df[SPLIT] == s).sum()),
            "failed": int(g.y.sum()), "succeeded": int((1 - g.y).sum())}
        for s, g in conv.groupby("split")
    }
    rep["per_group"] = {
        s: {"conversations": int(len(g)), "failure_rate": float(g.y.mean())} for s, g in conv.groupby("group")
    }
    dup = df.groupby([CID, TURN]).size()
    rep["duplicate_turn_rows"] = int((dup > 1).sum())
    rep["nan_features"] = {f: int(df[f].isna().sum()) for f in feats if df[f].isna().any()}
    rep["constant_features"] = [f for f in feats if df[f].nunique(dropna=True) <= 1]

    w = rep["warnings"]
    if rep["constant_features"]:
        w.append(f"Constant (zero-information) features: {rep['constant_features']} - ask Person A why.")
    if rep["duplicate_turn_rows"]:
        w.append(f"{rep['duplicate_turn_rows']} (conversation, turn) pairs have several rows (repeated runs/resamples?).")
    for s, v in rep["per_split"].items():
        if min(v["failed"], v["succeeded"]) < 5:
            w.append(f"Split '{s}' has only {min(v['failed'], v['succeeded'])} conversation(s) of its rarer class; "
                     "metrics on it are very noisy.")
    for g, v in rep["per_group"].items():
        if v["failure_rate"] in (0.0, 1.0) and v["conversations"] >= 3:
            w.append(f"Group '{g}': every one of its {v['conversations']} conversations "
                     f"{'failed' if v['failure_rate'] == 1 else 'succeeded'} (group is a confound).")
    return rep


def select_features(df: pd.DataFrame, cfg: dict) -> list[str]:
    d = cfg["data"]
    feats = [f for f in list(d["features"]) + list(d.get("extra_features", [])) if f in df.columns]
    if d.get("drop_constant_features", True):
        feats = [f for f in feats if df[f].nunique(dropna=True) > 1]
    return feats


def fingerprint(df: pd.DataFrame) -> str:
    ids = "|".join(sorted(df[CID].unique()))
    return hashlib.sha1(f"{len(df)}#{ids}".encode()).hexdigest()[:12]


def prepare(cfg: dict, verbose: bool = True):
    """Load + validate. Returns (df, used_feature_names, report)."""
    df = load_table(cfg)
    rep = validate(df, cfg)
    feats = select_features(df, cfg)
    rep["features_used"] = feats
    rep["fingerprint"] = fingerprint(df)
    if verbose:
        print(f"[data] {rep['n_conversations']} conversations / {rep['n_rows']} rows "
              f"(failure rate {rep['conversation_failure_rate']:.0%}) fingerprint={rep['fingerprint']}")
        for s, v in rep["per_split"].items():
            print(f"[data]   {s:5s} {v['conversations']:4d} convs  failed={v['failed']} succeeded={v['succeeded']}")
        print(f"[data] features used: {feats}")
        for m in rep["warnings"]:
            print(f"[data] WARNING: {m}")
    return df, feats, rep


if __name__ == "__main__":
    _cfg = load_config()
    _df, _f, _rep = prepare(_cfg)
    print(json.dumps({k: v for k, v in _rep.items() if k != "warnings"}, indent=2, default=str))
