"""Load, clean, validate and standardise the turn-level table written by Person A.

Canonical columns after loading (independent of the file's own column names):
    conversation_id  = one RUN (task + model + run index). The unit of a "conversation" for labels/metrics.
    task_id          = the task itself. Used to GROUP cross-validation and bootstrap, so repeated attempts of one
                       task (other models, other runs) never sit on both sides of a split.
    model, turn_index, <features...>, y, split, group, label_reason, n_resamples
where y = 1 means the run FAILED (that is what drift_score predicts).

Run `python -m src.classifier.data` for a data-quality report.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re

import numpy as np
import pandas as pd

from .config import load_config, resolve

CID, TASK, MODEL, TURN, Y, SPLIT, GROUP = ("conversation_id", "task_id", "model", "turn_index", "y", "split",
                                           "group")
REASON, NRES = "label_reason", "n_resamples"
_TRUE = {"true", "1", "yes", "pass", "passed", "success", "t"}
_FALSE = {"false", "0", "no", "fail", "failed", "failure", "f"}


# ------------------------------------------------------------------------------------------------- helpers
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


def hash_bucket(key: str, seed: int = 0) -> float:
    """Stable value in [0,1) for an id. Same id -> same bucket, forever (so splits survive new data)."""
    h = hashlib.md5(f"{seed}:{key}".encode()).hexdigest()
    return (int(h, 16) % 10_000) / 10_000


def assign_hash_split(keys: pd.Series, fractions: dict, seed: int) -> pd.Series:
    cut_train = fractions["train"]
    cut_val = cut_train + fractions["val"]
    uniq = {c: hash_bucket(c, seed) for c in keys.unique()}
    b = keys.map(uniq)
    return pd.Series(np.where(b < cut_train, "train", np.where(b < cut_val, "val", "test")), index=keys.index)


def _read_raw(cfg: dict) -> pd.DataFrame:
    d = cfg["data"]
    files = sorted(glob.glob(str(resolve(cfg, d["raw_logs_glob"]))))
    if not files:
        raise FileNotFoundError(f"No files match {d['raw_logs_glob']}")
    frames = []
    for i, f in enumerate(files):          # later file name = higher priority when rows collide
        x = pd.read_json(f, lines=True)
        x["_file"], x["_file_order"] = os.path.basename(f), i
        frames.append(x)
    df = pd.concat(frames, ignore_index=True)
    # split by TASK so every run of a task lands in the same split
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
    df = pd.concat(frames, ignore_index=True)
    df["_file"], df["_file_order"] = "splits", 0
    return df


def coalesce_aliases(raw: pd.DataFrame, cfg: dict):
    """Rescue values stored under a mistyped column name (e.g. 'self_consi  stency_variance').

    Explicit aliases come from data.column_aliases; any column that equals a known column once whitespace is
    removed is aliased automatically. Values fill only the cells where the real column is missing.
    """
    d = cfg["data"]
    known = set(d["features"]) | set(d.get("extra_features", [])) | {d["label_col"], d["turn_col"], d["id_col"]}
    alias = dict(d.get("column_aliases") or {})
    for c in raw.columns:
        n = re.sub(r"\s+", "", str(c))
        if c not in known and n in known and c not in alias:
            alias[c] = n
    log = {}
    for a, t in alias.items():
        if a not in raw.columns or a == t:
            continue
        if t in raw.columns:
            both = raw[t].notna() & raw[a].notna()
            conflict = int((both & ~np.isclose(pd.to_numeric(raw[t], errors="coerce"),
                                               pd.to_numeric(raw[a], errors="coerce"), equal_nan=True)).sum())
            rescued = int((raw[t].isna() & raw[a].notna()).sum())
            raw[t] = raw[t].fillna(raw[a])
        else:
            raw[t], conflict, rescued = raw[a], 0, int(raw[a].notna().sum())
        log[a] = {"into": t, "cells_rescued": rescued, "conflicting_values": conflict}
        raw = raw.drop(columns=[a])
    return raw, log


# ------------------------------------------------------------------------------------------------- loading
def load_table(cfg: dict, return_log: bool = False):
    d = cfg["data"]
    raw = _read_raw(cfg) if d["source"] == "raw_logs" else _read_splits(cfg)
    n_in = len(raw)
    raw, alias_log = coalesce_aliases(raw, cfg)

    feats = [f for f in list(d["features"]) + list(d.get("extra_features", [])) if f != d["turn_col"]]
    need = [d["id_col"], d["label_col"], d["turn_col"]] + feats
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Found {raw.columns.tolist()}. Fix data.* in the config.")

    df = raw.copy()
    df[TASK] = df[d["id_col"]].astype(str)
    mcol, rcol = d.get("model_col"), d.get("run_col")
    has_model = df[mcol].notna() if mcol in df.columns else pd.Series(False, index=df.index)
    df[MODEL] = df[mcol].where(has_model, "legacy").astype(str) if mcol in df.columns else "legacy"
    run_idx = pd.to_numeric(df[rcol], errors="coerce").fillna(0).astype(int) if rcol in df.columns else 0
    df[CID] = df[TASK] + "|" + df[MODEL] + "|" + pd.Series(run_idx, index=df.index).astype(str)
    df[TURN] = df[d["turn_col"]].astype(int)
    ok = _to_bool(df[d["label_col"]])
    df[Y] = (~ok if d["label_is_success"] else ok).astype(int)
    df[GROUP] = df[d["group_col"]].astype(str) if d.get("group_col") in df.columns else "all"
    for f in feats:
        df[f] = pd.to_numeric(df[f], errors="coerce")
    df[REASON] = df[REASON].astype(object) if REASON in df.columns else ""
    df[NRES] = pd.to_numeric(df[NRES], errors="coerce") if NRES in df.columns else np.nan
    df["_has_meta"] = has_model

    log = {"rows_in": int(n_in), "aliases": alias_log}

    # -- the same (run, turn) exported by several files: keep ONE copy (metadata beats recency, then newest file).
    #    Rows repeated inside a single file are left alone (they may be genuine repeats) and only reported.
    if d.get("dedupe_across_files", True) and df["_file_order"].nunique() > 1:
        df["_prio"] = df["_file_order"] + 1000 * df["_has_meta"].astype(int)
        best = df.groupby([CID, TURN])["_prio"].transform("max")
        sig = df[feats + [Y]].round(9).astype(str).agg("|".join, axis=1)
        kept_sigs = sig[df["_prio"] == best].groupby([df[CID], df[TURN]]).agg(set)
        drop = df["_prio"] < best
        conflicts = sum(s not in kept_sigs[(c, t)] for s, c, t in zip(sig[drop], df.loc[drop, CID], df.loc[drop, TURN]))
        log["dedupe_across_files"] = {"rows_dropped": int(drop.sum()), "dropped_rows_that_differed": int(conflicts)}
        df = df[~drop]

    # -- quality filters (all configurable). A flagged row removes its WHOLE run.
    f = d.get("filters") or {}
    flog = {}

    def drop_runs(mask: pd.Series, name: str):
        nonlocal df
        bad = set(df.loc[mask.fillna(False), CID])
        if bad:
            n_rows = int(df[CID].isin(bad).sum())
            df = df[~df[CID].isin(bad)]
            flog[name] = {"runs_dropped": len(bad), "rows_dropped": n_rows}

    num = lambda c: pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(np.nan, index=df.index)  # noqa: E731
    if f.get("require_label_valid", True):
        drop_runs(num("label_valid") == 0, "label_valid==0")
    if f.get("exclude_truncated", True):
        drop_runs((num("any_turn_truncated") == 1) | (num("truncated") == 1), "truncated")
    if f.get("exclude_label_reasons"):
        drop_runs(df[REASON].isin(f["exclude_label_reasons"]), "label_reason excluded")
    if f.get("extractor_versions") and "extractor_version" in df.columns:
        ev = df["extractor_version"]
        drop_runs(ev.notna() & ~ev.isin(f["extractor_versions"]), "extractor_version not allowed")
    if not f.get("keep_missing_metadata", True):
        drop_runs(~df["_has_meta"], "no metadata (legacy rows)")
    log["filters"] = flog

    if d.get("duplicate_policy", "keep") == "mean":
        agg = {c: "mean" for c in feats}
        agg.update({Y: "first", SPLIT: "first", GROUP: "first", TASK: "first", MODEL: "first", REASON: "first",
                    NRES: "first"})
        df = df.groupby([CID, TURN], as_index=False).agg(agg)

    cols = [CID, TASK, MODEL, TURN] + feats + [Y, SPLIT, GROUP, REASON, NRES]
    df = df[cols].sort_values([CID, TURN]).reset_index(drop=True)
    log["rows_out"] = int(len(df))
    return (df, log) if return_log else df


# ------------------------------------------------------------------------------------------------- checks
def validate(df: pd.DataFrame, cfg: dict) -> dict:
    """Hard errors for things that would silently corrupt results; warnings for the rest."""
    d = cfg["data"]
    df = df.copy()
    if TASK not in df:
        df[TASK] = df[CID]
    if MODEL not in df:
        df[MODEL] = "legacy"
    feats = [f for f in list(d["features"]) + list(d.get("extra_features", [])) if f in df.columns]
    rep: dict = {"warnings": []}
    w = rep["warnings"]

    per_run = df.groupby(CID)[Y].nunique()
    if (per_run > 1).any():
        raise ValueError(f"Label differs within a run: {per_run[per_run > 1].index[:5].tolist()}")
    leak = df.groupby(CID)[SPLIT].nunique()
    if (leak > 1).any():
        raise ValueError(f"Conversation appears in several splits (leakage): {leak[leak > 1].index[:5].tolist()}")
    task_split = df.groupby(TASK)[SPLIT].nunique()
    if (task_split > 1).any():
        w.append(f"{int((task_split > 1).sum())} task(s) have runs in several splits (task-level leakage in the "
                 "given split files). Cross-validation groups by task, so metrics are unaffected.")

    run = df.groupby(CID).agg(y=(Y, "first"), split=(SPLIT, "first"), group=(GROUP, "first"), model=(MODEL, "first"),
                              task=(TASK, "first"), reason=(REASON, "first"), turns=(TURN, "nunique"))
    rep["n_rows"], rep["n_conversations"] = int(len(df)), int(len(run))
    rep["n_tasks"] = int(run.task.nunique())
    rep["conversation_failure_rate"] = float(run.y.mean())
    rep["per_split"] = {s: {"conversations": int(len(g)), "rows": int((df[SPLIT] == s).sum()),
                            "failed": int(g.y.sum()), "succeeded": int((1 - g.y).sum())}
                        for s, g in run.groupby("split")}
    rep["per_group"] = {s: {"conversations": int(len(g)), "failure_rate": float(g.y.mean())}
                        for s, g in run.groupby("group")}
    rep["per_model"] = {s: {"conversations": int(len(g)), "failure_rate": float(g.y.mean())}
                        for s, g in run.groupby("model")}
    rep["source_by_model"] = {str(k): v for k, v in run.groupby(["group", "model"]).size().to_dict().items()}
    rep["label_reasons"] = run.reason.value_counts().head(12).to_dict()

    runs_per_task = run.groupby("task").size()
    both = run.groupby("task").y.nunique()
    rep["tasks_with_several_runs"] = int((runs_per_task > 1).sum())
    rep["tasks_with_both_outcomes"] = int((both > 1).sum())
    rep["duplicate_turn_rows"] = int((df.groupby([CID, TURN]).size() > 1).sum())
    rep["nan_features"] = {f: int(df[f].isna().sum()) for f in feats if df[f].isna().any()}
    rep["constant_features"] = [f for f in feats if df[f].nunique(dropna=True) <= 1]
    if NRES in df and df[NRES].notna().any():
        rep["n_resamples"] = {str(int(k)): int(v) for k, v in df[NRES].dropna().value_counts().items()}

    if rep["constant_features"]:
        w.append(f"Constant (zero-information) features: {rep['constant_features']} - ask Person A why.")
    if rep["duplicate_turn_rows"]:
        w.append(f"{rep['duplicate_turn_rows']} (run, turn) pairs still have several rows inside one file.")
    for s, v in rep["per_split"].items():
        if min(v["failed"], v["succeeded"]) < 5:
            w.append(f"Split '{s}' has only {min(v['failed'], v['succeeded'])} run(s) of its rarer outcome; "
                     "metrics on it are very noisy.")
    for label, table in (("Task source", rep["per_group"]), ("Model", rep["per_model"])):
        for g, v in table.items():
            if v["failure_rate"] in (0.0, 1.0) and v["conversations"] >= 5:
                w.append(f"{label} '{g}': all {v['conversations']} runs "
                         f"{'failed' if v['failure_rate'] == 1 else 'succeeded'} (confound: a model can score well "
                         "just by recognising it).")
    if rep["tasks_with_both_outcomes"] < 20:
        w.append(f"Only {rep['tasks_with_both_outcomes']} task(s) have runs with both outcomes; the cleanest "
                 "within-task test has very little to work with.")
    if "n_resamples" in rep and max(map(int, rep["n_resamples"])) < 3:
        w.append("self_consistency_variance is built from <=2 resamples (one pairwise distance): very noisy.")
    return rep


def select_features(df: pd.DataFrame, cfg: dict) -> list[str]:
    d = cfg["data"]
    feats = [f for f in list(d["features"]) + list(d.get("extra_features", [])) if f in df.columns]
    if d.get("drop_constant_features", True):
        feats = [f for f in feats if df[f].nunique(dropna=True) > 1]
    return feats


TRAJ_STATS = ("cummax", "cummean", "dfirst")


def add_trajectory_features(df: pd.DataFrame, feats: list[str], stats=TRAJ_STATS):
    """Causal per-run summaries of each signal: running max, running mean, and change since the run's first turn.

    Drift is a trajectory, and a single turn's value is noisy (self-consistency here rests on 2 resamples). Only
    PAST and CURRENT turns are used, so the same numbers can be built live (see predict.TrajectoryTracker).
    Needs rows ordered by turn within each run (load_table guarantees this).
    Returns (df_with_new_columns, feature_names_including_new_ones).
    """
    df = df.copy()
    new = []
    for f in feats:
        g = df.groupby(CID, sort=False)[f]
        if "cummax" in stats:
            df[f"{f}__cummax"] = g.cummax()
            new.append(f"{f}__cummax")
        if "cummean" in stats:
            df[f"{f}__cummean"] = g.transform(lambda s: s.expanding().mean())
            new.append(f"{f}__cummean")
        if "dfirst" in stats:
            df[f"{f}__dfirst"] = df[f] - g.transform("first")
            new.append(f"{f}__dfirst")
    return df, list(feats) + new


def fingerprint(df: pd.DataFrame) -> str:
    ids = "|".join(sorted(df[CID].unique()))
    return hashlib.sha1(f"{len(df)}#{ids}".encode()).hexdigest()[:12]


def prepare(cfg: dict, verbose: bool = True):
    """Load + clean + validate. Returns (df, used_feature_names, report)."""
    df, log = load_table(cfg, return_log=True)
    rep = validate(df, cfg)
    feats = select_features(df, cfg)
    dd = cfg["data"]
    if dd.get("trajectory_features"):
        df, feats = add_trajectory_features(df, feats, dd.get("trajectory_stats") or TRAJ_STATS)
    rep.update({"features_used": feats, "fingerprint": fingerprint(df), "load_log": log})
    if verbose:
        print(f"[data] {log['rows_in']} raw rows -> {rep['n_rows']} rows | {rep['n_conversations']} runs of "
              f"{rep['n_tasks']} tasks (failure rate {rep['conversation_failure_rate']:.0%}) "
              f"fingerprint={rep['fingerprint']}")
        for a, v in log["aliases"].items():
            print(f"[data] column '{a}' -> '{v['into']}': rescued {v['cells_rescued']} cells "
                  f"({v['conflicting_values']} conflicting)")
        if "dedupe_across_files" in log:
            v = log["dedupe_across_files"]
            print(f"[data] cross-file duplicates removed: {v['rows_dropped']} rows "
                  f"({v['dropped_rows_that_differed']} differed from the kept copy)")
        for k, v in log["filters"].items():
            print(f"[data] filter {k}: dropped {v['runs_dropped']} runs / {v['rows_dropped']} rows")
        for s, v in rep["per_split"].items():
            print(f"[data]   {s:5s} {v['conversations']:4d} runs  failed={v['failed']} succeeded={v['succeeded']}")
        print(f"[data] models: { {k: v['conversations'] for k, v in rep['per_model'].items()} }")
        print(f"[data] tasks with several runs: {rep['tasks_with_several_runs']} | "
              f"with both outcomes: {rep['tasks_with_both_outcomes']}")
        print(f"[data] features used: {feats}")
        for m in rep["warnings"]:
            print(f"[data] WARNING: {m}")
    return df, feats, rep


if __name__ == "__main__":
    _cfg = load_config()
    _df, _f, _rep = prepare(_cfg)
    print(json.dumps({k: v for k, v in _rep.items() if k not in ("warnings", "source_by_model")}, indent=2,
                     default=str))
