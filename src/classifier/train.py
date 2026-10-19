"""Train + compare the three classifiers, calibrate, choose the threshold, save a versioned bundle.

    python -m src.classifier.train                       # uses configs/classifier.yaml
    python -m src.classifier.train --models logreg mlp   # subset
    python -m src.classifier.train --mode cv --tune false
    python -m src.classifier.train --source raw_logs     # re-split from jsonl (after new data lands)

Re-run the same command whenever Person A delivers more conversations: nothing else changes.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from . import plots
from .calibration import Calibrator, cross_fit_calibrate, select_threshold, sweep_thresholds
from .config import load_config, resolve
from .cv import cross_val_oof, official_split_eval
from .data import CID, GROUP, TURN, Y, prepare
from .metrics import auroc, auroc_by_turn, bootstrap_ci, prauc, summarize, within_group_auroc
from .models import ORDER, fit_model, should_tune

warnings.filterwarnings("ignore")


def resolve_mode(cfg: dict, rep: dict) -> str:
    m = cfg["evaluation"]["mode"]
    if m != "auto":
        return m
    e = cfg["evaluation"]
    ps = rep["per_split"]
    for s in ("val", "test"):
        if s not in ps or ps[s]["conversations"] < e["min_conversations_per_split"] \
                or min(ps[s]["failed"], ps[s]["succeeded"]) < e["min_minority_per_split"]:
            return "cv"
    return "splits"


def _finish(name, est, used, calib, p_eval, p_rank, y_eval, meta_eval, g_eval, sel_auc, cfg, extra_metrics):
    """Shared tail: metrics, sweep, threshold for one model.

    p_eval = calibrated probabilities (used for Brier / ECE / threshold sweep);
    p_rank = raw scores (used for AUROC / PR-AUC / CI, which only depend on ranking).
    """
    method_step = cfg["threshold"]["grid_step"]
    m = summarize(y_eval, p_eval)
    m["auroc"], m["pr_auc"] = auroc(y_eval, p_rank), prauc(y_eval, p_rank)
    lo, hi = bootstrap_ci(y_eval, p_rank, g_eval, n=cfg["evaluation"]["bootstrap_samples"], seed=cfg["seed"])
    wg = within_group_auroc(y_eval, p_rank, meta_eval[CID].to_numpy(), meta_eval[GROUP].to_numpy())
    m.update({"auroc_ci_low": lo, "auroc_ci_high": hi, "auroc_within_group": wg["macro"],
              "within_group_detail": wg["per_group"], **extra_metrics})
    metric = cfg["selection"].get("metric", "auto")
    use_wg = (metric == "within_group") or (metric == "auto" and not np.isnan(wg["macro"]) and sel_auc is None)
    m["select_auroc"] = wg["macro"] if use_wg else (auroc(y_eval, p_rank) if sel_auc is None else sel_auc)
    sweep = sweep_thresholds(meta_eval, y_eval, p_eval, step=method_step)
    thr, row = select_threshold(sweep, cfg)
    return {"name": name, "est": est, "params": used, "calibrator": calib, "threshold": thr, "thr_row": row,
            "sweep": sweep, "metrics": m, "eval_y": np.asarray(y_eval), "eval_p": np.asarray(p_eval),
            "eval_meta": meta_eval}


def run_model_cv(name, df, feats, cfg, tune):
    y, g = df[Y].to_numpy(), df[CID].to_numpy()
    oof = cross_val_oof(df, feats, name, cfg, tune=tune)
    oof_mean = np.nanmean(oof, axis=0)
    per_rep = [auroc(y, oof[r]) for r in range(oof.shape[0])]
    method = cfg["calibration"]["method"]
    calib = Calibrator(method).fit(oof_mean, y)
    p_cal = cross_fit_calibrate(oof_mean, y, g, method, seed=cfg["seed"])
    t0 = time.perf_counter()
    est, used = fit_model(name, df[feats], y, g, cfg, tune=tune, seed=cfg["seed"])
    train_s = time.perf_counter() - t0
    res = _finish(name, est, used, calib, p_cal, oof_mean, y, df[[CID, TURN, GROUP]], g, None, cfg,
                  {"auroc_repeat_mean": float(np.mean(per_rep)), "auroc_repeat_sd": float(np.std(per_rep)),
                   "train_seconds": train_s, "eval_protocol": "repeated grouped CV (out-of-fold)"})
    res["eval_df"] = df
    return res


def run_model_splits(name, df, feats, cfg, tune):
    est, out = official_split_eval(df, feats, name, cfg, tune=tune)
    vdf, pv = out["val"]
    tdf, pt = out["test"]
    method = cfg["calibration"]["method"]
    calib = Calibrator(method).fit(pv, vdf[Y].to_numpy())
    pt_cal = calib.transform(pt)
    t0 = time.perf_counter()
    fit_model(name, df[df.split == "train"][feats], df[df.split == "train"][Y].to_numpy(),
              df[df.split == "train"][CID].to_numpy(), cfg, tune=tune, params=out["params"], seed=cfg["seed"])
    train_s = time.perf_counter() - t0
    # threshold must come from validation (never from test)
    pv_cal = calib.transform(pv)
    sweep_v = sweep_thresholds(vdf[[CID, TURN]], vdf[Y].to_numpy(), pv_cal, step=cfg["threshold"]["grid_step"])
    thr, row = select_threshold(sweep_v, cfg)
    res = _finish(name, est, out["params"], calib, pt_cal, pt, tdf[Y].to_numpy(), tdf[[CID, TURN, GROUP]],
                  tdf[CID].to_numpy(), auroc(vdf[Y].to_numpy(), pv), cfg,
                  {"train_seconds": train_s, "eval_protocol": "train -> calibrate/threshold on val -> report test"})
    res.update({"threshold": thr, "thr_row": row, "sweep": sweep_v})
    res["eval_df"] = tdf
    return res


def pick_primary(results: dict, cfg: dict) -> tuple[str, str]:
    want = cfg["selection"]["primary_model"]
    if want != "auto":
        return want, f"set in config (primary_model: {want})"
    best = max(r["metrics"]["select_auroc"] for r in results.values())
    tol = cfg["selection"]["tolerance"]
    near = [n for n in ORDER if n in results and results[n]["metrics"]["select_auroc"] >= best - tol]
    pick = near[0]
    why = (f"{pick} is within {tol} AUROC of the best ({best:.3f}) and is the simplest/most interpretable "
           f"such model" if len(near) > 1 or pick != max(results, key=lambda n: results[n]['metrics']['select_auroc'])
           else f"{pick} has the highest selection AUROC ({best:.3f})")
    return pick, why


def make_bundle(name, res, df, feats, cfg, run_id, mode, rep):
    healthy = df[df[Y] == 0]
    ref_src = healthy if len(healthy) else df
    healthy_ref = {f: float(ref_src[f].median()) for f in feats}
    d = cfg["data"]
    return {
        "model": res["est"], "calibrator": res["calibrator"], "threshold": float(res["threshold"]),
        "features": feats, "input_order": list(d["features"]) + list(d.get("extra_features", [])),
        "signal_groups": cfg["signal_groups"], "healthy_ref": healthy_ref,
        "meta": {"run_id": run_id, "model": name, "params": res["params"], "mode": mode,
                 "fingerprint": rep["fingerprint"], "n_conversations": rep["n_conversations"],
                 "threshold_strategy": cfg["threshold"]["strategy"], "metrics": res["metrics"]},
    }


def time_inference(bundle, n=200) -> float:
    from .predict import _frame, _score
    row = {f: 0.5 for f in bundle["features"]}
    t0 = time.perf_counter()
    for _ in range(n):
        _score(bundle, _frame(row, bundle))
    return (time.perf_counter() - t0) / n * 1000


def run_training(cfg: dict, models=None, mode=None, tune=None, verbose=True) -> dict:
    df, feats, rep = prepare(cfg, verbose=verbose)
    if not feats:
        raise ValueError("No usable features after dropping constants.")
    mode = mode or resolve_mode(cfg, rep)
    models = models or cfg["models"]
    n_conv = rep["n_conversations"]
    do_tune = should_tune(cfg, n_conv) if tune is None else tune
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    if verbose:
        print(f"[train] run={run_id} mode={mode} tune={do_tune} models={models}")

    runner = run_model_cv if mode == "cv" else run_model_splits
    results = {}
    for name in models:
        results[name] = runner(name, df, feats, cfg, do_tune)
        r = results[name]["metrics"]
        if verbose:
            print(f"[train]   {name:8s} AUROC {r['auroc']:.3f} [{r['auroc_ci_low']:.3f}, {r['auroc_ci_high']:.3f}] "
                  f"within-group {r['auroc_within_group']:.3f} "
                  f"Brier {r['brier']:.3f} ECE {r['ece']:.3f} thr={results[name]['threshold']:.2f}")

    primary, why = pick_primary(results, cfg)

    # ---- artifacts -------------------------------------------------------------------------------
    mdir = resolve(cfg, cfg["paths"]["models_dir"])
    rdir = resolve(cfg, cfg["paths"]["reports_dir"])
    run_m, run_r = mdir / run_id, rdir / run_id
    run_m.mkdir(parents=True, exist_ok=True)
    run_r.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, res in results.items():
        bundle = make_bundle(name, res, df, feats, cfg, run_id, mode, rep)
        res["metrics"]["predict_ms"] = time_inference(bundle)
        joblib.dump(bundle, run_m / f"{name}.joblib")
        if name == primary:
            shutil.copy(run_m / f"{name}.joblib", run_m / "primary.joblib")
        row = res["thr_row"]
        m = res["metrics"]
        rows.append({"model": name, "auroc": m["auroc"], "auroc_ci_low": m["auroc_ci_low"],
                     "auroc_ci_high": m["auroc_ci_high"], "pr_auc": m["pr_auc"], "brier": m["brier"],
                     "ece": m["ece"], "base_rate": m["base_rate"], "threshold": res["threshold"],
                     "auroc_within_group": m["auroc_within_group"], "turn_recall@thr": row["recall"], "turn_precision@thr": row["precision"],
                     "flag_rate@thr": row["flag_rate"], "conv_recall@thr": row["conv_recall"],
                     "conv_false_alarm@thr": row["conv_false_alarm"], "train_s": m["train_seconds"],
                     "predict_ms": m["predict_ms"], "primary": name == primary,
                     "params": json.dumps(res["params"], default=list)})
        res["sweep"].to_csv(run_r / f"threshold_sweep_{name}.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(run_r / "model_comparison.csv", index=False)

    pres = results[primary]
    ev = pres["eval_meta"].assign(y=pres["eval_y"], p=pres["eval_p"])
    auroc_by_turn(ev[TURN], ev["y"], ev["p"]).to_csv(run_r / "auroc_by_turn.csv", index=False)

    if "logreg" in results:  # interpretability: standardised coefficients = relative signal strength
        clf = results["logreg"]["est"].named_steps["clf"]
        pd.DataFrame({"feature": feats, "std_coef_toward_failure": clf.coef_[0][:len(feats)]}) \
            .sort_values("std_coef_toward_failure", key=abs, ascending=False) \
            .to_csv(run_r / "logreg_coefficients.csv", index=False)

    split_rows = []
    if mode == "cv":  # also report the plan's original protocol, flagged as noisy
        for name in models:
            _, out = official_split_eval(df, feats, name, cfg, tune=do_tune)
            for s in ("val", "test"):
                if s in out:
                    part, p = out[s]
                    split_rows.append({"model": name, "split": s, "conversations": part[CID].nunique(),
                                       "auroc": auroc(part[Y], p)})
        pd.DataFrame(split_rows).to_csv(run_r / "official_split_check.csv", index=False)

    plots.reliability(results, run_r / "reliability.png")
    plots.roc(results, run_r / "roc.png")
    plots.threshold_sweep(pres["sweep"], pres["threshold"], run_r / "threshold_sweep.png",
                          f"Threshold sweep ({primary})")
    (run_r / "data_report.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")

    info = {"run_id": run_id, "primary": primary, "primary_reason": why, "threshold": pres["threshold"],
            "mode": mode, "fingerprint": rep["fingerprint"], "n_conversations": n_conv, "n_rows": rep["n_rows"],
            "features": feats, "created": datetime.now().isoformat(timespec="seconds")}
    (run_m / "metadata.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (mdir / "latest.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    hist = rdir / "runs_history.csv"  # one line per retrain: watch AUROC move as data grows
    h = pd.DataFrame([{**{k: info[k] for k in ("run_id", "created", "mode", "n_conversations", "n_rows",
                                                "fingerprint", "primary", "threshold")},
                       "auroc": pres["metrics"]["auroc"], "ece": pres["metrics"]["ece"]}])
    h.to_csv(hist, mode="a", header=not hist.exists(), index=False)

    if verbose:
        pd.set_option("display.width", 200)
        print("\n" + summary[["model", "auroc", "auroc_ci_low", "auroc_ci_high", "auroc_within_group", "pr_auc", "brier", "ece",
                              "threshold", "flag_rate@thr", "conv_recall@thr", "conv_false_alarm@thr",
                              "train_s", "predict_ms"]].round(3).to_string(index=False))
        print(f"\n[train] primary = {primary}: {why}")
        print(f"[train] threshold = {pres['threshold']:.2f} ({cfg['threshold']['strategy']}) | "
              f"models -> {run_m} | reports -> {run_r}")
    return {"info": info, "results": results, "summary": summary, "df": df, "features": feats, "report": rep}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/classifier.yaml")
    ap.add_argument("--models", nargs="*", choices=ORDER)
    ap.add_argument("--mode", choices=["cv", "splits"])
    ap.add_argument("--tune", choices=["true", "false"])
    ap.add_argument("--primary")
    ap.add_argument("--strategy", choices=["youden", "fbeta", "target_recall", "max_flag_rate"])
    ap.add_argument("--source", choices=["splits", "raw_logs"])
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.primary:
        cfg["selection"]["primary_model"] = a.primary
    if a.strategy:
        cfg["threshold"]["strategy"] = a.strategy
    if a.source:
        cfg["data"]["source"] = a.source
    run_training(cfg, models=a.models, mode=a.mode, tune=None if a.tune is None else a.tune == "true")


if __name__ == "__main__":
    main()
