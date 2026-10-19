"""Ablations and honest baselines for the paper (plan Sec. 9.3).

    python -m src.classifier.evaluate                 # uses the latest trained run's primary model

Everything is out-of-fold (repeated grouped CV) and reported both POOLED and WITHIN-GROUP, because a task source
in which every conversation fails lets a model score well pooled just by recognising the source.

Outputs (reports/classifier/<run_id>/ablations/):
    baselines.csv        naive predictors the model must beat (constant / turn-index only / source only)
    signal_ablation.csv  each signal alone, and the full model with one signal removed
    learning_curve.csv   AUROC vs. number of training conversations (is more data worth it?)
    auroc_by_turn.csv    does the score separate outcomes EARLY in the conversation?
"""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd

from . import plots
from .config import load_config, resolve
from .cv import cross_val_oof
from .data import CID, GROUP, TURN, Y, prepare
from .metrics import auroc, auroc_by_turn, bootstrap_ci, within_group_auroc

warnings.filterwarnings("ignore")


def _score(df, feats, name, cfg, params=None, **kw) -> dict:
    oof = cross_val_oof(df, feats, name, cfg, params=params, **kw)
    p = np.nanmean(oof, axis=0)
    y, g = df[Y].to_numpy(), df[CID].to_numpy()
    wg = within_group_auroc(y, p, g, df[GROUP].to_numpy())
    lo, hi = bootstrap_ci(y, p, g, n=300, seed=cfg["seed"])
    return {"auroc_pooled": auroc(y, p), "pooled_ci_low": lo, "pooled_ci_high": hi,
            "auroc_within_group": wg["macro"], "p": p}


def run_evaluation(cfg: dict, model: str | None = None, verbose: bool = True) -> dict:
    mdir = resolve(cfg, cfg["paths"]["models_dir"])
    info = json.loads((mdir / "latest.json").read_text(encoding="utf-8"))
    model = model or info["primary"]
    params = json.loads(json.dumps(
        __import__("joblib").load(mdir / info["run_id"] / f"{model}.joblib")["meta"]["params"], default=list))
    df, feats, rep = prepare(cfg, verbose=False)
    out = resolve(cfg, cfg["paths"]["reports_dir"]) / info["run_id"] / "ablations"
    out.mkdir(parents=True, exist_ok=True)
    say = print if verbose else (lambda *a, **k: None)
    say(f"[eval] model={model} run={info['run_id']} features={feats}")

    # 1. naive baselines ------------------------------------------------------------------------------
    full = _score(df, feats, model, cfg, params)
    rows = [{"setting": f"{model}: all signals", **{k: v for k, v in full.items() if k != "p"}},
            {"setting": "constant predictor", "auroc_pooled": 0.5, "auroc_within_group": 0.5}]
    t = _score(df, [TURN], "logreg", cfg)
    rows.append({"setting": "turn_index only (logreg)", **{k: v for k, v in t.items() if k != "p"}})
    d2 = df.copy()
    onehot = []
    for gname in sorted(d2[GROUP].unique()):
        d2[f"src_{gname}"] = (d2[GROUP] == gname).astype(float)
        onehot.append(f"src_{gname}")
    s = _score(d2, onehot, "logreg", cfg)
    # Inside one source this predictor is constant, so its within-group AUROC is 0.5 by construction.
    rows.append({"setting": "task source only (confound probe)",
                 **{k: v for k, v in s.items() if k not in ("p", "auroc_within_group")}, "auroc_within_group": 0.5})
    base = pd.DataFrame(rows)
    base.to_csv(out / "baselines.csv", index=False)

    # 2. signal ablation ------------------------------------------------------------------------------
    arows = []
    for gname, cols in cfg["signal_groups"].items():
        cols = [c for c in cols if c in feats]
        if not cols:
            continue
        only = _score(df, cols, model, cfg, params)
        rest = [f for f in feats if f not in cols]
        drop = _score(df, rest, model, cfg, params) if rest else {"auroc_pooled": np.nan, "auroc_within_group": np.nan}
        arows.append({"signal": gname, "features": "+".join(cols),
                      "only_pooled": only["auroc_pooled"], "only_within_group": only["auroc_within_group"],
                      "without_pooled": drop["auroc_pooled"], "without_within_group": drop["auroc_within_group"],
                      "drop_in_within_group_when_removed": full["auroc_within_group"] - drop["auroc_within_group"]})
    abl = pd.DataFrame(arows)
    abl.to_csv(out / "signal_ablation.csv", index=False)
    plots.bars(["all"] + abl["signal"].tolist(),
               [full["auroc_within_group"]] + abl["only_within_group"].tolist(), out / "signal_ablation.png",
               "Within-group AUROC: all signals vs each signal alone")

    # 3. learning curve -------------------------------------------------------------------------------
    lrows = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        r = _score(df, feats, model, cfg, params, train_fraction=frac)
        lrows.append({"train_fraction": frac, "auroc_pooled": r["auroc_pooled"],
                      "auroc_within_group": r["auroc_within_group"]})
    lc = pd.DataFrame(lrows)
    lc.to_csv(out / "learning_curve.csv", index=False)
    plots.line(lc.train_fraction, {"pooled": lc.auroc_pooled, "within-group": lc.auroc_within_group},
               out / "learning_curve.png", "AUROC vs training data used", "fraction of training conversations")

    # 4. leading-indicator check ----------------------------------------------------------------------
    bt = auroc_by_turn(df[TURN], df[Y], full["p"])
    bt.to_csv(out / "auroc_by_turn.csv", index=False)

    if verbose:
        pd.set_option("display.width", 200)
        print("\n== Baselines (out-of-fold) ==");  print(base.round(3).to_string(index=False))
        print("\n== Signal ablation ==");           print(abl.round(3).to_string(index=False))
        print("\n== Learning curve ==");            print(lc.round(3).to_string(index=False))
        print("\n== AUROC by turn ==");             print(bt.round(3).to_string(index=False))
        print(f"\n[eval] wrote {out}")
    return {"baselines": base, "ablation": abl, "learning_curve": lc, "by_turn": bt}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/classifier.yaml")
    ap.add_argument("--model", choices=["logreg", "xgboost", "mlp"])
    a = ap.parse_args()
    run_evaluation(load_config(a.config), a.model)


if __name__ == "__main__":
    main()
