"""Run from the repo root:  pytest tests/test_classifier.py -q

Pure-function tests use tiny hand-written arrays. Pipeline tests use the real files in data/ and write to a
temp dir, so they never touch your models/ or reports/ folders (they skip if the data is absent).
"""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.classifier.calibration import Calibrator, select_threshold, sweep_thresholds
from src.classifier.config import REPO_ROOT, load_config
from src.classifier.data import CID, SPLIT, Y, assign_hash_split, prepare, validate
from src.classifier.metrics import auroc, ece, within_group_auroc

HAVE_DATA = (REPO_ROOT / "data" / "splits" / "train.csv").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="data/splits not present")


def _fast_cfg(tmp_path):
    cfg = load_config()
    cfg = copy.deepcopy(cfg)
    cfg["paths"] = {"models_dir": str(tmp_path / "models"), "reports_dir": str(tmp_path / "reports")}
    cfg["evaluation"].update({"mode": "cv", "cv_repeats": 1, "cv_folds": 3, "bootstrap_samples": 20})
    cfg["models"] = ["logreg", "xgboost"]
    return cfg


# ---------------------------------------------------------------- pure functions
def test_hash_split_is_stable_when_data_is_appended():
    old = pd.Series([f"conv-{i}" for i in range(200)])
    new = pd.concat([old, pd.Series([f"new-{i}" for i in range(100)])], ignore_index=True)
    fr = {"train": 0.7, "val": 0.15, "test": 0.15}
    a = assign_hash_split(old, fr, 0)
    b = assign_hash_split(new, fr, 0)[: len(old)]
    assert (a.values == b.values).all()  # existing conversations never move between splits


def test_calibrator_is_bounded_and_monotone():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 300)
    y = (rng.uniform(size=300) < p).astype(int)
    for method in ("platt", "isotonic", "none"):
        c = Calibrator(method).fit(p, y)
        out = c.transform(np.linspace(0, 1, 50))
        assert out.min() >= 0 and out.max() <= 1
        assert np.all(np.diff(out) >= -1e-9)


def test_ece_zero_for_perfectly_calibrated_constant():
    y = np.array([1] * 8 + [0] * 2)
    assert ece(y, np.full(10, 0.8)) == pytest.approx(0.0, abs=1e-9)


def test_threshold_sweep_and_youden():
    meta = pd.DataFrame({"conversation_id": ["a"] * 3 + ["b"] * 3, "turn_index": [1, 2, 3] * 2})
    y = np.array([1, 1, 1, 0, 0, 0])
    p = np.array([0.6, 0.8, 0.9, 0.1, 0.2, 0.3])
    sw = sweep_thresholds(meta, y, p, step=0.05)
    thr, row = select_threshold(sw, {"threshold": {"strategy": "youden"}})
    assert 0.3 < thr <= 0.6 and row["conv_recall"] == 1.0 and row["conv_false_alarm"] == 0.0


def test_within_group_auroc_ignores_one_class_groups():
    y = np.array([1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1])
    cids = np.array(list("aabbccddeeffgghh"))
    groups = np.array(["all_fail"] * 4 + ["mixed"] * 12)
    p = np.linspace(0, 1, len(y))
    wg = within_group_auroc(y, p, cids, groups, min_conv_per_class=2)
    assert "all_fail" not in wg["per_group"]


# ---------------------------------------------------------------- data checks
def test_leakage_across_splits_is_rejected():
    cfg = load_config()
    df = pd.DataFrame({CID: ["a", "a", "b"], "turn_index": [1, 2, 1], Y: [1, 1, 0],
                       SPLIT: ["train", "test", "val"], "group": "g",
                       **{f: [0.1, 0.2, 0.3] for f in cfg["data"]["features"]}})
    with pytest.raises(ValueError, match="several splits"):
        validate(df, cfg)


@needs_data
def test_real_data_loads_and_label_means_failure():
    cfg = load_config()
    df, feats, rep = prepare(cfg, verbose=False)
    raw = pd.concat([pd.read_csv(REPO_ROOT / "data/splits" / f"{s}.csv") for s in ("train", "val", "test")])
    assert len(df) == len(raw)
    assert df[Y].sum() == (~raw["eventual_success"]).sum()  # y=1 <=> conversation FAILED
    assert set(df[SPLIT]) <= {"train", "val", "test"}
    assert all(df.groupby(CID)[SPLIT].nunique() == 1)


# ---------------------------------------------------------------- end-to-end contract
@needs_data
def test_train_then_predict_contract(tmp_path):
    from src.classifier import predict
    from src.classifier.train import run_training

    cfg = _fast_cfg(tmp_path)
    run_training(cfg, verbose=False)
    mdir = tmp_path / "models"
    assert json.loads((mdir / "latest.json").read_text())["primary"] in ("logreg", "xgboost")
    predict.clear_cache()

    feats = {"self_consistency_variance": 0.4, "contradiction_score": 0.9, "confidence": 0.5,
             "confidence_delta": -0.3, "silent_commitment_ratio": 0.0}
    for x in (feats, list(feats.values()), {"confidence": 0.5}):          # dict, agreed-order list, partial dict
        s = predict.predict_drift_score(x, models_dir=mdir)
        assert isinstance(s, float) and 0.0 <= s <= 1.0
    assert isinstance(predict.should_intervene(feats, models_dir=mdir), bool)
    ex = predict.explain(feats, models_dir=mdir)
    assert set(ex["contributions"]) == set(cfg["signal_groups"])
    assert ex["dominant_signal"] in set(cfg["signal_groups"]) | {None}
    with pytest.raises(ValueError):
        predict.predict_drift_score([1.0, 2.0], models_dir=mdir)          # wrong length list


@needs_data
def test_retrain_after_new_conversations_arrive(tmp_path):
    """Simulates Person A delivering more data: append a jsonl, rerun, nothing else changes."""
    from src.classifier.train import run_training

    cfg = _fast_cfg(tmp_path)
    raw = pd.concat([pd.read_csv(REPO_ROOT / "data/splits" / f"{s}.csv") for s in ("train", "val", "test")])
    ddir = tmp_path / "logs"
    ddir.mkdir()
    raw.to_json(ddir / "batch1.jsonl", orient="records", lines=True)
    cfg["data"].update({"source": "raw_logs", "raw_logs_glob": str(ddir / "*.jsonl")})
    first = run_training(cfg, verbose=False)

    extra = raw.copy()
    extra["conversation_id"] = "NEW-" + extra["conversation_id"]
    extra.to_json(ddir / "batch2.jsonl", orient="records", lines=True)
    second = run_training(cfg, verbose=False)

    assert second["info"]["fingerprint"] != first["info"]["fingerprint"]
    assert second["info"]["n_conversations"] == 2 * first["info"]["n_conversations"]
    d1, d2 = first["df"], second["df"]
    s1 = d1.groupby(CID)[SPLIT].first()
    s2 = d2.groupby(CID)[SPLIT].first()
    assert (s2.loc[s1.index] == s1).all()                                 # old conversations kept their split
    hist = pd.read_csv(tmp_path / "reports" / "runs_history.csv")
    assert len(hist) == 2
