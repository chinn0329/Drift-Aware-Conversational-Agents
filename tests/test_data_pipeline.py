"""Tests for the multi-file / multi-model data handling. The fixture REPRODUCES the quirks seen in the real logs:
  * legacy rows (9 columns, no model/run metadata) repeated identically in two files
  * the same task run by different models with DIFFERENT outcomes (not a label error)
  * runs flagged label_valid=0 / truncated
  * a mistyped column ('self_consi  stency_variance') holding values the real column lacks
It is a unit-test fixture, not training data.
"""
import copy

import numpy as np
import pandas as pd
import pytest

from src.classifier.config import load_config
from src.classifier.cv import make_folds
from src.classifier.data import CID, MODEL, SPLIT, TASK, Y, load_table, prepare
from src.classifier.diagnostics import signal_by_source, within_task_paired
from src.classifier.metrics import bootstrap_diff_ci, within_task_auroc

FEATS = ["self_consistency_variance", "contradiction_score", "confidence", "confidence_delta",
         "silent_commitment_ratio"]
TYPO = "self_consi  stency_variance"


def _rows(tasks, success, model=None, run=None, n_turns=3, seed=0, meta=None):
    rng = np.random.default_rng(seed)
    out = []
    for t in tasks:
        for turn in range(1, n_turns + 1):
            r = {"conversation_id": f"sharded-{t}", "turn_index": turn,
                 **{f: float(rng.uniform(0, 1)) for f in FEATS}, "eventual_success": bool(success(t)),
                 "source": "humaneval" if int(t[1:]) % 2 == 0 else "lcb_easy"}
            if model:
                r.update({"assistant_model": model, "run_index": run, "label_valid": 1.0, "truncated": 0.0,
                          "any_turn_truncated": 0.0, "label_reason": "ok", "extractor_version": "v5",
                          "n_resamples": 2.0, **(meta or {}).get(t, {})})
            out.append(r)
    return out


@pytest.fixture()
def logs(tmp_path):
    tasks = [f"T{i}" for i in range(24)]
    legacy = _rows(tasks[:12], lambda t: int(t[1:]) % 3 == 0, seed=1)
    gpt = _rows(tasks[:8], lambda t: int(t[1:]) % 2 == 0, "gptX", 0, seed=2)
    for r in gpt:                                       # values live ONLY under the mistyped column name
        r[TYPO], r["self_consistency_variance"] = r["self_consistency_variance"], np.nan
    trunc = {t: {"label_valid": 0.0, "label_reason": "truncated", "any_turn_truncated": 1.0} for t in ("T4", "T5")}
    llama = _rows(tasks[:8], lambda t: False, "llamaY", 0, seed=3, meta=trunc)
    pd.DataFrame(legacy + gpt).to_json(tmp_path / "raw_logs2.jsonl", orient="records", lines=True)
    pd.DataFrame(legacy + llama).to_json(tmp_path / "raw_logs3.jsonl", orient="records", lines=True)
    cfg = copy.deepcopy(load_config())
    cfg["data"].update({"source": "raw_logs", "raw_logs_glob": str(tmp_path / "*.jsonl")})
    return cfg


def test_same_task_different_models_is_not_a_label_conflict(logs):
    df = load_table(logs)                                     # would raise if keyed by task id alone
    assert df[CID].nunique() == df.groupby([TASK, MODEL]).ngroups
    t0 = df[df[TASK] == "sharded-T0"]
    assert set(t0[MODEL]) == {"legacy", "gptX", "llamaY"}     # three separate runs of one task
    assert t0.groupby(CID)[Y].first().nunique() == 2          # and they ended differently


def test_legacy_rows_repeated_across_files_are_deduplicated(logs):
    df, log = load_table(logs, return_log=True)
    assert log["dedupe_across_files"] == {"rows_dropped": 36, "dropped_rows_that_differed": 0}
    assert df[df[MODEL] == "legacy"][CID].nunique() == 12
    assert len(df[df[MODEL] == "legacy"]) == 36               # one copy of each (run, turn), not two


def test_invalid_and_truncated_runs_are_dropped_whole(logs):
    df, log = load_table(logs, return_log=True)
    assert log["filters"]["label_valid==0"]["runs_dropped"] == 2
    assert not df[CID].str.contains(r"T4\|llamaY|T5\|llamaY").any()
    assert df[df[MODEL] == "llamaY"][CID].nunique() == 6      # 8 runs - 2 truncated
    logs["data"]["filters"]["require_label_valid"] = False    # configurable
    logs["data"]["filters"]["exclude_truncated"] = False
    assert load_table(logs)[df.columns[0]].nunique() > df[CID].nunique()


def test_mistyped_column_values_are_rescued(logs):
    df, log = load_table(logs, return_log=True)
    assert log["aliases"][TYPO] == {"into": "self_consistency_variance", "cells_rescued": 24, "conflicting_values": 0}
    assert df["self_consistency_variance"].notna().all()
    assert TYPO not in df.columns


def test_all_runs_of_a_task_share_a_split(logs):
    df = load_table(logs)
    assert (df.groupby(TASK)[SPLIT].nunique() == 1).all()


def test_report_exposes_what_matters(logs):
    _, _, rep = prepare(logs, verbose=False)
    assert rep["tasks_with_both_outcomes"] >= 1 and rep["tasks_with_several_runs"] >= 8
    assert set(rep["per_model"]) == {"legacy", "gptX", "llamaY"}
    assert "2" in rep["n_resamples"]
    assert any("<=2 resamples" in w for w in rep["warnings"])


def test_folds_never_split_a_task_across_train_and_test(logs):
    df = load_table(logs)
    for seed in range(3):
        for tr, te in make_folds(df, 3, seed):
            assert not set(df.iloc[tr][TASK]) & set(df.iloc[te][TASK])


def test_within_task_auroc_uses_only_same_task_pairs():
    # task A: failed run scores higher (right); task B: failed run scores lower (wrong) -> 0.5
    tasks = ["A", "A", "B", "B", "C", "C"]
    runs = ["a1", "a2", "b1", "b2", "c1", "c2"]
    y = [1, 0, 1, 0, 1, 1]            # task C has no success -> ignored
    p = [0.9, 0.2, 0.1, 0.8, 0.99, 0.01]
    r = within_task_auroc(tasks, runs, y, p, n_boot=50)
    assert (r["n_tasks"], r["n_pairs"]) == (2, 2) and r["auroc"] == pytest.approx(0.5)
    assert np.isnan(within_task_auroc(["A", "A"], ["a", "b"], [1, 1], [0.1, 0.2])["auroc"])


def test_bootstrap_diff_detects_a_real_gain():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    groups = np.repeat(np.arange(100), 4)
    good = y + rng.normal(0, 0.4, 400)
    noise = rng.normal(0, 1, 400)
    d, lo, hi = bootstrap_diff_ci(y, good, noise, groups, n=200)
    assert d > 0.2 and lo > 0
    d2, lo2, hi2 = bootstrap_diff_ci(y, noise, rng.normal(0, 1, 400), groups, n=200)
    assert lo2 < 0 < hi2              # two noise scores: interval contains 0


def test_diagnostics_run_on_the_fixture(logs):
    df = load_table(logs)
    feats = FEATS
    s = signal_by_source(df, feats, min_runs=2, n_boot=20)
    assert set(s.columns) >= {"source", "signal", "auroc", "ci_low", "ci_high", "as_expected"}
    w = within_task_paired(df, feats, n_boot=20)
    assert len(w) == len(feats) and w.n_tasks.iloc[0] >= 1


def test_leave_one_model_out_reports_when_nothing_is_left_to_train_on():
    from src.classifier.diagnostics import leave_one_group_out
    rng = np.random.default_rng(0)
    rows = []
    for task in range(30):                      # both models ran every task -> strict protocol has no training data
        for model in ("a", "b"):
            fail = int(rng.uniform() < 0.5)
            for turn in (1, 2):
                rows.append({"conversation_id": f"{task}|{model}|0", "task_id": f"t{task}", "model": model,
                             "turn_index": turn, "y": fail, "group": "g", "f": rng.uniform()})
    df = pd.DataFrame(rows)
    cfg = load_config()
    strict = leave_one_group_out(df, ["f"], "logreg", cfg, None, "model", unseen_tasks=True)
    assert (strict.train_runs == 0).all() and strict.note.str.contains("no training runs left").all()
    lenient = leave_one_group_out(df, ["f"], "logreg", cfg, None, "model", unseen_tasks=False)
    assert lenient.auroc.notna().all() and (lenient.train_runs == 30).all()
