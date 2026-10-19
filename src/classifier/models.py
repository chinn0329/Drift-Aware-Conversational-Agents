"""The three candidate classifiers (logistic regression / XGBoost / MLP) behind one interface."""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:  # pragma: no cover - fallback keeps the pipeline runnable without xgboost
    HAS_XGB = False

ORDER = ["logreg", "xgboost", "mlp"]  # simplest first; used to break near-ties in model selection
_PIPE = {"logreg", "mlp"}             # models wrapped in an imputer+scaler pipeline (params get a clf__ prefix)


def _tuple_lists(params: dict) -> dict:
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in params.items()}


def make_estimator(name: str, params: dict, seed: int, balance: bool = True):
    params = _tuple_lists(params)
    if name == "logreg":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced" if balance else None,
                                 random_state=seed, **params)
        return Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                         ("sc", StandardScaler()), ("clf", clf)])
    if name == "xgboost":
        if HAS_XGB:
            return XGBClassifier(eval_metric="logloss", random_state=seed, n_jobs=1, verbosity=0, **params)
        p = {"max_iter": params.get("n_estimators", 200), "learning_rate": params.get("learning_rate", 0.05),
             "max_depth": params.get("max_depth", 2), "l2_regularization": params.get("reg_lambda", 5.0),
             "min_samples_leaf": 5}
        return HistGradientBoostingClassifier(random_state=seed, **p)
    if name == "mlp":
        clf = MLPClassifier(random_state=seed, **params)
        return Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                         ("sc", StandardScaler()), ("clf", clf)])
    raise ValueError(f"Unknown model '{name}'")


def _prefixed(name: str, params: dict) -> dict:
    return {f"clf__{k}": v for k, v in params.items()} if name in _PIPE else dict(params)


def _unprefix(name: str, params: dict) -> dict:
    return {k.removeprefix("clf__"): v for k, v in params.items()} if name in _PIPE else dict(params)


def _fit_kwargs(name: str, y, balance: bool) -> dict:
    # XGBoost / HGB have no class_weight argument -> use balanced sample weights.
    if balance and name == "xgboost":
        return {"sample_weight": compute_sample_weight("balanced", y)}
    return {}


def should_tune(cfg: dict, n_conversations: int) -> bool:
    t = cfg["training"]["tune"]
    if t == "auto":
        return n_conversations >= cfg["training"]["tune_min_conversations"]
    return bool(t)


def fit_model(name, X, y, groups, cfg, tune=False, params=None, seed=0):
    """Fit one model. Returns (fitted_estimator, params_actually_used)."""
    balance = cfg["training"]["balance_classes"]
    base = dict(cfg["model_params"][name])
    if params:
        base.update(params)
    est = make_estimator(name, base, seed, balance)
    y = np.asarray(y)
    grid = cfg.get("tune_grids", {}).get(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if tune and grid and len(set(groups)) >= 9:
            k = max(2, min(cfg["training"]["tune_folds"], len(set(groups[y == 1])), len(set(groups[y == 0]))))
            param_grid = _prefixed(name, {kk: [tuple(x) if isinstance(x, list) else x for x in vv]
                                          for kk, vv in grid.items()})
            gs = GridSearchCV(est, param_grid, scoring="roc_auc", refit=True, error_score=np.nan,
                              cv=StratifiedGroupKFold(k, shuffle=True, random_state=seed))
            gs.fit(X, y, groups=groups, **_fit_kwargs(name, y, balance))
            used = {**base, **_unprefix(name, gs.best_params_)}
            return gs.best_estimator_, used
        est.fit(X, y, **_fit_kwargs(name, y, balance))
    return est, base
