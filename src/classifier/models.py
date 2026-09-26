"""
Section 6.1 — Model choices and rationale

Three classifiers are trained and reported side by side (accuracy/AUROC,
training time, interpretability) rather than committing to one in advance:

  - Logistic Regression : primary classifier; 5 inputs, fast, interpretable coefficients.
  - XGBoost / LightGBM   : captures feature interactions; still interpretable via SHAP.
  - Small MLP (1-2 hidden layers) : tests whether a non-linear boundary helps at all.
"""


def build_logistic_regression():
    """Return an sklearn LogisticRegression configured for the 5-feature input."""
    raise NotImplementedError


def build_xgboost():
    """Return an XGBoost/LightGBM classifier configured for the 5-feature input."""
    raise NotImplementedError


def build_mlp():
    """Return a small (1-2 hidden layer) MLP classifier (PyTorch or sklearn MLPClassifier)."""
    raise NotImplementedError
