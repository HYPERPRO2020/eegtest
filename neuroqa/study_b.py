"""NeuroQA Study B — three regressions relating quality, severity, FAA, and group.

    1. quality ~ group + severity
       Does the endpoint-aware (alpha) quality index track raw artifact
       severity once group is accounted for, or does group leak into
       quality on its own (which would be a red flag -- quality is supposed
       to measure signal cleanliness, not depression status)?
    2. FAA ~ group + quality
       Does data quality confound the FAA-group relationship that's the
       whole reason FAA gets computed in the first place?
    3. quality alone -> group classifier
       If quality-only can predict group above chance, that's exactly the
       leakage regression 1 is checking for, from the other direction:
       quality should NOT be a usable depression classifier on its own.

Operates on an in-memory list of per-recording rows (one dict per accepted,
successfully-scored upload) rather than reading fixed local CSVs -- the
caller (pipeline.py) assembles that list from whatever the user uploaded.
Each row needs: file, group ("healthy"/"depressed"), quality_alpha_pct,
raw_severity_mean, faa.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

CV_SEED = 0


def rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["quality_alpha_pct", "raw_severity_mean", "faa", "group"])
    df["group_mdd"] = (df.group == "depressed").astype(int)
    return df.reset_index(drop=True)


def regression_1(df: pd.DataFrame):
    """quality ~ group + severity"""
    X = sm.add_constant(df[["group_mdd", "raw_severity_mean"]])
    y = df["quality_alpha_pct"]
    return sm.OLS(y, X).fit()


def regression_2(df: pd.DataFrame):
    """FAA ~ group + quality"""
    X = sm.add_constant(df[["group_mdd", "quality_alpha_pct"]])
    y = df["faa"]
    return sm.OLS(y, X).fit()


def regression_3(df: pd.DataFrame, n_splits: int = 5) -> dict:
    """quality alone -> group classifier, stratified CV.

    n_splits is capped to the smaller class's count (StratifiedKFold can't
    have more folds than the rarest class has members) -- relevant for
    small uploaded batches near the MIN_PER_GROUP floor in manifest.py.
    """
    X = df[["quality_alpha_pct"]].values
    y = df["group_mdd"].values
    n_splits = max(2, min(n_splits, int(np.bincount(y).min())))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=CV_SEED)
    clf = LogisticRegression()
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    acc = accuracy_score(y, pred)
    auc = roc_auc_score(y, proba) if len(set(y)) > 1 else float("nan")
    baseline = max(y.mean(), 1 - y.mean())
    return {
        "accuracy": float(acc), "auc": float(auc),
        "majority_class_baseline": float(baseline), "n": int(len(y)), "n_splits": n_splits,
        "leakage_flag": bool(auc > 0.65) if not np.isnan(auc) else False,
    }


def _ols_summary(model) -> dict:
    return {
        "params": {k: float(v) for k, v in model.params.items()},
        "pvalues": {k: float(v) for k, v in model.pvalues.items()},
        "conf_int": {k: [float(v) for v in row] for k, row in model.conf_int().iterrows()},
        "rsquared": float(model.rsquared),
        "nobs": int(model.nobs),
    }


def run_study_b(rows: list[dict]) -> dict:
    """Run all three analyses over an uploaded batch's per-recording rows.

    Returns a JSON-safe dict: {n, regression_1, regression_2, regression_3,
    finding}. `finding` is a short, honest, direction-agnostic statement --
    see module docstring; this does not get tuned toward a dramatic result.
    """
    df = rows_to_frame(rows)
    m1 = regression_1(df)
    m2 = regression_2(df)
    r3 = regression_3(df)

    group_coef_p = m2.pvalues.get("group_mdd", 1.0)
    quality_coef_p = m2.pvalues.get("quality_alpha_pct", 1.0)
    if r3["leakage_flag"]:
        finding = (
            "quality alone predicts group meaningfully above chance "
            f"(AUC={r3['auc']:.2f} vs. {r3['majority_class_baseline']:.2f} baseline) -- "
            "that's a leakage warning: the group difference in FAA may be substantially "
            "a contamination artifact rather than a brain-signal difference."
        )
    elif group_coef_p < 0.05 and quality_coef_p < 0.05:
        finding = (
            "group predicts FAA independent of quality, and quality alone doesn't predict "
            "group above chance -- consistent with FAA carrying real signal in this sample, "
            "not just contamination correlated with diagnosis."
        )
    else:
        finding = (
            "no strong evidence either way in this sample (group and/or quality effects on "
            "FAA are not significant at p<0.05) -- treat this as inconclusive, not as support "
            "for either an artifact or a real-signal reading of FAA."
        )

    return {
        "n": int(len(df)),
        "regression_1_quality_on_group_severity": _ols_summary(m1),
        "regression_2_faa_on_group_quality": _ols_summary(m2),
        "regression_3_quality_classifies_group": r3,
        "finding": finding,
    }
