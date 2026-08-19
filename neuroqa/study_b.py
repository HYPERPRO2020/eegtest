"""NeuroQA Study B — three regressions relating quality, severity, FAA, and group.

    1. quality ~ group + severity
       Is contamination itself different between groups, once each
       recording's own *clinical* severity (BDI/HAM-D or similar, from the
       upload manifest -- see manifest.py) is controlled for? This is the
       brief's Test B.1 and needs the diagnosis-side severity score, not a
       property of the recording itself -- using an EEG-derived quantity
       here (e.g. mean raw artifact severity) would make this regression
       circular, since quality is itself computed from artifact severity.
       Skipped (not silently substituted) when a batch has no severity data
       at all -- see run_study_b.
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
Each row needs: file, group ("healthy"/"depressed"), quality_alpha_pct, faa,
and clinical_severity (may be None/NaN -- see run_study_b).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

CV_SEED = 0
MIN_PER_GROUP_FOR_SEVERITY = 2  # mirrors manifest.MIN_PER_GROUP -- below this,
# a group's severity slice is too thin to regress on, not just "missing".


def rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    """Base frame for regressions 2 and 3, which don't need severity."""
    df = pd.DataFrame(rows)
    if "clinical_severity" not in df.columns:
        df["clinical_severity"] = np.nan
    # Callers disagree on type (run_local.py passes a parsed float or None
    # straight from manifest.py; webapp.py passes whatever raw JSON value the
    # browser sent, which may be a numeric string or "") -- coerce once here
    # rather than trusting every caller to normalize it themselves.
    df["clinical_severity"] = pd.to_numeric(df["clinical_severity"], errors="coerce")
    df = df.dropna(subset=["quality_alpha_pct", "faa", "group"])
    df["group_mdd"] = (df.group == "depressed").astype(int)
    return df.reset_index(drop=True)


def severity_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Subset of `df` with a usable clinical_severity value, for regression 1
    only. Separate from rows_to_frame's base frame so a batch with no
    severity data (e.g. Mumtaz/HUSM's public deposit) doesn't lose every
    recording from regressions 2/3 just because regression 1 can't run."""
    return df.dropna(subset=["clinical_severity"]).reset_index(drop=True)


def regression_1(df: pd.DataFrame):
    """quality ~ group + severity (clinical severity, not an EEG-derived
    quantity -- see module docstring). Caller must pass severity_frame(df),
    already filtered to rows with a real value."""
    X = sm.add_constant(df[["group_mdd", "clinical_severity"]])
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
    finding}. Regression 1 is None (with a `regression_1_skipped_reason`)
    when too few recordings in this batch carry a clinical severity score --
    reported honestly rather than silently dropped or faked. `finding` is a
    short, honest, direction-agnostic statement -- see module docstring;
    this does not get tuned toward a dramatic result.
    """
    df = rows_to_frame(rows)
    m2 = regression_2(df)
    r3 = regression_3(df)

    sev_df = severity_frame(df)
    sev_counts = sev_df.groupby("group_mdd").size() if len(sev_df) else pd.Series(dtype=int)
    enough_for_severity = (sev_counts >= MIN_PER_GROUP_FOR_SEVERITY).sum() >= 2 if len(sev_counts) else False
    m1 = regression_1(sev_df) if enough_for_severity else None
    skip_reason = None if enough_for_severity else (
        "no clinical severity scores in this batch -- Test B.1 (quality ~ group + severity) needs "
        f"at least {MIN_PER_GROUP_FOR_SEVERITY} recordings per group with a severity value"
        if len(sev_df) == 0 else
        f"only {len(sev_df)} recording(s) in this batch carry a severity score -- too few to regress on"
    )

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
        "n_with_severity": int(len(sev_df)),
        "regression_1_quality_on_group_severity": _ols_summary(m1) if m1 is not None else None,
        "regression_1_skipped_reason": skip_reason,
        "regression_2_faa_on_group_quality": _ols_summary(m2),
        "regression_3_quality_classifies_group": r3,
        "finding": finding,
    }
