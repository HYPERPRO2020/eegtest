"""NeuroQA -- Peter's ask: 4-5 independent FAA-based depression classifiers,
one per Study A preprocessing pipeline, that don't share data or talk to
each other, with a result reported for each.

Distinct from study_b.py's classifier (quality-alone -> group, checking for
contamination leakage): this one asks "if you just used this pipeline's FAA
number by itself to guess depressed/healthy, how well would that pipeline
do" -- for every pipeline in study_a.PIPELINES independently, on the same
Study A subsample rows, so the 5 results are directly comparable to each
other (same recordings, same CV seed/folds) without any pipeline's result
depending on another's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from study_a import PIPELINES
from study_b import bootstrap_auc_ci

CV_SEED = 0
MIN_N = 4  # need at least this many labeled recordings to fit+CV anything meaningful


def classify_by_pipeline(study_a_long_rows: list[dict], reference: str = "original") -> dict:
    """One independent classifier per pipeline: FAA alone -> group, stratified CV.

    study_a_long_rows: the same long-format rows study_a.spread_stats consumes
    ({"file","group","pipeline","reference","faa"}), e.g. results.json's
    study_a.long, or pipeline.aggregate's study_a_rows before spread_stats.
    reference: which reference scheme's FAA to classify on (default "original"
    -- classifying on both references would double-count the same
    recordings' signal into 10 "independent" classifiers, which they
    wouldn't really be).

    Returns {pipeline_name: {accuracy, auc, baseline, n, n_splits}} or
    {pipeline_name: {"error": "..."}} if that pipeline's subsample is too
    small/single-class in this batch.
    """
    df = pd.DataFrame(study_a_long_rows)
    if len(df) == 0:
        return {p: {"error": "no Study A rows in this batch"} for p in PIPELINES}
    df = df[df.reference == reference]

    results = {}
    for pipeline in PIPELINES:
        sub = df[df.pipeline == pipeline].dropna(subset=["faa", "group"])
        y = (sub.group == "depressed").astype(int).values
        if len(sub) < MIN_N or len(set(y)) < 2:
            results[pipeline] = {"error": f"only {len(sub)} recording(s) / "
                                           f"{len(set(y))} class(es) -- need >={MIN_N} and both classes"}
            continue

        X = sub[["faa"]].values
        n_splits = max(2, min(5, int(np.bincount(y).min())))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=CV_SEED)
        clf = LogisticRegression()
        proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
        pred = (proba >= 0.5).astype(int)
        baseline = float(max(y.mean(), 1 - y.mean()))
        auc = float(roc_auc_score(y, proba)) if len(set(y)) > 1 else float("nan")
        # Bootstrap CI: with n as small as MIN_N=4, a bare point-estimate AUC
        # invites over-interpretation -- see study_b.bootstrap_auc_ci.
        _, auc_ci_lo, auc_ci_hi = bootstrap_auc_ci(y, proba) if not np.isnan(auc) else (float("nan"),) * 3
        results[pipeline] = {
            "accuracy": float(accuracy_score(y, pred)),
            "auc": auc,
            "auc_ci_lo": auc_ci_lo, "auc_ci_hi": auc_ci_hi,
            "baseline": baseline,
            "n": int(len(y)),
            "n_splits": n_splits,
        }
    return results
