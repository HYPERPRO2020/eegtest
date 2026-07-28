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

Uses quality_summary.csv (score.py) and faa_summary.csv (faa.py), merged on
file, restricted to condition=="EC" (one row per subject, avoiding the
pseudo-replication of counting EC and EO as independent) and deduplicated by
md5 (see ingest.py -- this dataset ships byte-identical recordings under
different subject labels).

Usage (from repo root, after score.py and faa.py):
    .venv/Scripts/python.exe neuroqa/study_b.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score

OUT_DIR = Path(__file__).resolve().parent / "outputs"


def load_merged() -> pd.DataFrame:
    quality = pd.read_csv(OUT_DIR / "quality_summary.csv")
    faa = pd.read_csv(OUT_DIR / "faa_summary.csv")
    manifest = pd.read_csv(OUT_DIR / "ingest_manifest.csv")[["file", "md5"]]

    df = quality.merge(faa[["file", "faa"]], on="file", how="inner")
    df = df.merge(manifest, on="file", how="left")
    df = df[df.condition == "EC"]
    df = df.drop_duplicates(subset="md5")
    df = df.dropna(subset=["quality_alpha_pct", "raw_severity_mean", "faa"])
    df["group_mdd"] = (df.group == "MDD").astype(int)
    return df.reset_index(drop=True)


def regression_1(df: pd.DataFrame):
    """quality ~ group + severity"""
    X = sm.add_constant(df[["group_mdd", "raw_severity_mean"]])
    y = df["quality_alpha_pct"]
    model = sm.OLS(y, X).fit()
    return model


def regression_2(df: pd.DataFrame):
    """FAA ~ group + quality"""
    X = sm.add_constant(df[["group_mdd", "quality_alpha_pct"]])
    y = df["faa"]
    model = sm.OLS(y, X).fit()
    return model


def regression_3(df: pd.DataFrame):
    """quality alone -> group classifier, 5-fold stratified CV."""
    X = df[["quality_alpha_pct"]].values
    y = df["group_mdd"].values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    clf = LogisticRegression()
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    acc = accuracy_score(y, pred)
    auc = roc_auc_score(y, proba)
    baseline = max(y.mean(), 1 - y.mean())  # majority-class accuracy
    return {"accuracy": acc, "auc": auc, "majority_class_baseline": baseline, "n": len(y)}


def main():
    df = load_merged()
    print(f"Study B sample: {len(df)} recordings "
          f"({df.group.value_counts().to_dict()}), condition=EC, deduplicated by md5\n")

    print("=" * 70)
    print("Regression 1: quality[alpha] ~ group + raw_severity_mean")
    print("=" * 70)
    m1 = regression_1(df)
    print(m1.summary().tables[1])

    print("\n" + "=" * 70)
    print("Regression 2: FAA ~ group + quality[alpha]")
    print("=" * 70)
    m2 = regression_2(df)
    print(m2.summary().tables[1])

    print("\n" + "=" * 70)
    print("Regression 3: quality[alpha] alone -> group (5-fold CV logistic regression)")
    print("=" * 70)
    r3 = regression_3(df)
    print(f"  accuracy               : {r3['accuracy']:.3f}")
    print(f"  AUC                    : {r3['auc']:.3f}")
    print(f"  majority-class baseline: {r3['majority_class_baseline']:.3f}")
    print(f"  n                      : {r3['n']}")
    if r3["auc"] > 0.65:
        print("  NOTE: quality alone predicts group meaningfully above chance --")
        print("  that's a leakage warning (quality should track signal cleanliness,")
        print("  not depression status), not a feature to lean on.")
    else:
        print("  quality alone is close to chance at predicting group -- consistent")
        print("  with quality measuring signal cleanliness rather than leaking group.")

    summary_path = OUT_DIR / "study_b_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Regression 1: quality[alpha] ~ group + raw_severity_mean\n")
        f.write(str(m1.summary()))
        f.write("\n\nRegression 2: FAA ~ group + quality[alpha]\n")
        f.write(str(m2.summary()))
        f.write("\n\nRegression 3: quality[alpha] alone -> group (5-fold CV)\n")
        for k, v in r3.items():
            f.write(f"  {k}: {v}\n")
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
