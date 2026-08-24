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

       Caveat this regression cannot resolve on its own (see methodology
       review, 2026-08): `faa` here is NeuroQA's own quality-weighted FAA
       (pipeline.py's score_and_faa weights F3/F4 alpha power by the exact
       per-channel quality array that quality_alpha_frontal_pct then
       averages over), so "is the quality coefficient significant" is
       partly circular by construction and biased toward under-detecting a
       real quality-FAA association. regression_2_raw below reruns this
       same regression on `faa_raw` (unweighted FAA) as a robustness check.
       Separately, a significant/non-significant quality coefficient here
       is not by itself proof of "confound" vs. "not a confound" -- if
       quality is a MEDIATOR (severity -> movement/blink/muscle tension ->
       contamination -> biased FAA, which is this project's own stated
       rationale for why quality and severity might correlate) rather than
       a pre-existing confounder, this is the "Table 2 Fallacy"
       (Westreich & Greenland 2013): a regression coefficient is only
       cleanly causally interpretable for the variable the model was built
       to isolate (group), not for a second covariate read off the same
       fit. mediation_analysis below is the purpose-built tool for the
       mediator question; treat this regression as an association-level
       robustness check, not a causal confound-elimination result.
    3. quality alone -> group classifier
       If quality-only can predict group above chance, that's exactly the
       leakage regression 1 is checking for, from the other direction:
       quality should NOT be a usable depression classifier on its own.

Each of the three runs twice per batch: once on `quality_alpha_frontal_pct`
(F3/F4/Fp1/Fp2 only -- the direct test of "frontal muscle contamination
confounds FAA," the hypothesis this project actually tests) and once on
`quality_alpha_pct` (whole-scalp mean, diluted with 15+ channels that have
nothing to do with frontal EMG or FAA). Frontal is primary -- it's what the
`finding` text and Holm-Bonferroni-corrected significance calls are based
on -- whole-scalp is reported alongside as secondary/comparison context,
never silently dropped, so picking whichever channel subset gives the
better-looking number isn't an option analysts (or readers) can quietly
reach for. See score.py's FRONTAL_QUALITY_CHANNELS for the exact channel set.

Operates on an in-memory list of per-recording rows (one dict per accepted,
successfully-scored upload) rather than reading fixed local CSVs -- the
caller (pipeline.py) assembles that list from whatever the user uploaded.
Each row needs: file, group ("healthy"/"depressed"), quality_alpha_pct,
quality_alpha_frontal_pct, faa, and clinical_severity (may be None/NaN --
see run_study_b).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from statsmodels.stats.multitest import multipletests

CV_SEED = 0
BOOTSTRAP_N = 2000  # resamples for regression_3's AUC CI and mediation_analysis's ACME/ADE CIs
MIN_PER_GROUP_FOR_SEVERITY = 2  # mirrors manifest.MIN_PER_GROUP -- below this,
# a group's severity slice is too thin to regress on, not just "missing".
MIN_N_FOR_OLS = 6  # floor for any 3-parameter OLS fit here (const + group + one
# more term) -- below this a fit can technically run (e.g. 4 obs, 1 residual
# df) and produce a p-value indistinguishable in the output from a
# well-powered one; 6 leaves >=3 residual df.

PRIMARY_QUALITY_COL = "quality_alpha_frontal_pct"   # the actual hypothesis test
SECONDARY_QUALITY_COL = "quality_alpha_pct"          # whole-scalp, reported alongside


def rows_to_frame(rows: list[dict], quality_col: str) -> pd.DataFrame:
    """Base frame for regressions 2 and 3, which don't need severity."""
    df = pd.DataFrame(rows)
    if "clinical_severity" not in df.columns:
        df["clinical_severity"] = np.nan
    if quality_col not in df.columns:
        df[quality_col] = np.nan
    # Callers disagree on type (run_local.py passes a parsed float or None
    # straight from manifest.py; webapp.py passes whatever raw JSON value the
    # browser sent, which may be a numeric string or "") -- coerce once here
    # rather than trusting every caller to normalize it themselves.
    df["clinical_severity"] = pd.to_numeric(df["clinical_severity"], errors="coerce")
    df = df.dropna(subset=[quality_col, "faa", "group"])
    df["group_mdd"] = (df.group == "depressed").astype(int)
    return df.reset_index(drop=True)


def severity_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Subset of `df` with a usable clinical_severity value, for regression 1
    only. Separate from rows_to_frame's base frame so a batch with no
    severity data (e.g. Mumtaz/HUSM's public deposit) doesn't lose every
    recording from regressions 2/3 just because regression 1 can't run."""
    return df.dropna(subset=["clinical_severity"]).reset_index(drop=True)


def regression_1(df: pd.DataFrame, quality_col: str):
    """quality ~ group + severity (clinical severity, not an EEG-derived
    quantity -- see module docstring). Caller must pass severity_frame(df),
    already filtered to rows with a real value."""
    X = sm.add_constant(df[["group_mdd", "clinical_severity"]])
    y = df[quality_col]
    return sm.OLS(y, X).fit()


def regression_2(df: pd.DataFrame, quality_col: str):
    """FAA ~ group + quality"""
    X = sm.add_constant(df[["group_mdd", quality_col]])
    y = df["faa"]
    return sm.OLS(y, X).fit()


def regression_2_raw(df: pd.DataFrame, quality_col: str):
    """FAA_raw ~ group + quality -- robustness check for regression_2 using
    unweighted FAA (see module docstring: regression_2's usual `faa` DV is
    partly built from the same per-channel quality array that quality_col
    aggregates, which biases it toward under-detecting a real association).
    Caller must pass a frame already filtered to rows with a real faa_raw
    value (older cached rows predating this column won't have one)."""
    X = sm.add_constant(df[["group_mdd", quality_col]])
    y = df["faa_raw"]
    return sm.OLS(y, X).fit()


def mediation_analysis(df: pd.DataFrame, quality_col: str,
                        n_boot: int = BOOTSTRAP_N, seed: int = CV_SEED) -> dict:
    """Bootstrapped causal mediation analysis (product-of-coefficients /
    Baron-Kenny method, per Imai, Keele & Tingley 2010's ACME/ADE framing)
    for group -> quality -> FAA: is contamination quality a MEDIATOR of the
    group-FAA relationship, as opposed to regression_2's bare "is the
    quality coefficient significant" check, which can't distinguish a
    mediator from a confounder (Table 2 Fallacy -- see module docstring).

    ACME  (a*b): the group -> quality -> FAA indirect/mediated effect.
    ADE   (c'):  the group -> FAA direct effect holding quality constant
                 (= regression_2's group_mdd coefficient).
    total (c):   the group -> FAA total effect with quality left out of the
                 model entirely.
    proportion_mediated = ACME / total -- undefined (nan) if total is ~0.
    """
    def fit_paths(frame: pd.DataFrame) -> tuple[float, float, float]:
        a = sm.OLS(frame[quality_col], sm.add_constant(frame[["group_mdd"]])).fit().params["group_mdd"]
        m = sm.OLS(frame["faa"], sm.add_constant(frame[["group_mdd", quality_col]])).fit()
        b, c_prime = m.params[quality_col], m.params["group_mdd"]
        c = sm.OLS(frame["faa"], sm.add_constant(frame[["group_mdd"]])).fit().params["group_mdd"]
        return a * b, c_prime, c

    acme_point, ade_point, total_point = fit_paths(df)

    rng = np.random.RandomState(seed)
    idx = np.arange(len(df))
    acmes, ades, totals = [], [], []
    for _ in range(n_boot):
        b_idx = rng.choice(idx, size=len(idx), replace=True)
        boot_df = df.iloc[b_idx].reset_index(drop=True)
        if boot_df["group_mdd"].nunique() < 2:
            continue  # degenerate resample (all one group) -- skip, don't crash
        try:
            acme, ade, total = fit_paths(boot_df)
        except (ValueError, np.linalg.LinAlgError):
            continue
        acmes.append(acme); ades.append(ade); totals.append(total)

    def summarize(point: float, boots: list[float]) -> dict:
        if not boots:
            return {"point": point, "ci_lo": float("nan"), "ci_hi": float("nan")}
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return {"point": float(point), "ci_lo": float(lo), "ci_hi": float(hi)}

    return {
        "acme": summarize(acme_point, acmes),
        "ade": summarize(ade_point, ades),
        "total_effect": summarize(total_point, totals),
        "proportion_mediated": float(acme_point / total_point) if abs(total_point) > 1e-12 else float("nan"),
        "n_boot_valid": len(acmes),
    }


def bootstrap_auc_ci(y_true: np.ndarray, y_score: np.ndarray,
                      n_boot: int = BOOTSTRAP_N, seed: int = CV_SEED) -> tuple[float, float, float]:
    """Bootstrap 95% CI for an AUC: resample (y_true, y_score) pairs together
    with replacement, recompute AUC each time. A bare point-estimate AUC from
    a handful of CV folds invites over-interpretation on small batches;
    reporting the interval alongside it makes that uncertainty visible
    instead of implied precision the sample size doesn't support."""
    rng = np.random.RandomState(seed)
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    idx = np.arange(len(y_true))
    aucs = []
    for _ in range(n_boot):
        b = rng.choice(idx, size=len(idx), replace=True)
        if len(set(y_true[b])) < 2:
            continue  # a resample with only one class has no AUC -- skip, don't crash
        aucs.append(roc_auc_score(y_true[b], y_score[b]))
    if not aucs:
        return float("nan"), float("nan"), float("nan")
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(np.mean(aucs)), float(lo), float(hi)


def regression_3(df: pd.DataFrame, quality_col: str, n_splits: int = 5) -> dict:
    """quality alone -> group classifier, stratified CV.

    n_splits is capped to the smaller class's count (StratifiedKFold can't
    have more folds than the rarest class has members) -- relevant for
    small uploaded batches near the MIN_PER_GROUP floor in manifest.py.
    """
    X = df[[quality_col]].values
    y = df["group_mdd"].values
    n_splits = max(2, min(n_splits, int(np.bincount(y).min())))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=CV_SEED)
    clf = LogisticRegression()
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    acc = accuracy_score(y, pred)
    auc = roc_auc_score(y, proba) if len(set(y)) > 1 else float("nan")
    auc_boot_mean, auc_ci_lo, auc_ci_hi = (
        bootstrap_auc_ci(y, proba) if not np.isnan(auc) else (float("nan"),) * 3
    )
    baseline = max(y.mean(), 1 - y.mean())
    return {
        "accuracy": float(acc), "auc": float(auc),
        "auc_ci_lo": auc_ci_lo, "auc_ci_hi": auc_ci_hi,
        "majority_class_baseline": float(baseline), "n": int(len(y)), "n_splits": n_splits,
        # CI-anchored, not a bare point-estimate threshold: a lone auc>0.65
        # cutoff is an arbitrary round number disconnected from the null AUC
        # of 0.5 and ignores exactly the uncertainty auc_ci_lo/hi exist to
        # capture (e.g. auc=0.63 with CI [0.51,0.75] would wrongly go
        # unflagged under a bare >0.65 rule; auc=0.66 with CI [0.35,0.90]
        # would wrongly get flagged despite spanning 0.5 by a wide margin).
        "leakage_flag": bool(not np.isnan(auc_ci_lo) and auc_ci_lo > 0.5),
    }


def _ols_summary(model) -> dict:
    return {
        "params": {k: float(v) for k, v in model.params.items()},
        "pvalues": {k: float(v) for k, v in model.pvalues.items()},
        "conf_int": {k: [float(v) for v in row] for k, row in model.conf_int().iterrows()},
        "rsquared": float(model.rsquared),
        "nobs": int(model.nobs),
    }


def _run_one_quality_variant(rows: list[dict], quality_col: str) -> dict:
    """Regressions 1-3 for a single quality column (frontal or whole-scalp).
    Returns the raw per-regression results; run_study_b adds the
    Holm-corrected p-values and finding text on top of the primary variant.
    """
    df = rows_to_frame(rows, quality_col)
    m2 = regression_2(df, quality_col)
    r3 = regression_3(df, quality_col)

    raw_df = df.dropna(subset=["faa_raw"]) if "faa_raw" in df.columns else df.iloc[0:0]
    m2_raw = regression_2_raw(raw_df, quality_col) if len(raw_df) >= MIN_N_FOR_OLS else None

    mediation = mediation_analysis(df, quality_col) if len(df) >= MIN_N_FOR_OLS else None

    sev_df = severity_frame(df)
    sev_counts = sev_df.groupby("group_mdd").size() if len(sev_df) else pd.Series(dtype=int)
    enough_for_severity = (
        (sev_counts >= MIN_PER_GROUP_FOR_SEVERITY).sum() >= 2 and len(sev_df) >= MIN_N_FOR_OLS
        if len(sev_counts) else False
    )
    m1 = regression_1(sev_df, quality_col) if enough_for_severity else None
    skip_reason = None if enough_for_severity else (
        "no clinical severity scores in this batch -- Test B.1 (quality ~ group + severity) needs "
        f"at least {MIN_PER_GROUP_FOR_SEVERITY} recordings per group with a severity value"
        if len(sev_df) == 0 else
        f"only {len(sev_df)} recording(s) in this batch carry a severity score -- below the "
        f"{MIN_N_FOR_OLS}-observation floor for a 3-parameter regression to be worth trusting"
    )

    # Holm-Bonferroni across this variant's own p-value family: group's and
    # quality's coefficients in regression 2, plus severity's in regression 1
    # when it ran. Corrects the raw p-values statsmodels reports before
    # anything downstream (the finding text) treats one as "significant" --
    # testing three-ish hypotheses per variant and not correcting for it
    # would understate how easy p<0.05 is to hit by chance alone.
    pval_names = ["group_mdd", quality_col]
    pvals = [m2.pvalues.get("group_mdd", 1.0), m2.pvalues.get(quality_col, 1.0)]
    if m1 is not None:
        pval_names.append("severity")
        pvals.append(m1.pvalues.get("clinical_severity", 1.0))
    _, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method="holm")
    corrected = dict(zip(pval_names, (float(p) for p in pvals_corrected)))

    return {
        "n": int(len(df)),
        "n_with_severity": int(len(sev_df)),
        "regression_1_quality_on_group_severity": _ols_summary(m1) if m1 is not None else None,
        "regression_1_skipped_reason": skip_reason,
        "regression_2_faa_on_group_quality": _ols_summary(m2),
        # Robustness check, not part of the corrected significance family
        # above (deliberately reported uncorrected/separately -- see module
        # docstring): same regression, unweighted FAA as the DV.
        "regression_2_raw_faa_on_group_quality": _ols_summary(m2_raw) if m2_raw is not None else None,
        "regression_2_raw_skipped_reason": None if m2_raw is not None else (
            f"fewer than {MIN_N_FOR_OLS} rows have a raw (unweighted) FAA value"),
        "mediation_analysis": mediation,
        "mediation_analysis_skipped_reason": None if mediation is not None else (
            f"fewer than {MIN_N_FOR_OLS} rows in this batch"),
        "regression_3_quality_classifies_group": r3,
        "holm_corrected_pvalues": corrected,
    }


def run_study_b(rows: list[dict]) -> dict:
    """Run all three analyses over an uploaded batch's per-recording rows,
    once for the frontal-specific quality variant (primary -- the direct
    test of the frontal-contamination hypothesis) and once for whole-scalp
    (secondary context, see module docstring). Both are always returned;
    `finding` is based on the frontal variant's Holm-corrected group
    p-value. A short, honest, direction-agnostic statement -- this does not
    get tuned toward a dramatic result.
    """
    frontal = _run_one_quality_variant(rows, PRIMARY_QUALITY_COL)
    whole_scalp = _run_one_quality_variant(rows, SECONDARY_QUALITY_COL)

    r3 = frontal["regression_3_quality_classifies_group"]
    group_coef_p_corrected = frontal["holm_corrected_pvalues"]["group_mdd"]

    # Robustness note: does the same group->FAA question, asked with an
    # unweighted FAA instead of NeuroQA's own quality-weighted one, land on
    # the same side of p<0.05? Uncorrected comparison (regression_2_raw
    # isn't part of the Holm family above) -- a coarse agreement check, not
    # a second confirmed finding.
    raw_m2 = frontal.get("regression_2_raw_faa_on_group_quality")
    raw_note = ""
    if raw_m2 is not None:
        raw_group_p = raw_m2["pvalues"].get("group_mdd")
        if raw_group_p is not None:
            agrees = (raw_group_p < 0.05) == (group_coef_p_corrected < 0.05)
            raw_note = (
                " A robustness check using unweighted (raw) FAA in place of NeuroQA's own "
                "quality-weighted FAA " +
                ("reaches the same conclusion" if agrees else "reaches a different conclusion") +
                f" (raw-FAA group p={raw_group_p:.3f}, uncorrected)."
            )

    if r3["leakage_flag"]:
        finding = (
            "quality alone predicts group meaningfully above chance "
            f"(AUC={r3['auc']:.2f} vs. {r3['majority_class_baseline']:.2f} baseline) -- "
            "that's a leakage warning: the group difference in FAA may be substantially "
            "a contamination artifact rather than a brain-signal difference." + raw_note
        )
    elif group_coef_p_corrected < 0.05:
        finding = (
            "group predicts FAA independent of frontal quality (Holm-Bonferroni corrected), "
            "and quality alone doesn't predict group above chance -- consistent with FAA "
            "carrying real signal in this sample, not just contamination correlated with diagnosis." + raw_note
        )
    else:
        finding = (
            "no strong evidence either way in this sample (group's effect on FAA is not "
            "significant at corrected p<0.05, controlling for frontal quality) -- treat this "
            "as inconclusive, not as support for either an artifact or a real-signal reading of FAA." + raw_note
        )

    return {
        "n": frontal["n"],
        "n_with_severity": frontal["n_with_severity"],
        "quality_variant_primary": PRIMARY_QUALITY_COL,
        "quality_variant_secondary": SECONDARY_QUALITY_COL,
        "frontal": frontal,
        "whole_scalp": whole_scalp,
        # top-level mirrors of the primary (frontal) variant, for callers/UI
        # that haven't been updated to read the nested structure yet.
        "regression_1_quality_on_group_severity": frontal["regression_1_quality_on_group_severity"],
        "regression_1_skipped_reason": frontal["regression_1_skipped_reason"],
        "regression_2_faa_on_group_quality": frontal["regression_2_faa_on_group_quality"],
        "regression_2_raw_faa_on_group_quality": frontal["regression_2_raw_faa_on_group_quality"],
        "mediation_analysis": frontal["mediation_analysis"],
        "regression_3_quality_classifies_group": frontal["regression_3_quality_classifies_group"],
        "finding": finding,
    }
