"""Neureidos FAA Spectral-Composition Tool (v1) -- CLI driver.

Per Neureidos_Tool_Build_Spec_v1: runs the same accepted-batch validation
Phase 1 uses, then for every accepted recording x reference (original,
average) x aperiodic_mode (fixed, knee), fits FOOOF/specparam on F3/F4's
quality-weighted averaged PSD and computes the three FAA variants +
oscillatory share (see neuroqa/spectral_composition.py). Writes a tidy
per-subject CSV plus a summary JSON: Analysis A (composition distributions),
B (classic vs. periodic-only FAA agreement), C (confound coupling: age,
arousal proxies, EMG-band power), D (severity correlation + partial
correlation controlling for the aperiodic component -- naturally degrades to
"skipped" for a severity-less dataset like Mumtaz/HUSM, not a crash).

Usage: python scripts/run_spectral_composition.py <data_dir> <manifest.csv>
       --out outputs/<dataset> --condition ec [--line-freq 60.0]
       [--participants-tsv data/<dataset>/participants.tsv]

Needs `pip install fooof==1.1.1` -- not added to neuroqa/requirements.txt
(the deployed Vercel webapp's dependency set, size-budgeted in
ARCHITECTURE.md) since this tool isn't wired into webapp.py's API routes,
same as the other research-only scripts in this directory.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "neuroqa"))

from manifest import parse_manifest_csv, validate_batch  # noqa: E402
from study_a import LINE_FREQ, load_referenced_raw, make_epochs  # noqa: E402
from spectral_composition import (  # noqa: E402
    ALPHA_BAND, APERIODIC_MODES, EMG_BAND, alpha_power_trend, band_power, fit_fooof,
    faa_variants, quality_weighted_psd, theta_alpha_ratio,
)

REFERENCES = ("original", "average")


def load_age_lookup(participants_tsv: Path | None) -> dict[str, float]:
    """participant_id -> age, from a dataset's own participants.tsv. Returns
    {} (not an error) when the path is None/missing -- some datasets in this
    spec genuinely ship no age data at all (Mumtaz/HUSM; see spec Sec. 2's
    per-dataset table), and Analysis C must degrade gracefully for those,
    not crash or fabricate a value."""
    if participants_tsv is None or not participants_tsv.exists():
        return {}
    rows = list(csv.DictReader(participants_tsv.read_text().splitlines(), delimiter="\t"))
    out = {}
    for row in rows:
        pid = row.get("participant_id")
        age = row.get("age") or row.get("Age")
        if not pid or age in (None, "", "n/a", "N/A", "NaN"):
            continue
        try:
            out[pid] = float(age)
        except ValueError:
            continue
    return out


def subject_id_from_filename(filename: str) -> str | None:
    m = re.match(r"(sub-[A-Za-z0-9]+)", filename)
    return m.group(1) if m else None


def row_for_recording(path: Path, filename: str, group: str, severity, condition: str, line_freq: float,
                       age_lookup: dict[str, float]) -> list[dict]:
    """One row per (reference, aperiodic_mode) for this recording."""
    rows = []
    subject_id = subject_id_from_filename(filename)
    age = age_lookup.get(subject_id) if subject_id else None
    for reference in REFERENCES:
        raw, ch_names = load_referenced_raw(str(path), reference, line_freq=line_freq)
        epochs = make_epochs(raw)
        data_uv = epochs.get_data() * 1e6
        if "F3" not in ch_names or "F4" not in ch_names:
            continue
        freqs, psd = quality_weighted_psd(data_uv, ch_names, raw.info["sfreq"])
        i3, i4 = ch_names.index("F3"), ch_names.index("F4")
        trend = alpha_power_trend(data_uv, ch_names, raw.info["sfreq"])
        for mode in APERIODIC_MODES:
            f3 = fit_fooof(freqs, psd[i3], aperiodic_mode=mode)
            f4 = fit_fooof(freqs, psd[i4], aperiodic_mode=mode)
            f3["classic_band_power"] = band_power(freqs, psd[i3], ALPHA_BAND)
            f4["classic_band_power"] = band_power(freqs, psd[i4], ALPHA_BAND)
            f3["theta_alpha_ratio"] = theta_alpha_ratio(freqs, psd[i3])
            f4["theta_alpha_ratio"] = theta_alpha_ratio(freqs, psd[i4])
            f3["emg_band_power"] = band_power(freqs, psd[i3], EMG_BAND)
            f4["emg_band_power"] = band_power(freqs, psd[i4], EMG_BAND)
            f3["alpha_power_trend"] = trend.get("F3", float("nan"))
            f4["alpha_power_trend"] = trend.get("F4", float("nan"))
            variants = faa_variants(f3, f4)
            row = {
                "file": filename, "group": group, "clinical_severity": severity, "age": age,
                "condition": condition, "reference": reference, "aperiodic_mode": mode,
                "n_epochs": int(data_uv.shape[0]),
            }
            for ch, params in (("f3", f3), ("f4", f4)):
                for k, v in params.items():
                    row[f"{ch}_{k}"] = v
            row.update(variants)
            rows.append(row)
    return rows


def analysis_a(rows: list[dict]) -> dict:
    """Composition: distribution of oscillatory_share, per condition x
    reference x aperiodic_mode (collapsing across subjects), per channel."""
    out = {}
    for condition in sorted({r["condition"] for r in rows}):
        for reference in REFERENCES:
            for mode in APERIODIC_MODES:
                subset = [r for r in rows if r["condition"] == condition and r["reference"] == reference
                          and r["aperiodic_mode"] == mode]
                if not subset:
                    continue
                key = f"{condition}_{reference}_{mode}"
                out[key] = {}
                for ch in ("f3", "f4"):
                    # Low-quality fits (spec Sec. 5: "flag ... and report how
                    # many were excluded and why") are excluded from the
                    # composition distribution -- a bad aperiodic+peak
                    # decomposition isn't trustworthy data, unlike a
                    # legitimate no-peak result, which stays IN (share ~ 0
                    # is data, not missingness -- but that's a different rule
                    # from fit quality).
                    good = [r for r in subset if not r[f"{ch}_low_quality_fit"]]
                    peak_rate = float(np.mean([r[f"{ch}_peak_present"] for r in good])) if good else float("nan")
                    shares = np.array([r[f"{ch}_oscillatory_share"] for r in good])
                    all_r2 = np.array([r[f"{ch}_r_squared"] for r in subset])
                    out[key][ch] = {
                        "n_total": len(subset), "n_low_quality_fit_excluded": len(subset) - len(good),
                        # Full r_squared distribution (not just the pass/fail
                        # count against MIN_R_SQUARED) -- so a reader can see
                        # the whole picture, not just a threshold-filtered
                        # summary. See spectral_composition.py's MIN_R_SQUARED
                        # docstring: today's placeholder WEIGHT/decay function
                        # may be under-suppressing contamination for a
                        # broadband fit, which would show up here as a low
                        # r_squared distribution even after quality weighting.
                        "r_squared_median": float(np.median(all_r2)),
                        "r_squared_mean": float(all_r2.mean()),
                        "r_squared_p25": float(np.percentile(all_r2, 25)),
                        "r_squared_p75": float(np.percentile(all_r2, 75)),
                        "peak_present_rate": peak_rate,
                        "oscillatory_share_median": float(np.median(shares)) if len(shares) else float("nan"),
                        "oscillatory_share_mean": float(shares.mean()) if len(shares) else float("nan"),
                        "oscillatory_share_p25": float(np.percentile(shares, 25)) if len(shares) else float("nan"),
                        "oscillatory_share_p75": float(np.percentile(shares, 75)) if len(shares) else float("nan"),
                    }
    return out


def analysis_b(rows: list[dict]) -> dict:
    """Do classic and periodic-only FAA agree? Pearson r between classic_faa
    and periodic_only_faa across subjects, per condition x reference x mode.
    Two separate, disclosed exclusion reasons (never silently pooled or
    imputed): no peak at one/both channels (periodic_only_faa is NaN by
    construction -- spec Sec. 5), and low fit quality at either channel
    (spec Sec. 5's "flag ... and report" rule -- an untrustworthy
    decomposition, distinct from a legitimate no-peak result)."""
    out = {}
    for condition in sorted({r["condition"] for r in rows}):
        for reference in REFERENCES:
            for mode in APERIODIC_MODES:
                subset = [r for r in rows if r["condition"] == condition and r["reference"] == reference
                          and r["aperiodic_mode"] == mode]
                if not subset:
                    continue
                key = f"{condition}_{reference}_{mode}"
                classic = np.array([r["classic_faa"] for r in subset])
                periodic = np.array([r["periodic_only_faa"] for r in subset])
                low_quality = np.array([r["f3_low_quality_fit"] or r["f4_low_quality_fit"] for r in subset])
                no_peak = np.isnan(periodic)
                valid = ~no_peak & ~low_quality
                n_valid, n_total = int(valid.sum()), len(subset)
                if n_valid >= 3:
                    r_value = float(np.corrcoef(classic[valid], periodic[valid])[0, 1])
                else:
                    r_value = float("nan")
                out[key] = {
                    "n_total": n_total, "n_valid": n_valid,
                    "n_dropped_no_peak": int((no_peak & ~low_quality).sum()),
                    "n_dropped_low_quality_fit": int(low_quality.sum()),
                    "pearson_r_classic_vs_periodic": r_value,
                }
    return out


CONFOUND_MEASURES = ("aperiodic_asymmetry_exponent", "aperiodic_asymmetry_offset", "oscillatory_share")
CONFOUNDS = ("age", "theta_alpha_ratio", "alpha_power_trend", "emg_band_power")
MIN_N_FOR_CORRELATION = 5  # a defensible floor, not tuned to any dataset's result -- see
# study_b.py's MIN_N_FOR_OLS for the same reasoning applied to a different tool


def _mean_f3_f4(rows: list[dict], field: str) -> np.ndarray:
    return np.array([(r[f"f3_{field}"] + r[f"f4_{field}"]) / 2.0 for r in rows])


def _safe_corr(x: np.ndarray, y: np.ndarray) -> dict:
    valid = ~np.isnan(x) & ~np.isnan(y)
    n = int(valid.sum())
    if n >= MIN_N_FOR_CORRELATION and np.std(x[valid]) > 0 and np.std(y[valid]) > 0:
        r_value = float(np.corrcoef(x[valid], y[valid])[0, 1])
    else:
        r_value = float("nan")
    return {"n": n, "pearson_r": r_value}


def analysis_c(rows: list[dict]) -> dict:
    """Confound coupling (spec Sec. 4.C, "the unifying test"): correlate
    aperiodic measures (aperiodic asymmetry in exponent/offset, and
    oscillatory_share averaged across F3/F4 into one per-subject composition
    score) against age, two arousal proxies (theta/alpha ratio, decline in
    alpha power across the recording), and residual EMG-band (20-45 Hz)
    power -- both proxies and EMG power themselves averaged across F3/F4.
    Only rows with a good fit (not low_quality_fit) at BOTH channels are
    used; age-less datasets (Mumtaz/HUSM) simply get n=0 for that one
    confound, not a crash or a fabricated value."""
    out = {}
    for condition in sorted({r["condition"] for r in rows}):
        for reference in REFERENCES:
            for mode in APERIODIC_MODES:
                subset = [r for r in rows if r["condition"] == condition and r["reference"] == reference
                          and r["aperiodic_mode"] == mode]
                if not subset:
                    continue
                good = [r for r in subset if not r["f3_low_quality_fit"] and not r["f4_low_quality_fit"]]
                key = f"{condition}_{reference}_{mode}"
                measures = {
                    "aperiodic_asymmetry_exponent": np.array([r["aperiodic_asymmetry_exponent"] for r in good]),
                    "aperiodic_asymmetry_offset": np.array([r["aperiodic_asymmetry_offset"] for r in good]),
                    "oscillatory_share": _mean_f3_f4(good, "oscillatory_share"),
                }
                confounds = {
                    "age": np.array([r["age"] if r["age"] is not None else float("nan") for r in good]),
                    "theta_alpha_ratio": _mean_f3_f4(good, "theta_alpha_ratio"),
                    "alpha_power_trend": _mean_f3_f4(good, "alpha_power_trend"),
                    "emg_band_power": _mean_f3_f4(good, "emg_band_power"),
                }
                out[key] = {"n_good_fit": len(good), "n_total": len(subset), "correlations": {
                    m_name: {c_name: _safe_corr(m_vals, c_vals) for c_name, c_vals in confounds.items()}
                    for m_name, m_vals in measures.items()
                }}
    return out


def analysis_d(rows: list[dict]) -> dict:
    """Severity analysis (spec Sec. 4.D, Arizona/Oslo only -- degrades to
    n=0/skipped for a severity-less dataset like Mumtaz, not a crash):
    (1) simple correlations of the aperiodic measures with clinical severity,
    (2) whether classic FAA's relationship to severity survives controlling
    for the aperiodic component (OLS: classic_faa ~ severity + aperiodic
    measures; the severity coefficient IS the partial-correlation answer)."""
    import statsmodels.api as sm

    out = {}
    for condition in sorted({r["condition"] for r in rows}):
        for reference in REFERENCES:
            for mode in APERIODIC_MODES:
                subset = [r for r in rows if r["condition"] == condition and r["reference"] == reference
                          and r["aperiodic_mode"] == mode]
                if not subset:
                    continue
                key = f"{condition}_{reference}_{mode}"
                good = [r for r in subset if not r["f3_low_quality_fit"] and not r["f4_low_quality_fit"]
                        and r["clinical_severity"] not in (None, "")]
                if len(good) < MIN_N_FOR_CORRELATION + 1:  # +1: OLS below fits 5 params
                    out[key] = {"n": len(good),
                                "skipped_reason": f"fewer than {MIN_N_FOR_CORRELATION + 1} usable rows with severity"}
                    continue
                severity = np.array([float(r["clinical_severity"]) for r in good])
                classic = np.array([r["classic_faa"] for r in good])
                osc_share = _mean_f3_f4(good, "oscillatory_share")
                aae = np.array([r["aperiodic_asymmetry_exponent"] for r in good])
                aao = np.array([r["aperiodic_asymmetry_offset"] for r in good])

                simple_corr = {
                    "oscillatory_share": _safe_corr(osc_share, severity),
                    "aperiodic_asymmetry_exponent": _safe_corr(aae, severity),
                    "aperiodic_asymmetry_offset": _safe_corr(aao, severity),
                }

                X = sm.add_constant(np.column_stack([severity, aae, aao, osc_share]))
                model = sm.OLS(classic, X).fit()
                out[key] = {
                    "n": len(good),
                    "aperiodic_vs_severity_simple_correlations": simple_corr,
                    "classic_faa_vs_severity_simple_r": _safe_corr(classic, severity),
                    "classic_faa_vs_severity_controlling_for_aperiodic": {
                        "severity_coef": float(model.params[1]),
                        "severity_coef_pvalue": float(model.pvalues[1]),
                        "r_squared": float(model.rsquared),
                        "nobs": int(model.nobs),
                    },
                }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("manifest_csv", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--condition", default="ec", help="eyes-closed/eyes-open label for this batch (spec: never pool)")
    parser.add_argument("--line-freq", type=float, default=None)
    parser.add_argument("--participants-tsv", type=Path, default=None,
                         help="dataset's participants.tsv for age (Analysis C) -- omit for datasets with no age data (e.g. Mumtaz/HUSM)")
    args = parser.parse_args()

    line_freq = args.line_freq if args.line_freq is not None else LINE_FREQ
    age_lookup = load_age_lookup(args.participants_tsv)
    print(f"age lookup: {len(age_lookup)} subjects" if age_lookup else "age lookup: none (Analysis C's age confound will show n=0)")
    manifest_rows = parse_manifest_csv(args.manifest_csv.read_text())
    paths = sorted(p for p in args.data_dir.iterdir() if p.is_file())
    batch = validate_batch(paths, manifest_rows)
    print(f"accepted: {len(batch.accepted)} rejected: {len(batch.rejected)}")
    path_by_name = {p.name: p for p in paths}

    all_rows = []
    for i, r in enumerate(batch.accepted):
        print(f"  [{i+1}/{len(batch.accepted)}] {r.filename} ...")
        try:
            all_rows.extend(row_for_recording(
                path_by_name[r.filename], r.filename, r.diagnosis, r.severity, args.condition, line_freq, age_lookup))
        except Exception as e:
            print(f"    [ERROR] {r.filename}: {e}")

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / f"spectral_composition_{args.condition}.csv"
    if all_rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
    print(f"wrote {csv_path} ({len(all_rows)} rows)")

    summary = {"n_recordings": len(batch.accepted), "n_rows": len(all_rows),
               "analysis_a_composition": analysis_a(all_rows),
               "analysis_b_classic_vs_periodic": analysis_b(all_rows),
               "analysis_c_confound_coupling": analysis_c(all_rows),
               "analysis_d_severity": analysis_d(all_rows)}
    summary_path = args.out / f"spectral_composition_summary_{args.condition}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
