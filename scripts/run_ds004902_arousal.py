"""Neureidos spec's optional arousal extension (Sec. 2): does a REAL,
controlled arousal shift (sleep deprivation, matched within-subject) move
the aperiodic component -- upgrading the arousal question from the other
three datasets' proxy correlations (Analysis C: theta/alpha ratio, alpha-
power trend) to a genuine within-subject comparison.

For each subject with both a normal-sleep (NS) and sleep-deprivation (SD)
eyes-closed recording (see download_ds004902.py for how NS/SD is resolved
per subject), computes the same spectral-composition measures
(spectral_composition.py) for both sessions and reports the paired
NS->SD difference: mean, bootstrap 95% CI, paired t-test -- per aperiodic
measure, per channel (F3/F4) and averaged across them.

Usage: python scripts/run_ds004902_arousal.py [--out outputs/ds004902] [--line-freq 50.0]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "neuroqa"))

from preprocess import LINE_FREQ, preprocess_file  # noqa: E402
from spectral_composition import fit_fooof, quality_weighted_psd  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "ds004902"
CHANNELS = ("F3", "F4")
MEASURES = ("exponent", "offset", "oscillatory_share")
BOOTSTRAP_N = 2000
SEED = 0


def measures_for_recording(path: Path, line_freq: float) -> dict[str, dict] | None:
    data_uv, ch_names, sfreq = preprocess_file(path, line_freq=line_freq)
    if "F3" not in ch_names or "F4" not in ch_names:
        return None
    freqs, psd = quality_weighted_psd(data_uv, ch_names, sfreq)
    out = {}
    for ch in CHANNELS:
        i = ch_names.index(ch)
        out[ch] = fit_fooof(freqs, psd[i], aperiodic_mode="fixed")
    return out


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = BOOTSTRAP_N, seed: int = SEED) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    idx = np.arange(len(values))
    boots = [values[rng.choice(idx, size=len(idx), replace=True)].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "ds004902")
    parser.add_argument("--line-freq", type=float, default=None)
    args = parser.parse_args()
    line_freq = args.line_freq if args.line_freq is not None else LINE_FREQ

    manifest_path = DATA_DIR / "manifest.csv"
    rows = list(csv.DictReader(manifest_path.read_text().splitlines()))
    by_subject: dict[str, dict[str, str]] = {}
    for r in rows:
        by_subject.setdefault(r["subject_id"], {})[r["session_condition"]] = r["filename"]

    paired_subjects = {s: files for s, files in by_subject.items() if "NS" in files and "SD" in files}
    print(f"{len(paired_subjects)}/{len(by_subject)} subjects have both NS and SD recordings")

    per_subject_rows = []
    for i, (subj, files) in enumerate(sorted(paired_subjects.items()), 1):
        print(f"  [{i}/{len(paired_subjects)}] {subj} ...")
        try:
            ns = measures_for_recording(DATA_DIR / files["NS"], line_freq)
            sd = measures_for_recording(DATA_DIR / files["SD"], line_freq)
        except Exception as e:
            print(f"    [ERROR] {subj}: {e}")
            continue
        if ns is None or sd is None:
            continue
        row = {"subject_id": subj}
        for ch in CHANNELS:
            for m in MEASURES:
                row[f"ns_{ch}_{m}"] = ns[ch][m]
                row[f"sd_{ch}_{m}"] = sd[ch][m]
            row[f"ns_{ch}_low_quality_fit"] = ns[ch]["low_quality_fit"]
            row[f"sd_{ch}_low_quality_fit"] = sd[ch]["low_quality_fit"]
        per_subject_rows.append(row)

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "ds004902_arousal.csv"
    if per_subject_rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_subject_rows[0].keys()))
            w.writeheader()
            w.writerows(per_subject_rows)
    print(f"wrote {csv_path} ({len(per_subject_rows)} rows)")

    # Paired NS->SD comparison, per channel and averaged across F3/F4,
    # excluding a subject from a given measure/channel if either session's
    # fit was low_quality_fit (same rule as the other 3 datasets' Analysis
    # A/B -- an untrustworthy decomposition shouldn't feed a paired test).
    summary = {}
    for ch in CHANNELS + ("mean_f3_f4",):
        summary[ch] = {}
        for m in MEASURES:
            if ch == "mean_f3_f4":
                good = [r for r in per_subject_rows
                        if not r["ns_F3_low_quality_fit"] and not r["sd_F3_low_quality_fit"]
                        and not r["ns_F4_low_quality_fit"] and not r["sd_F4_low_quality_fit"]]
                ns_vals = np.array([(r[f"ns_F3_{m}"] + r[f"ns_F4_{m}"]) / 2 for r in good])
                sd_vals = np.array([(r[f"sd_F3_{m}"] + r[f"sd_F4_{m}"]) / 2 for r in good])
            else:
                good = [r for r in per_subject_rows if not r[f"ns_{ch}_low_quality_fit"] and not r[f"sd_{ch}_low_quality_fit"]]
                ns_vals = np.array([r[f"ns_{ch}_{m}"] for r in good])
                sd_vals = np.array([r[f"sd_{ch}_{m}"] for r in good])
            diff = sd_vals - ns_vals
            valid = ~np.isnan(diff)
            diff = diff[valid]
            if len(diff) >= 3:
                lo, hi = bootstrap_mean_ci(diff)
                t_stat, p_value = stats.ttest_rel(sd_vals[valid], ns_vals[valid])
                summary[ch][m] = {
                    "n": int(len(diff)), "mean_sd_minus_ns": float(diff.mean()),
                    "ci_lo": lo, "ci_hi": hi, "nonzero": bool(not (lo <= 0.0 <= hi)),
                    "paired_ttest_p": float(p_value),
                }
            else:
                summary[ch][m] = {"n": int(len(diff)), "skipped_reason": "fewer than 3 usable paired subjects"}

    summary_path = args.out / "ds004902_arousal_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {summary_path}")
    for ch, measures in summary.items():
        for m, s in measures.items():
            if "mean_sd_minus_ns" in s:
                print(f"  {ch:12s} {m:20s} n={s['n']:3d} SD-NS={s['mean_sd_minus_ns']:+.4f} "
                      f"CI=[{s['ci_lo']:+.4f},{s['ci_hi']:+.4f}] p={s['paired_ttest_p']:.3f} nonzero={s['nonzero']}")


if __name__ == "__main__":
    main()
