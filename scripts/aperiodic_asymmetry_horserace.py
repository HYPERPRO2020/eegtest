"""The pivot: a quality-weighted frontal aperiodic ASYMMETRY index vs.
classic band-power FAA, head-to-head, on continuous clinical severity.

Per the literature check (2026-09-06): the plain "does absolute aperiodic
level predict depression" question has already been directly tested by at
least four groups since 2023 with real BDI/BDI-II scores, and they don't
even agree on the sign. Two things survived a thorough search as genuinely
open: (1) a frontal aperiodic exponent/offset ASYMMETRY index (F4 minus
F3, built the same way classic FAA is, but on the aperiodic parameters
instead of band power) tested against depression severity -- every
existing study uses absolute/regional levels, none computes a laterality
contrast; (2) contamination-quality-weighted extraction applied to a
clinical aperiodic question -- a Jan 2026 methods paper shows ocular/
muscle artifacts bias the aperiodic exponent by margins as large as the
clinical effects reported in the depression literature, and no existing
depression-aperiodic study controls for it. Our pipeline already does (2)
for every number in this file (quality_weighted_psd, see
spectral_composition.py); this script adds the head-to-head test for (1).

Both aperiodic_asymmetry_exponent/_offset (Step 3.3 of the original spec)
and clinical_severity are already columns in spectral_composition_*.csv --
this reads that existing output directly, no new data collection or
FOOOF fitting needed.

Three nested OLS models per dataset/condition:
  M1: severity ~ classic_faa                                  (classic alone)
  M2: severity ~ aperiodic_asymmetry_exponent + _offset        (aperiodic alone)
  M3: severity ~ classic_faa + aperiodic_asymmetry_exponent + _offset  (both)
Nested F-tests (M3 vs M1, M3 vs M2) answer the actual head-to-head
question: does each add anything beyond what the other already explains?
No directional hypothesis going in -- reported whichever way it comes out.

Usage: python scripts/aperiodic_asymmetry_horserace.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "outputs"

COMBOS = [
    {"key": "ds003478", "condition": "ec"},
    {"key": "ds007615", "condition": "ec"},
    {"key": "ds007615", "condition": "eo"},
]
MIN_N = 8  # M3 fits 4 params (const + 3 predictors); floor gives >=4 residual df,
# same "disclosed default, not tuned to this data" reasoning as study_b.MIN_N_FOR_OLS


def load_rows(key: str, condition: str) -> pd.DataFrame | None:
    path = OUT_DIR / key / f"spectral_composition_{condition}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[(df["reference"] == "original") & (df["aperiodic_mode"] == "fixed")]
    df["clinical_severity"] = pd.to_numeric(df["clinical_severity"], errors="coerce")
    df = df.dropna(subset=["clinical_severity", "classic_faa", "aperiodic_asymmetry_exponent",
                            "aperiodic_asymmetry_offset"])
    df = df[~df["f3_low_quality_fit"] & ~df["f4_low_quality_fit"]]
    return df.reset_index(drop=True)


def nested_f_test(rss_reduced: float, df_reduced: int, rss_full: float, df_full: int) -> tuple[float, float]:
    """F-test for whether the full model's extra predictor(s) explain
    significantly more residual variance than the reduced model."""
    df_diff = df_reduced - df_full
    if df_diff <= 0 or df_full <= 0 or rss_full <= 0:
        return float("nan"), float("nan")
    f_stat = ((rss_reduced - rss_full) / df_diff) / (rss_full / df_full)
    if f_stat < 0:
        f_stat = 0.0
    p_value = float(1 - stats.f.cdf(f_stat, df_diff, df_full))
    return float(f_stat), p_value


def run_horserace(df: pd.DataFrame) -> dict:
    n = len(df)
    if n < MIN_N:
        return {"n": n, "skipped_reason": f"fewer than {MIN_N} usable rows with severity and good fits"}

    y = df["clinical_severity"].values
    classic = df[["classic_faa"]].values
    aperiodic = df[["aperiodic_asymmetry_exponent", "aperiodic_asymmetry_offset"]].values
    both = df[["classic_faa", "aperiodic_asymmetry_exponent", "aperiodic_asymmetry_offset"]].values

    m1 = sm.OLS(y, sm.add_constant(classic)).fit()
    m2 = sm.OLS(y, sm.add_constant(aperiodic)).fit()
    m3 = sm.OLS(y, sm.add_constant(both)).fit()

    f_beyond_classic, p_beyond_classic = nested_f_test(m1.ssr, int(m1.df_resid), m3.ssr, int(m3.df_resid))
    f_beyond_aperiodic, p_beyond_aperiodic = nested_f_test(m2.ssr, int(m2.df_resid), m3.ssr, int(m3.df_resid))

    r_classic = float(np.corrcoef(df["classic_faa"], y)[0, 1])
    r_aae = float(np.corrcoef(df["aperiodic_asymmetry_exponent"], y)[0, 1])
    r_aao = float(np.corrcoef(df["aperiodic_asymmetry_offset"], y)[0, 1])

    return {
        "n": n,
        "simple_correlations_with_severity": {
            "classic_faa": r_classic,
            "aperiodic_asymmetry_exponent": r_aae,
            "aperiodic_asymmetry_offset": r_aao,
        },
        "model_classic_faa_only": {"r_squared": float(m1.rsquared), "nobs": int(m1.nobs)},
        "model_aperiodic_asymmetry_only": {"r_squared": float(m2.rsquared), "nobs": int(m2.nobs)},
        "model_both": {"r_squared": float(m3.rsquared), "nobs": int(m3.nobs)},
        "does_aperiodic_add_beyond_classic": {
            "f_stat": f_beyond_classic, "p_value": p_beyond_classic,
            "significant": bool(not np.isnan(p_beyond_classic) and p_beyond_classic < 0.05),
        },
        "does_classic_add_beyond_aperiodic": {
            "f_stat": f_beyond_aperiodic, "p_value": p_beyond_aperiodic,
            "significant": bool(not np.isnan(p_beyond_aperiodic) and p_beyond_aperiodic < 0.05),
        },
    }


def main():
    all_results = {}
    for combo in COMBOS:
        df = load_rows(combo["key"], combo["condition"])
        key = f"{combo['key']}_{combo['condition']}"
        if df is None:
            print(f"{key}: no data file, skipping")
            continue
        result = run_horserace(df)
        all_results[key] = result
        print(f"\n=== {key} ===")
        if "skipped_reason" in result:
            print(f"  SKIPPED: {result['skipped_reason']}")
            continue
        print(f"  n = {result['n']}")
        sc = result["simple_correlations_with_severity"]
        print(f"  simple r vs severity: classic_faa={sc['classic_faa']:+.3f}  "
              f"aperiodic_asymmetry_exponent={sc['aperiodic_asymmetry_exponent']:+.3f}  "
              f"aperiodic_asymmetry_offset={sc['aperiodic_asymmetry_offset']:+.3f}")
        print(f"  R^2: classic-only={result['model_classic_faa_only']['r_squared']:.3f}  "
              f"aperiodic-only={result['model_aperiodic_asymmetry_only']['r_squared']:.3f}  "
              f"both={result['model_both']['r_squared']:.3f}")
        dab = result["does_aperiodic_add_beyond_classic"]
        dca = result["does_classic_add_beyond_aperiodic"]
        print(f"  aperiodic asymmetry adds beyond classic FAA: F={dab['f_stat']:.2f} p={dab['p_value']:.3f} "
              f"significant={dab['significant']}")
        print(f"  classic FAA adds beyond aperiodic asymmetry: F={dca['f_stat']:.2f} p={dca['p_value']:.3f} "
              f"significant={dca['significant']}")

    out_path = OUT_DIR / "aperiodic_asymmetry_horserace.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
