"""NeuroQA Step 4 — endpoint-aware scoring and output.

Runs the Step 3 detectors once per recording, then scores the result with
quality_index.py's endpoint-aware penalty for every canonical EEG band. The
headline deliverable is not one quality number per recording -- it's that
the number moves depending on which band you're about to measure (see
quality_index.py's docstring for why). This script reports one quality_pct
column per band so that's directly visible in the output CSV, plus a
per-channel detail table and an artifact-type penalty breakdown for the
alpha band specifically (the endpoint Study A/B and faa.py care about, since
FAA is an alpha-band measurement).

Usage (from repo root, after ingest.py and preprocess.py):
    .venv/Scripts/python.exe neuroqa/score.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from artifact_detectors import DETECTORS, run_all
from bands import EEG_BANDS
from quality_index import compute_quality, contributions

# pandas is only needed by main()'s manifest/summary CSV I/O, not by
# score_recording() or grade_from_pct() (which analyze.py imports) --
# imported lazily in main() so the webapp doesn't drag pandas in.

OUT_DIR = Path(__file__).resolve().parent / "outputs"
EPOCH_DIR = OUT_DIR / "epochs"

PRIMARY_ENDPOINT = "alpha"  # what channel detail / artifact breakdown report on
GRADE_BINS = [(90, "A"), (80, "B"), (70, "C"), (60, "D"), (0, "F")]


def grade_from_pct(pct: float) -> str:
    for cutoff, letter in GRADE_BINS:
        if pct >= cutoff:
            return letter
    return "F"  # unreachable, last bin is (0, "F")


def score_recording(npz_path: Path) -> tuple[dict, list[dict]]:
    d = np.load(npz_path, allow_pickle=True)
    data, ch_names, sfreq = d["data"], list(d["ch_names"]), float(d["sfreq"])
    n_epochs, n_channels, _ = data.shape

    # Detector severities don't depend on the endpoint band, so compute once
    # and reuse across every band -- only the overlap weighting changes.
    detector_scores = run_all(data, ch_names, sfreq)

    row = {"n_epochs": n_epochs, "n_channels": n_channels}
    per_band_quality = {}
    for band_name, band in EEG_BANDS.items():
        result = compute_quality(data, ch_names, sfreq, band, detector_scores=detector_scores)
        per_band_quality[band_name] = result["quality"]
        row[f"quality_{band_name}_pct"] = round(float(result["quality"].mean()), 2)

    primary_quality = per_band_quality[PRIMARY_ENDPOINT]  # (n_epochs, n_channels)
    channel_quality_pct = primary_quality.mean(axis=0)  # (n_channels,)
    worst_idx = int(np.argmin(channel_quality_pct))
    row["grade"] = grade_from_pct(row[f"quality_{PRIMARY_ENDPOINT}_pct"])
    row["worst_channel"] = ch_names[worst_idx]
    row["worst_channel_quality_pct"] = round(float(channel_quality_pct[worst_idx]), 2)

    # Endpoint-independent "how much raw artifact content is in this
    # recording" -- mean detector severity across all 6 detectors, with no
    # WEIGHT or spectral_overlap applied. Used as the `severity` covariate in
    # study_b.py's quality~group+severity regression, kept separate from the
    # (endpoint-dependent) quality_*_pct columns above.
    row["raw_severity_mean"] = round(float(np.mean([s.mean() for s in detector_scores.values()])), 4)

    # Artifact-type penalty breakdown, alpha endpoint: how much of the
    # penalty each detector type is responsible for (mean over epoch-channel).
    primary_contribs = contributions(detector_scores, EEG_BANDS[PRIMARY_ENDPOINT])
    for name in DETECTORS:
        row[f"{name}_penalty_contrib"] = round(float(primary_contribs[name].mean()), 4)

    channel_detail = [
        {"channel": ch, "quality_pct": round(float(q), 2)}
        for ch, q in zip(ch_names, channel_quality_pct)
    ]
    return row, channel_detail


def main():
    import pandas as pd

    manifest = pd.read_csv(OUT_DIR / "ingest_manifest.csv")
    manifest = manifest[manifest.load_error.isna()]

    summary_rows = []
    channel_rows = []
    for _, r in manifest.iterrows():
        npz_path = EPOCH_DIR / (Path(r["file"]).stem + ".npz")
        if not npz_path.exists():
            print(f"  [skip] no preprocessed epochs for {r['file']}")
            continue
        row, channel_detail = score_recording(npz_path)
        row = {
            "file": r["file"], "group": r["group"], "subject": r["subject"],
            "condition": r["condition"], "is_duplicate": bool(r["is_duplicate"]),
            **row,
        }
        summary_rows.append(row)
        for c in channel_detail:
            channel_rows.append({"file": r["file"], **c})
        print(f"  {r['file']:28s} grade={row['grade']}  "
              f"quality[{PRIMARY_ENDPOINT}]={row[f'quality_{PRIMARY_ENDPOINT}_pct']:5.1f}%  "
              f"quality[delta]={row['quality_delta_pct']:5.1f}%  "
              f"worst_ch={row['worst_channel']}({row['worst_channel_quality_pct']:.0f}%)")

    summary = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / "quality_summary.csv"
    summary.to_csv(summary_path, index=False)

    detail = pd.DataFrame(channel_rows)
    detail_path = OUT_DIR / "channel_quality_detail.csv"
    detail.to_csv(detail_path, index=False)

    print(f"\nscored {len(summary)} recordings")
    print("grade distribution (alpha endpoint):")
    print(summary.grade.value_counts().sort_index().to_string())
    print("\nmean quality_pct by endpoint band (shows the score moving with the endpoint):")
    for band_name in EEG_BANDS:
        print(f"  {band_name:6s} {summary[f'quality_{band_name}_pct'].mean():5.1f}%")
    print(f"\nwrote {summary_path}  (the deliverable)")
    print(f"wrote {detail_path}  (per-channel drill-down, alpha endpoint)")


if __name__ == "__main__":
    main()
