"""NeuroQA Step 4 — endpoint-aware scoring.

Runs the Step 3 detectors once per recording, then scores the result with
quality_index.py's endpoint-aware penalty for every canonical EEG band. The
headline deliverable is not one quality number per recording -- it's that
the number moves depending on which band you're about to measure (see
quality_index.py's docstring for why). Reports one quality_pct column per
band, plus a per-channel detail table and an artifact-type penalty
breakdown for the alpha band specifically (the endpoint Study A/B and
faa.py care about, since FAA is an alpha-band measurement).

score_recording() takes already-epoched (data, ch_names, sfreq) in memory --
callers (pipeline.py for the upload-driven flow, analyze.py for the
single-file webapp) are responsible for getting a recording into that shape
via preprocess.py, this module doesn't do any file/manifest I/O itself.
"""

from __future__ import annotations

import numpy as np

from artifact_detectors import DETECTORS, run_all
from bands import EEG_BANDS
from quality_index import compute_quality, contributions

PRIMARY_ENDPOINT = "alpha"  # what channel detail / artifact breakdown report on
GRADE_BINS = [(90, "A"), (80, "B"), (70, "C"), (60, "D"), (0, "F")]


def grade_from_pct(pct: float) -> str:
    for cutoff, letter in GRADE_BINS:
        if pct >= cutoff:
            return letter
    return "F"  # unreachable, last bin is (0, "F")


def score_recording(data: np.ndarray, ch_names: list[str], sfreq: float) -> tuple[dict, list[dict]]:
    """Score one recording's epoched data across every canonical EEG band.

    data: (n_epochs, n_channels, n_samples) in microvolts (see preprocess.py).
    Returns (row, channel_detail): `row` is a flat dict of summary numbers
    (one quality_<band>_pct per band, grade, worst channel, raw_severity_mean,
    per-detector alpha-endpoint penalty share); `channel_detail` is a list of
    {channel, quality_pct} for the primary (alpha) endpoint.
    """
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
    # WEIGHT or spectral_overlap applied. A data-quality diagnostic, kept
    # separate from the (endpoint-dependent) quality_*_pct columns above --
    # NOT the same thing as clinical severity (BDI/HAM-D, from the upload
    # manifest), which is what study_b.py's quality~group+severity
    # regression actually uses. Conflating the two would make that
    # regression circular, since quality is itself derived from this value.
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
