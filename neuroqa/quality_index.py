"""NeuroQA — the endpoint-aware quality index.

This is the actual invention described in the project brief: not a
classifier, not a generic "fraction of clean epochs" score, but a penalty
that depends on *which frequency band you're about to measure*. A blink
detected by artifact_detectors.detect_eog is a large penalty if you're
about to report delta power and a near-zero penalty if you're about to
report alpha power (FAA), because a blink's energy doesn't live in alpha.

    for each detected_artifact in recording:
        overlap = spectral_overlap(artifact.band, ENDPOINT_BAND)
        penalty += artifact.severity * WEIGHT[artifact.type] * overlap
    quality = f(penalty)

Mapped onto this codebase: "detected_artifact.severity" is the Step 3
detector's per-epoch-per-channel score in [0, 1] (artifact_detectors.py),
"artifact.type" is the detector name, "artifact.band" and "WEIGHT" come from
bands.py. This module is pure computation (no I/O) so causal_quality.py and
the study scripts can reuse the exact same penalty logic offline and online.
"""

from __future__ import annotations

import numpy as np

from artifact_detectors import DETECTORS, run_all
from bands import ARTIFACT_BANDS, WEIGHT, spectral_overlap

Band = tuple[float, float]


def detector_overlaps(endpoint_band: Band) -> dict[str, float]:
    """Precompute each detector's fixed spectral_overlap against one endpoint
    band -- a scalar per detector, since a detector's artifact band doesn't
    change epoch to epoch, only which endpoint we're scoring against does."""
    return {name: spectral_overlap(ARTIFACT_BANDS[name], endpoint_band) for name in DETECTORS}


def contributions(detector_scores: dict[str, np.ndarray], endpoint_band: Band) -> dict[str, np.ndarray]:
    """Per-detector penalty contribution: severity * WEIGHT[type] * overlap.

    Each value is (n_epochs, n_channels), same shape as the detector scores.
    Kept separate (not pre-summed) so callers can report an artifact-type
    breakdown of the penalty, not just the total.
    """
    overlap = detector_overlaps(endpoint_band)
    return {
        name: scores * WEIGHT[name] * overlap[name]
        for name, scores in detector_scores.items()
    }


def penalty_from_contributions(contribs: dict[str, np.ndarray]) -> np.ndarray:
    total = None
    for arr in contribs.values():
        total = arr.copy() if total is None else total + arr
    return total


def quality_from_penalty(penalty: np.ndarray) -> np.ndarray:
    """f(penalty) -> quality in (0, 100].

    Exponential decay: quality=100 at zero penalty, monotonically decreasing,
    never negative or discontinuous. The decay rate (implicitly 1 penalty-
    unit per e-fold) is a placeholder functional form, same status as WEIGHT
    in bands.py -- pending Peter's preferred f(), not fit to any accuracy
    number here.
    """
    return 100.0 * np.exp(-penalty)


def compute_quality(
    data: np.ndarray,
    ch_names: list[str],
    sfreq: float,
    endpoint_band: Band,
    detector_scores: dict[str, np.ndarray] | None = None,
) -> dict:
    """Full endpoint-aware quality computation for one recording's epoched data.

    Returns dict with:
      detector_scores : {name: (n_epochs, n_channels)} raw Step 3 severities
      contributions    : {name: (n_epochs, n_channels)} weighted, overlap-scaled
      penalty          : (n_epochs, n_channels)
      quality          : (n_epochs, n_channels), in (0, 100]
    """
    if detector_scores is None:
        detector_scores = run_all(data, ch_names, sfreq)
    contribs = contributions(detector_scores, endpoint_band)
    penalty = penalty_from_contributions(contribs)
    quality = quality_from_penalty(penalty)
    return {
        "detector_scores": detector_scores,
        "contributions": contribs,
        "penalty": penalty,
        "quality": quality,
    }
