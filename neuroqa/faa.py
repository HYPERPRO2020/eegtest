"""NeuroQA — frontal alpha asymmetry (FAA).

FAA = ln(alpha power at F4) - ln(alpha power at F3)

Two power values and a subtraction, per the brief. The only design choices
worth documenting:

  - alpha power per epoch is Welch band power (8-13 Hz, matches
    bands.EEG_BANDS["alpha"]) -- same PSD machinery artifact_detectors.py
    already uses, so this stays consistent with the rest of the pipeline.
  - aggregating to one FAA value per recording averages *power* across
    epochs first, then takes ln and subtracts -- not the other way around
    (averaging per-epoch ln-differences). ln(mean(power)) is the standard
    FAA convention; mean(ln(power_f4) - ln(power_f3)) is a different
    quantity (a log-ratio averaged in log-space) that shows up in some
    papers too, but isn't what's specified here.
  - `weights` (optional, e.g. per-epoch quality at F3/F4 from
    quality_index.py) lets a caller do quality-weighted power averaging
    instead of a flat mean -- this is what study_a.py's "ours" pipeline
    uses instead of hard-rejecting epochs.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

from bands import EEG_BANDS

F3, F4 = "F3", "F4"
ALPHA_BAND = EEG_BANDS["alpha"]


def alpha_power(data: np.ndarray, ch_names: list[str], sfreq: float) -> np.ndarray:
    """Per-epoch, per-channel alpha (8-13 Hz) band power via Welch PSD.

    data: (n_epochs, n_channels, n_samples). Returns (n_epochs, n_channels).
    """
    freqs, psd = welch(data, fs=sfreq, nperseg=min(512, data.shape[-1]), axis=-1)
    band = (freqs >= ALPHA_BAND[0]) & (freqs <= ALPHA_BAND[1])
    return psd[:, :, band].mean(axis=2)


def _weighted_mean(x: np.ndarray, weights: np.ndarray | None) -> float:
    if weights is None:
        return float(np.mean(x))
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if total <= 0:
        return float(np.mean(x))  # degenerate all-zero-weight case, fall back
    return float(np.sum(x * weights) / total)


def compute_faa(
    data: np.ndarray,
    ch_names: list[str],
    sfreq: float,
    weights_f3: np.ndarray | None = None,
    weights_f4: np.ndarray | None = None,
) -> dict:
    """Compute per-epoch and aggregate FAA for one recording.

    weights_f3 / weights_f4 (optional, each (n_epochs,)): non-negative
    per-epoch weights (e.g. quality) for the aggregate power average. If
    omitted, aggregate power is a flat mean across epochs.

    Returns dict with:
      alpha_power_f3, alpha_power_f4 : (n_epochs,) per-epoch band power
      faa_per_epoch                  : (n_epochs,) ln(f4) - ln(f3), per epoch
      faa                            : scalar, ln(mean_power_f4) - ln(mean_power_f3)
    """
    if F3 not in ch_names or F4 not in ch_names:
        raise ValueError(f"FAA requires both F3 and F4; got channels {ch_names}")
    i3, i4 = ch_names.index(F3), ch_names.index(F4)

    power = alpha_power(data, ch_names, sfreq)  # (n_epochs, n_channels)
    p3, p4 = power[:, i3], power[:, i4]

    mean_p3 = _weighted_mean(p3, weights_f3)
    mean_p4 = _weighted_mean(p4, weights_f4)

    return {
        "alpha_power_f3": p3,
        "alpha_power_f4": p4,
        "faa_per_epoch": np.log(p4 + 1e-20) - np.log(p3 + 1e-20),
        "faa": float(np.log(mean_p4 + 1e-20) - np.log(mean_p3 + 1e-20)),
    }
