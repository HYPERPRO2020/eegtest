"""NeuroQA Step 3 — per-artifact-class detectors.

STATUS: the scientist's taxonomy doc with real numeric thresholds was never
delivered, so every threshold below is a PLACEHOLDER — a defensible starting
point from standard scalp-EEG QC conventions (and the same order of magnitude
as the 150 uV epoch-rejection threshold `v1/src/features.py` already uses),
not a validated clinical cutoff. Swap `THRESHOLDS` for the scientist's numbers
the moment they exist; nothing downstream needs to change shape.

None of these recordings has a dedicated EOG or ECG channel (confirmed in
`v1/README.md` — the dataset ships no aux/stim channels), so EOG and cardiac
detection here are scalp-signal heuristics (frontal amplitude, PSD peakiness),
not reference-based. That's a structural limitation of the data, not the code.

Each detector takes epoched data shaped (n_epochs, n_channels, n_samples) and
returns a (n_epochs, n_channels) float array in [0, 1] — 0 = clean,
1 = maximally confident artifact. A per-cell boolean flag is score > 0.5 in
every detector, i.e. "past the placeholder threshold at least once over."
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

from bands import welch_nperseg

# ---- PLACEHOLDER thresholds — replace once the taxonomy doc lands ----------
THRESHOLDS = {
    "eog_ptp_uv": 100.0,          # frontal peak-to-peak amplitude, blink/saccade
    "emg_rel_power": 0.30,        # fraction of epoch power in 20-45 Hz
    "pop_diff_uv": 40.0,          # max single-sample jump (electrode pop)
    "line_ratio": 3.0,            # 49-51 Hz power vs adjacent-band power
    "motion_ptp_uv": 150.0,       # per-channel PTP counted toward a motion epoch
    "motion_channel_frac": 0.5,   # fraction of channels that must co-fire
    "cardiac_peakiness": 4.0,     # 0.8-2 Hz PSD peak vs surrounding band
}

FRONTAL_CHANNELS = {"Fp1", "Fp2", "F7", "F8"}


def _score(raw: np.ndarray, threshold: float) -> np.ndarray:
    """Linear ramp 0->1 as `raw` crosses `threshold`, clipped at 1."""
    return np.clip(raw / threshold, 0.0, 1.0)


def detect_eog(data: np.ndarray, ch_names: list[str], sfreq: float) -> np.ndarray:
    """Blink/saccade: large peak-to-peak amplitude, scored only on frontal
    channels (where ocular volume conduction is strongest)."""
    ptp = data.max(axis=2) - data.min(axis=2)  # (n_epochs, n_channels), volts -> already uV upstream
    score = _score(ptp, THRESHOLDS["eog_ptp_uv"])
    mask = np.array([ch in FRONTAL_CHANNELS for ch in ch_names])
    score = score * mask[None, :]
    return score


def detect_emg(data: np.ndarray, ch_names: list[str], sfreq: float) -> np.ndarray:
    """Muscle: elevated relative power in the 20-45 Hz band."""
    freqs, psd = welch(data, fs=sfreq, nperseg=welch_nperseg(sfreq, data.shape[-1]), axis=-1)
    hi = (freqs >= 20) & (freqs <= 45)
    rel_power = psd[:, :, hi].sum(axis=2) / (psd.sum(axis=2) + 1e-20)
    return _score(rel_power, THRESHOLDS["emg_rel_power"])


def detect_pop(data: np.ndarray, ch_names: list[str], sfreq: float) -> np.ndarray:
    """Electrode pop: an abrupt single-sample step far larger than the
    surrounding signal — a real discontinuity, not a slow deflection."""
    diffs = np.abs(np.diff(data, axis=2))
    max_diff = diffs.max(axis=2)  # (n_epochs, n_channels)
    return _score(max_diff, THRESHOLDS["pop_diff_uv"])


def detect_line_noise(data: np.ndarray, ch_names: list[str], sfreq: float) -> np.ndarray:
    """Residual line noise the notch filter didn't fully remove: power right
    at 50 Hz well above its immediate neighbors."""
    freqs, psd = welch(data, fs=sfreq, nperseg=welch_nperseg(sfreq, data.shape[-1]), axis=-1)
    line = (freqs >= 49) & (freqs <= 51)
    neighbor = ((freqs >= 45) & (freqs < 49)) | ((freqs > 51) & (freqs <= 55))
    line_power = psd[:, :, line].mean(axis=2)
    neighbor_power = psd[:, :, neighbor].mean(axis=2) + 1e-20
    ratio = line_power / neighbor_power
    return _score(ratio, THRESHOLDS["line_ratio"])


def detect_motion(data: np.ndarray, ch_names: list[str], sfreq: float) -> np.ndarray:
    """Motion: large-amplitude deflection that shows up on most channels at
    once (unlike EOG, which is frontal-localized)."""
    ptp = data.max(axis=2) - data.min(axis=2)  # (n_epochs, n_channels)
    channel_hit = ptp > THRESHOLDS["motion_ptp_uv"]
    epoch_frac = channel_hit.mean(axis=1)  # (n_epochs,)
    epoch_is_motion = epoch_frac > THRESHOLDS["motion_channel_frac"]
    score = channel_hit.astype(float) * epoch_is_motion[:, None]
    return score


def detect_cardiac(data: np.ndarray, ch_names: list[str], sfreq: float) -> np.ndarray:
    """Pulse/ECG contamination: a narrow, prominent PSD peak in the resting
    heart-rate band (0.8-2 Hz, ~48-120 bpm).

    KNOWN WEAK DETECTOR. An earlier version of this function used nperseg
    equal to the full epoch (no Welch segment-averaging) and took the raw
    max() over the cardiac band — that combination flags 50-65% of every
    recording's epoch-channels, because an unaveraged periodogram is a very
    high-variance PSD estimate and max() over a handful of bins amplifies
    single-bin noise spikes into fake "peaks". Averaging 3 overlapping
    512-sample segments and using mean() instead of max() over the band
    brings the flag rate down to a plausible ~15-30% on real recordings, but
    this is still the least trustworthy of the six: there's no ECG reference
    channel in this dataset, so it's a scalp-PSD proxy, not a real pulse
    detector, and resting EEG's natural 1/f slope still partially confounds
    "peakiness near 1 Hz" with genuine cardiac contamination. Treat its flags
    as the lowest-confidence of the six until the taxonomy doc gives a
    sharper method or a real ECG channel becomes available."""
    freqs, psd = welch(data, fs=sfreq, nperseg=welch_nperseg(sfreq, data.shape[-1]), axis=-1)
    band = (freqs >= 0.8) & (freqs <= 2.0)
    surround = (freqs >= 0.3) & (freqs < 0.8) | (freqs > 2.0) & (freqs <= 3.0)
    peak_power = psd[:, :, band].mean(axis=2)
    surround_power = psd[:, :, surround].mean(axis=2) + 1e-20
    peakiness = peak_power / surround_power
    return _score(peakiness, THRESHOLDS["cardiac_peakiness"])


DETECTORS = {
    "eog": detect_eog,
    "emg": detect_emg,
    "pop": detect_pop,
    "line_noise": detect_line_noise,
    "motion": detect_motion,
    "cardiac": detect_cardiac,
}


def run_all(data: np.ndarray, ch_names: list[str], sfreq: float) -> dict[str, np.ndarray]:
    """Run every detector; returns {name: (n_epochs, n_channels) score array}."""
    return {name: fn(data, ch_names, sfreq) for name, fn in DETECTORS.items()}
