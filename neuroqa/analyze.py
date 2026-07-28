"""NeuroQA — single-file analysis for the upload-and-grade UI.

Wraps the same ingest -> preprocess -> detect -> score pipeline the batch
scripts (ingest.py/preprocess.py/score.py) run over the whole dataset, but
for exactly one uploaded .edf, synchronously, and returns one JSON-ready
dict with everything webapp.py needs to render "where it lost points":
overall grade per band, per-channel quality, a per-epoch quality timeline,
a penalty breakdown by artifact type, and a ranked list of the worst
epoch-channel cells with the detector responsible for each.

Computed once for all 5 canonical EEG_BANDS so the UI's band selector is a
pure client-side re-render, no repeat upload/recompute per band switch.
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np

from artifact_detectors import run_all
from bands import EEG_BANDS
from ingest import EXPECTED_CHANNELS, clean_channel_name
from preprocess import CHANNEL_ORDER, EPOCH_SEC, H_FREQ, L_FREQ, LINE_FREQ, OVERLAP_SEC
from quality_index import compute_quality
from score import grade_from_pct

mne.set_log_level("ERROR")

TOP_OFFENDERS_N = 15

# Waveform decimation: two rules, whichever bites.
#   1. Never decimate below MIN_DISPLAY_HZ -- the signal is bandpassed to
#      45 Hz, so a naive stride-decimate (no anti-alias lowpass) below
#      ~2x that would visibly alias exactly the muscle-band (20-45 Hz)
#      artifacts this tool exists to show. 100 Hz keeps real headroom.
#   2. If even that leaves the payload too big (pathologically long
#      recordings), fall back to a hard cap on total values shipped --
#      accepting some aliasing risk only in that edge case, not the
#      ~5 minute resting-state clips this pipeline is built around.
MIN_DISPLAY_HZ = 100.0
MAX_WAVEFORM_VALUES = 900_000


class AnalysisError(ValueError):
    """Raised for problems with the uploaded file itself (bad channels, too
    short, unreadable) -- distinct from unexpected bugs, so webapp.py can
    show a clean message instead of a stack trace."""


def _load_epochs(path: str | Path) -> tuple[np.ndarray, np.ndarray, float]:
    try:
        raw = mne.io.read_raw_edf(str(path), preload=True, verbose=False)
    except Exception as e:
        raise AnalysisError(f"couldn't read this as an EDF file: {e}") from e

    raw.rename_channels({ch: clean_channel_name(ch) for ch in raw.ch_names})
    missing = EXPECTED_CHANNELS - set(raw.ch_names)
    if missing:
        raise AnalysisError(
            "missing required 10-20 channels: " + ", ".join(sorted(missing)) +
            f" (found: {', '.join(sorted(raw.ch_names))})"
        )
    raw.pick(CHANNEL_ORDER)
    raw.reorder_channels(CHANNEL_ORDER)
    raw.notch_filter(LINE_FREQ, verbose=False)
    raw.filter(L_FREQ, H_FREQ, verbose=False)

    # Continuous filtered signal, for the waveform viewer -- captured before
    # epoching so it isn't duplicated across the 50%-overlapping epoch windows.
    continuous_uv = raw.get_data() * 1e6  # (n_channels, n_samples)

    epochs = mne.make_fixed_length_epochs(
        raw, duration=EPOCH_SEC, overlap=OVERLAP_SEC, preload=True, verbose=False,
    )
    if len(epochs) == 0:
        raise AnalysisError(
            f"recording is too short to produce a single {EPOCH_SEC:.0f}s epoch"
        )
    data_uv = epochs.get_data() * 1e6
    return data_uv, continuous_uv, float(raw.info["sfreq"])


def _score_band(data_uv: np.ndarray, sfreq: float, band_name: str,
                 detector_scores: dict[str, np.ndarray], step_sec: float) -> dict:
    result = compute_quality(data_uv, CHANNEL_ORDER, sfreq, EEG_BANDS[band_name],
                              detector_scores=detector_scores)
    quality = result["quality"]  # (n_epochs, n_channels)
    contribs = result["contributions"]  # {name: (n_epochs, n_channels)}

    total_contrib = np.zeros_like(quality)
    for arr in contribs.values():
        total_contrib += arr

    overall_pct = float(quality.mean())
    channel_quality_pct = {
        ch: round(float(quality[:, i].mean()), 2) for i, ch in enumerate(CHANNEL_ORDER)
    }
    epoch_quality_pct = [round(float(v), 2) for v in quality.mean(axis=1)]
    # Per-channel, per-epoch (not averaged across channels) -- lets the
    # waveform viewer highlight exactly which channel is bad at which moment,
    # instead of tinting every channel the same shared, averaged color.
    channel_epoch_quality_pct = {
        ch: [round(float(v), 1) for v in quality[:, i]] for i, ch in enumerate(CHANNEL_ORDER)
    }

    penalty_by_detector = {name: float(arr.sum()) for name, arr in contribs.items()}
    total_penalty = sum(penalty_by_detector.values())
    detector_penalty_share_pct = {
        name: round(100.0 * v / total_penalty, 2) if total_penalty > 0 else 0.0
        for name, v in penalty_by_detector.items()
    }

    flat_order = np.argsort(total_contrib.ravel())[::-1][:TOP_OFFENDERS_N]
    top_offenders = []
    for flat_idx in flat_order:
        e, ch_i = np.unravel_index(flat_idx, total_contrib.shape)
        penalty = float(total_contrib[e, ch_i])
        if penalty <= 0:
            break  # sorted descending -- nothing meaningful left
        dominant_name = max(contribs, key=lambda n: contribs[n][e, ch_i])
        top_offenders.append({
            "epoch": int(e),
            "time_sec": round(float(e) * step_sec, 1),
            "channel": CHANNEL_ORDER[ch_i],
            "quality_pct": round(float(quality[e, ch_i]), 1),
            "penalty": round(penalty, 3),
            "dominant_detector": dominant_name,
            "dominant_detector_severity": round(float(detector_scores[dominant_name][e, ch_i]), 3),
        })

    return {
        "overall_quality_pct": round(overall_pct, 2),
        "grade": grade_from_pct(overall_pct),
        "channel_quality_pct": channel_quality_pct,
        "epoch_quality_pct": epoch_quality_pct,
        "channel_epoch_quality_pct": channel_epoch_quality_pct,
        "detector_penalty_share_pct": detector_penalty_share_pct,
        "top_offenders": top_offenders,
    }


def _build_waveform_payload(continuous_uv: np.ndarray, sfreq: float) -> dict:
    """JSON-ready waveform block: {sfreq, channels, duration_sec, samples}.

    samples is channel-major: [[ch0_sample0, ch0_sample1, ...], [ch1...], ...].
    Sent whole in the /analyze response (see MAX_WAVEFORM_VALUES) so the
    frontend can pan/zoom by slicing this array locally -- no follow-up
    request to a server that, on a stateless/serverless deployment, might not
    even be the same process that did the original analysis.

    Decimation, when it kicks in, is simple stride-based downsampling (no
    anti-alias lowpass first) -- a real simplification, acceptable here
    because it only engages for recordings long enough to need it and the
    signal's already lowpassed to 45 Hz, but worth knowing if the display
    ever looks aliased on an unusually long recording.
    """
    n_channels, n_samples = continuous_uv.shape
    stride = max(1, int(sfreq // MIN_DISPLAY_HZ))
    if (n_channels * n_samples) / stride > MAX_WAVEFORM_VALUES:
        stride = max(1, round((n_channels * n_samples) / MAX_WAVEFORM_VALUES))
    decimated = continuous_uv[:, ::stride]
    return {
        "sfreq": sfreq / stride,
        "channels": CHANNEL_ORDER,
        "duration_sec": n_samples / sfreq,
        "samples": np.round(decimated, 1).tolist(),
    }


def analyze_edf(path: str | Path, filename: str | None = None) -> dict:
    """Run the full endpoint-aware pipeline on one uploaded .edf.

    Returns one JSON-ready dict: {filename, n_epochs, n_channels, sfreq,
    epoch_sec, step_sec, channels, primary_band, bands: {...}, waveform: {...}}.
    Everything the UI needs -- including the (possibly decimated) waveform
    for panning/zooming -- comes back in this single response; nothing is
    held server-side between requests. Raises AnalysisError for problems
    with the file itself.
    """
    data_uv, continuous_uv, sfreq = _load_epochs(path)
    n_epochs, n_channels, _ = data_uv.shape
    step_sec = EPOCH_SEC - OVERLAP_SEC

    detector_scores = run_all(data_uv, CHANNEL_ORDER, sfreq)

    bands_out = {
        band_name: _score_band(data_uv, sfreq, band_name, detector_scores, step_sec)
        for band_name in EEG_BANDS
    }

    return {
        "filename": filename or Path(path).name,
        "n_epochs": n_epochs,
        "n_channels": n_channels,
        "sfreq": sfreq,
        "epoch_sec": EPOCH_SEC,
        "step_sec": step_sec,
        "channels": CHANNEL_ORDER,
        "primary_band": "alpha",
        "bands": bands_out,
        "waveform": _build_waveform_payload(continuous_uv, sfreq),
    }
