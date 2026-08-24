"""NeuroQA Study C -- synthetic artifact-injection dose-response test.

Test A (study_a.py) and Test B (study_b.py) are both observational: Test A
shows FAA moves when the CHOICE of cleaning pipeline changes on real,
already-collected data; Test B shows FAA and contamination quality are (or
aren't) statistically associated across real subjects. Neither can show
CAUSATION -- that contamination, specifically, produces a predictable shift
in FAA, as opposed to pipelines simply differing for unrelated reasons, or
quality and FAA being associated through some other real-subject variable.

Study C closes that gap the only way a causal claim about contamination can
be closed short of pharmacological paralysis (Whitham et al. 2007): inject a
KNOWN amount of synthetic contamination into REAL clean epochs and see
whether FAA drifts as a predictable, monotonic function of the injected
dose. If it does -- and does so more for contamination that spectrally
overlaps alpha than for contamination that doesn't -- that's direct,
mechanistic support for the exact mechanism NeuroQA's own quality score is
built on (bands.spectral_overlap). If it doesn't, that's evidence the
observed Test A/B associations aren't simply "more injected noise always
moves FAA."

Scope, stated plainly: this is spectral/amplitude injection on the same
(n_epochs, n_channels, n_samples) arrays the rest of the pipeline already
works with -- bandpass-filtered noise added at frontal channels, calibrated
per-recording to that channel's own amplitude. It is NOT anatomically
realistic forward-model/dipole simulation (see mne.simulation or SEREEGA for
that, a larger undertaking requiring a head model this project doesn't have
set up) -- injected contamination has the right frequency content and
channel location but not a physiologically accurate spatial topography.
That's a real limitation to disclose, not something to paper over: this test
answers "does alpha-overlapping contamination at these channels bias FAA in
a dose-dependent way", not "does a real blink/jaw-clench of this amplitude".

Three injection kinds, chosen to test spectral SPECIFICITY, not just
"more noise = more error":
  eog                 : (0.5, 4.0) Hz  -- matches bands.ARTIFACT_BANDS["eog"]
  emg_alpha_overlap    : (8.0, 13.0) Hz -- the alpha band itself; the
                         worst-case, maximal-overlap slice of the
                         Peter-approved EMG band (8-45 Hz, bands.py)
  emg_no_overlap       : (20.0, 45.0) Hz -- still inside the Peter-approved
                         EMG band, but outside alpha; the specificity
                         control. If FAA drifts here as much as under
                         emg_alpha_overlap, the mechanism isn't specifically
                         about alpha-band overlap.
No new artifact-band definitions are introduced -- eog and both emg slices
are sub-ranges of bands already in bands.py, approved for other uses.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import linregress

from bands import ARTIFACT_BANDS, EEG_BANDS
from faa import compute_faa

FRONTAL_INJECT_CHANNELS = ("Fp1", "Fp2", "F3", "F4")

INJECTION_BANDS: dict[str, tuple[float, float]] = {
    "eog": ARTIFACT_BANDS["eog"],                       # (0.5, 4.0)
    "emg_alpha_overlap": EEG_BANDS["alpha"],             # (8.0, 13.0)
    "emg_no_overlap": (20.0, ARTIFACT_BANDS["emg"][1]),  # (20.0, 45.0)
}

DOSE_LEVELS = (0.0, 0.5, 1.0, 2.0, 4.0)  # multiples of each channel's own
# robust amplitude anchor (see inject_artifact) -- calibrated per-recording,
# not an arbitrary global constant, following Klados & Bamidis (2016).

SEED = 0
BOOTSTRAP_N = 2000


def _bandlimited_noise(n_samples: int, sfreq: float, band: tuple[float, float], rng: np.random.RandomState) -> np.ndarray:
    white = rng.standard_normal(n_samples)
    nyq = sfreq / 2.0
    lo, hi = band[0], min(band[1], nyq - 1.0)
    sos = butter(4, [lo, hi], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, white)


def inject_artifact(data: np.ndarray, ch_names: list[str], sfreq: float,
                     kind: str, dose: float, seed: int = SEED) -> np.ndarray:
    """Return a COPY of `data` with synthetic `kind` contamination added at
    FRONTAL_INJECT_CHANNELS, scaled to `dose` x that channel's own
    recording-specific amplitude anchor (median absolute sample value across
    all epochs) -- so a dose of "1.0" means "roughly this channel's own
    typical amplitude", calibrated per recording rather than a fixed
    microvolt constant that would mean something different on every device/
    dataset.

    dose=0.0 returns an unmodified copy (the no-injection baseline point).

    Each (channel, epoch)'s injected noise *shape* is seeded deterministically
    from (seed, channel) alone, independent of `dose` -- so calling this with
    the same seed at doses 1.0, 2.0, 4.0 ... adds the SAME waveform shape,
    just scaled by a bigger amplitude, rather than a fresh independent random
    draw at every dose. That's what makes a dose-response curve
    interpretable: varying dose is a pure scalar multiplication of one fixed
    contamination pattern, not a resample -- without this, per-dose FAA
    values wobble on independent noise realizations instead of tracing a
    smooth function of dose (confirmed empirically before this fix).
    """
    band = INJECTION_BANDS[kind]
    out = data.copy()
    if dose == 0.0:
        return out
    n_epochs, _n_channels, n_samples = data.shape
    for ch_idx, ch in enumerate(FRONTAL_INJECT_CHANNELS):
        if ch not in ch_names:
            continue
        i = ch_names.index(ch)
        channel_scale = float(np.median(np.abs(data[:, i, :])))
        rng = np.random.RandomState(seed + ch_idx)  # dose-independent -> same shape at every dose
        for e in range(n_epochs):
            noise = _bandlimited_noise(n_samples, sfreq, band, rng)
            peak = np.abs(noise).max()
            if peak <= 0:
                continue
            out[e, i, :] += noise / peak * channel_scale * dose
    return out


def dose_response(data: np.ndarray, ch_names: list[str], sfreq: float,
                   kind: str, doses: tuple[float, ...] = DOSE_LEVELS, seed: int = SEED) -> dict:
    """Recompute unweighted FAA (faa.compute_faa, no quality weighting --
    Study C tests the raw signal's response to contamination, independent of
    NeuroQA's own scorer) at each injected dose for one recording, and fit a
    linear dose-response (scipy.stats.linregress: slope, intercept, r,
    p-value, stderr).
    """
    points = []
    for d in doses:
        contaminated = inject_artifact(data, ch_names, sfreq, kind, d, seed=seed)
        result = compute_faa(contaminated, ch_names, sfreq)
        points.append({"dose": float(d), "faa": float(result["faa"])})
    doses_arr = np.array([p["dose"] for p in points])
    faa_arr = np.array([p["faa"] for p in points])
    fit = linregress(doses_arr, faa_arr)
    return {
        "kind": kind,
        "band": INJECTION_BANDS[kind],
        "points": points,
        "slope": float(fit.slope),
        "intercept": float(fit.intercept),
        "r": float(fit.rvalue),
        "slope_pvalue": float(fit.pvalue),
        "slope_stderr": float(fit.stderr),
    }


def dose_response_for_recording(path, kinds: tuple[str, ...] = tuple(INJECTION_BANDS),
                                 doses: tuple[float, ...] = DOSE_LEVELS,
                                 seed: int = SEED, line_freq: float | None = None) -> dict[str, dict]:
    """dose_response for every injection kind, for one recording (by path).
    Reuses preprocess.py so this sees the same filtered/epoched data the
    rest of the pipeline scores -- no separate preprocessing path to drift
    out of sync with Test A/B."""
    from preprocess import LINE_FREQ, preprocess_file

    data_uv, ch_names, sfreq = preprocess_file(path, line_freq=line_freq if line_freq is not None else LINE_FREQ)
    return {kind: dose_response(data_uv, ch_names, sfreq, kind, doses, seed) for kind in kinds}


def _bootstrap_mean_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    idx = np.arange(len(values))
    boots = [values[rng.choice(idx, size=len(idx), replace=True)].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def run_study_c(paths: list, kinds: tuple[str, ...] = tuple(INJECTION_BANDS),
                 doses: tuple[float, ...] = DOSE_LEVELS, seed: int = SEED,
                 line_freq: float | None = None, n_boot: int = BOOTSTRAP_N) -> dict:
    """Run the dose-response test across multiple recordings and summarize
    each injection kind's population-level slope (mean across subjects, with
    a subject-level bootstrap CI -- NOT a per-recording CI, since each
    recording only contributes one slope estimate to this level).

    A kind whose slope CI excludes 0 shows FAA drifting systematically (not
    just noisily) with injected dose; comparing emg_alpha_overlap's CI to
    emg_no_overlap's CI is the spectral-specificity check (see module
    docstring) -- if only the alpha-overlapping band shows a nonzero slope,
    that's evidence for the specific spectral-overlap mechanism, not just
    generic noise sensitivity.
    """
    per_subject = []
    for p in paths:
        try:
            per_subject.append({"file": str(p), "results": dose_response_for_recording(
                p, kinds=kinds, doses=doses, seed=seed, line_freq=line_freq)})
        except Exception as e:
            per_subject.append({"file": str(p), "error": str(e)})

    summary = {}
    for kind in kinds:
        slopes = np.array([
            s["results"][kind]["slope"] for s in per_subject
            if "results" in s and kind in s["results"]
        ])
        lo, hi = _bootstrap_mean_ci(slopes, n_boot, seed)
        summary[kind] = {
            "band": INJECTION_BANDS[kind],
            "n_subjects": int(len(slopes)),
            "mean_slope": float(slopes.mean()) if len(slopes) else float("nan"),
            "slope_ci_lo": lo,
            "slope_ci_hi": hi,
            "nonzero_slope": bool(len(slopes) and not (lo <= 0.0 <= hi)),
        }

    return {"n_recordings": len(paths), "per_subject": per_subject, "summary": summary}
