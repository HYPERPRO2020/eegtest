"""NeuroQA -- FAA Spectral-Composition Tool (v1).

Per Neureidos_Tool_Build_Spec_v1: classic FAA (ln bandpower F4 - ln bandpower
F3, 8-13 Hz) silently mixes a genuine oscillatory alpha peak with the
aperiodic ("1/f") background. This module decomposes each subject's spectrum
with specparam/FOOOF and asks what the FAA number is actually made of, and
whether that composition tracks known confounds (age, arousal, residual EMG)
or clinical severity. A measurement-composition tool, not a classifier --
see the spec's design note: Phase 1 already showed FAA-only classifiers sit
at chance, so questions here are answerable whether or not FAA separates
groups, because they're about the measurement itself, not group differences.

Design decisions made in translating the spec's sketch into code (documented
here since the spec didn't fully pin these down numerically):

  - PSD input to the FOOOF fit is a QUALITY-WEIGHTED mean of per-epoch Welch
    PSDs, not a flat mean -- per the spec's own opening line in Sec. 3,
    "Reuse existing preprocessed, artifact-handled epochs." A flat mean was
    tried first and rejected: on real ds003478 recordings, a handful of
    heavily-contaminated epochs distorted the averaged spectrum enough that
    FOOOF's r_squared collapsed (e.g. one channel: 0.11 flat-averaged vs.
    0.98 using only its cleanest half of epochs -- confirmed empirically,
    not assumed). Weights come from quality_index.compute_quality with
    endpoint_band=FIT_FREQ_RANGE (1-40 Hz, broadband -- there's no single
    canonical endpoint band for a whole-spectrum fit the way alpha has one
    for FAA) -- the same WEIGHT/ARTIFACT_BANDS/decay-function machinery
    Phase 1 already uses, just pointed at a different (broader) endpoint,
    which is a mechanical parameter choice, not a new WEIGHT value or band
    definition requiring domain sign-off. classic_faa, periodic_only_faa,
    and the aperiodic/oscillatory measures are therefore all computed from
    the SAME single (quality-weighted) spectrum per recording -- "classic"
    here labels "total band power" vs. "periodic-only power", not a claim
    of being the fully-naive literature definition Phase 1's own "raw"
    Study A pipeline already represents separately.
  - oscillatory_share is computed from FOOOF's additive-in-log-space model:
    given the full model fit (aperiodic + periodic, in log10 power) and the
    aperiodic-only fit, the LINEAR power attributable to the periodic
    component at each frequency is 10**full_fit - 10**aperiodic_fit (the
    model is multiplicative in linear space, so this recovers the periodic
    component's own additive linear-power contribution). Summed over the
    alpha band and divided by the actual MEASURED (not reconstructed) linear
    PSD summed over the same band -- keeps the denominator identical to
    what classic FAA's band power measures. This is 0.0 automatically when
    no peak was fit (peak_fit is all zeros), satisfying the spec's "share
    is ~0 / undefined is data, not missingness" rule without a special case.
  - Periodic-only FAA uses FOOOF's PW (peak power above the aperiodic
    baseline, log10 units) directly, per the spec's formula -- NaN if
    either channel has no peak, never imputed to 0 (a materially different
    convention from oscillatory_share above; see spec Sec. 5).
"""

from __future__ import annotations

import warnings

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)  # fooof -> specparam rename notice
    from fooof import FOOOF
    from fooof.analysis import get_band_peak_fm
from scipy.signal import welch

from bands import EEG_BANDS, welch_nperseg
from quality_index import compute_quality

FIT_FREQ_RANGE = (1.0, 40.0)  # stops below 50/60 Hz notch -- spec Sec. 3
MIN_R_SQUARED = 0.90  # spec Sec. 5 says to flag+report low-r_squared fits but
# doesn't supply a numeric threshold; 0.90 is a defensible, disclosed default
# (common in specparam-based literature), not a scientific parameter --
# unlike WEIGHT/band definitions this is a QC/reporting convention, not a
# domain-physics judgment call. Real resting EEG over the full 1-40 Hz range
# routinely fits worse than this (EMG bleed near 40 Hz, filter-edge effects
# near 1 Hz are both real, not artifacts of this code -- confirmed against
# real ds003478 recordings before this threshold was added) -- this flag
# exists so those cases get excluded and counted, not silently included as
# if the decomposition were trustworthy.
ALPHA_BAND = EEG_BANDS["alpha"]  # (8.0, 13.0), same band classic FAA uses

# Exact FOOOF params from the spec's code sketch (Sec. "Step 2") -- fixed up
# front and held constant across datasets/conditions, per spec Sec. 6
# ("Fit range and peak settings are the main researcher-degrees-of-freedom
# here -- fix them up front... don't tune per dataset").
FOOOF_PARAMS = dict(peak_width_limits=[1, 8], max_n_peaks=6, min_peak_height=0.10)
APERIODIC_MODES = ("fixed", "knee")  # run both, per spec Sec. 6 robustness check


def avg_psd(data: np.ndarray, ch_names: list[str], sfreq: float,
            weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Mean PSD per channel across epochs, via Welch -- quality-weighted if
    `weights` (n_epochs, n_channels) is given, flat otherwise (used only by
    tests/diagnostics; the real pipeline always weights, see module
    docstring).

    data: (n_epochs, n_channels, n_samples) in microvolts.
    Returns (freqs, psd) with psd shape (n_channels, n_freqs).
    """
    freqs, per_epoch_psd = welch(data, fs=sfreq, nperseg=welch_nperseg(sfreq, data.shape[-1]), axis=-1)
    if weights is None:
        return freqs, per_epoch_psd.mean(axis=0)
    weights_bc = weights[:, :, None]  # (n_epochs, n_channels, 1), broadcasts against per_epoch_psd's (n_epochs, n_channels, n_freqs)
    channel_totals = weights_bc.sum(axis=0)  # (n_channels, 1)
    flat = per_epoch_psd.mean(axis=0)
    weighted = (per_epoch_psd * weights_bc).sum(axis=0) / np.where(channel_totals > 0, channel_totals, 1.0)
    zero_weight_channels = channel_totals[:, 0] <= 0  # degenerate all-zero-weight channel -> fall back to flat mean
    weighted[zero_weight_channels] = flat[zero_weight_channels]
    return freqs, weighted


def quality_weighted_psd(data: np.ndarray, ch_names: list[str], sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    """The pipeline's actual PSD input: per-epoch Welch PSD averaged with
    quality_index.compute_quality weights (endpoint_band=FIT_FREQ_RANGE,
    broadband -- see module docstring for why flat averaging was rejected).
    """
    quality = compute_quality(data, ch_names, sfreq, FIT_FREQ_RANGE)["quality"]  # (n_epochs, n_channels)
    return avg_psd(data, ch_names, sfreq, weights=quality)


def band_power(freqs: np.ndarray, psd_1d: np.ndarray, band: tuple[float, float]) -> float:
    """Sum of PSD within `band` for one channel's spectrum -- same convention
    faa.py's alpha_power uses (mean, not sum, of the in-band bins), kept
    consistent so classic FAA computed here matches faa.py's own number."""
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return float(psd_1d[mask].mean()) if mask.any() else float("nan")


def fit_fooof(freqs: np.ndarray, psd_1d: np.ndarray, aperiodic_mode: str = "fixed") -> dict:
    """Fit one channel's averaged PSD with FOOOF/specparam.

    Returns offset, exponent, r_squared (aperiodic + fit quality), alpha
    peak CF/PW/BW and peak_present (periodic), and oscillatory_share (see
    module docstring for its derivation). PW/CF/BW are NaN when no peak is
    detected above the background -- oscillatory_share is 0.0 in that case,
    not NaN (see module docstring: these two use deliberately different
    no-peak conventions, per spec Sec. 5).
    """
    fm = FOOOF(aperiodic_mode=aperiodic_mode, verbose=False, **FOOOF_PARAMS)
    fm.fit(freqs, psd_1d, list(FIT_FREQ_RANGE))

    aperiodic = fm.get_params("aperiodic_params")
    offset = float(aperiodic[0])
    exponent = float(aperiodic[-1])  # last entry is always the exponent (2nd for 'fixed', 3rd for 'knee')
    r_squared = float(fm.get_params("r_squared"))

    cf, pw, bw = get_band_peak_fm(fm, list(ALPHA_BAND))
    peak_present = not np.isnan(cf)

    alpha_mask = (fm.freqs >= ALPHA_BAND[0]) & (fm.freqs <= ALPHA_BAND[1])
    if alpha_mask.any():
        periodic_linear = np.clip(10 ** fm.fooofed_spectrum_ - 10 ** fm._ap_fit, 0, None)
        measured_linear = 10 ** fm.power_spectrum
        denom = measured_linear[alpha_mask].sum()
        oscillatory_share = float(np.clip(periodic_linear[alpha_mask].sum() / denom, 0.0, 1.0)) if denom > 0 else float("nan")
    else:
        oscillatory_share = float("nan")  # alpha band entirely outside the fit range -- shouldn't happen given FIT_FREQ_RANGE

    return {
        "offset": offset, "exponent": exponent, "r_squared": r_squared,
        "low_quality_fit": bool(r_squared < MIN_R_SQUARED),
        "alpha_cf": float(cf) if peak_present else float("nan"),
        "alpha_pw": float(pw) if peak_present else float("nan"),
        "alpha_bw": float(bw) if peak_present else float("nan"),
        "peak_present": bool(peak_present),
        "oscillatory_share": oscillatory_share,
    }


def spectral_composition_for_recording(data: np.ndarray, ch_names: list[str], sfreq: float,
                                        aperiodic_mode: str = "fixed",
                                        channels: tuple[str, ...] = ("F3", "F4")) -> dict[str, dict]:
    """FOOOF fit + classic band power for each of `channels` (default F3/F4,
    the v1 target -- spec Sec. 3 says keep all channels stored for later,
    but v1 only targets F3/F4). Uses quality_weighted_psd, not a flat
    average -- see module docstring."""
    freqs, psd = quality_weighted_psd(data, ch_names, sfreq)
    out = {}
    for ch in channels:
        if ch not in ch_names:
            continue
        i = ch_names.index(ch)
        result = fit_fooof(freqs, psd[i], aperiodic_mode=aperiodic_mode)
        result["classic_band_power"] = band_power(freqs, psd[i], ALPHA_BAND)
        out[ch] = result
    return out


THETA_BAND = EEG_BANDS["theta"]  # (4.0, 8.0), for the theta/alpha arousal proxy
EMG_BAND = (20.0, 45.0)  # spec Sec. 4.C's literal example range for "residual
# EMG-band power" -- distinct from bands.ARTIFACT_BANDS["emg"] (8-45 Hz,
# Peter-approved for the alpha-overlap penalty in the *other* NeuroQA tool);
# here it's a direct band-power confound measurement the spec names
# explicitly, not an artifact-detector severity score.


def theta_alpha_ratio(freqs: np.ndarray, psd_1d: np.ndarray) -> float:
    """Arousal proxy (spec Sec. 4.C): theta/alpha power ratio, one channel's
    already-computed spectrum -- higher theta/alpha is associated with lower
    arousal/drowsier states in the literature this proxy is drawn from."""
    theta = band_power(freqs, psd_1d, THETA_BAND)
    alpha = band_power(freqs, psd_1d, ALPHA_BAND)
    return float(theta / alpha) if alpha > 0 else float("nan")


def alpha_power_trend(data: np.ndarray, ch_names: list[str], sfreq: float,
                       channels: tuple[str, ...] = ("F3", "F4")) -> dict[str, float]:
    """Second arousal proxy (spec Sec. 4.C): slope of per-epoch alpha-band
    power across the recording, per channel. Negative = alpha power
    declining over the session (the spec's "decline in alpha power across
    the recording" phrasing), positive = increasing. Uses the UNWEIGHTED
    per-epoch PSD (there's no single-epoch "quality" to weight a trend
    across epochs by -- this measures how epoch-to-epoch power itself
    changes, which is the whole point)."""
    freqs, per_epoch_psd = welch(data, fs=sfreq, nperseg=welch_nperseg(sfreq, data.shape[-1]), axis=-1)
    mask = (freqs >= ALPHA_BAND[0]) & (freqs <= ALPHA_BAND[1])
    epoch_idx = np.arange(per_epoch_psd.shape[0])
    out = {}
    for ch in channels:
        if ch not in ch_names:
            continue
        i = ch_names.index(ch)
        alpha_per_epoch = per_epoch_psd[:, i, mask].mean(axis=1)
        out[ch] = float(np.polyfit(epoch_idx, alpha_per_epoch, 1)[0]) if len(epoch_idx) >= 2 else float("nan")
    return out


def faa_variants(f3: dict, f4: dict) -> dict:
    """The three FAA variants (spec Sec. 3), from one recording's F3/F4
    spectral_composition_for_recording() results.

    classic_faa: ln(bandpower_F4) - ln(bandpower_F3), total power.
    periodic_only_faa: PW_F4 - PW_F3 (FOOOF's log10 peak power above the
      aperiodic baseline) -- NaN if EITHER channel lacks a detected peak,
      recorded as such, never imputed to 0 (spec Sec. 5).
    aperiodic_asymmetry_exponent / _offset: F4 - F3 for each aperiodic param.
    """
    classic_faa = float(np.log(f4["classic_band_power"] + 1e-20) - np.log(f3["classic_band_power"] + 1e-20))
    both_peaks = f3["peak_present"] and f4["peak_present"]
    periodic_only_faa = float(f4["alpha_pw"] - f3["alpha_pw"]) if both_peaks else float("nan")
    return {
        "classic_faa": classic_faa,
        "periodic_only_faa": periodic_only_faa,
        "both_peaks_present": bool(both_peaks),
        "aperiodic_asymmetry_exponent": float(f4["exponent"] - f3["exponent"]),
        "aperiodic_asymmetry_offset": float(f4["offset"] - f3["offset"]),
    }
