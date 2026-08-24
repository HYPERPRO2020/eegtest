"""NeuroQA — frequency bands and the endpoint-overlap primitive.

This module holds the pieces of the endpoint-aware quality index that are
*not* code, they're neuroscience judgment calls: which frequency band each
artifact type lives in, how much a detector's finding should count
(WEIGHT), and the function that turns "artifact band X, we're measuring
band Y" into a single overlap number in [0, 1].

STATUS: ARTIFACT_BANDS below are read off the same frequency ranges the
Step 3 detectors already use internally (see artifact_detectors.py) or, for
EOG/cardiac, standard scalp-EEG convention. WEIGHT is a PLACEHOLDER — every
entry is 1.0 (no artifact type counts for more than any other). Per the
project brief, WEIGHT and the overlap logic are supposed to come from Peter,
derived from artifact physics, not fit to data. Nothing in this repo tunes
WEIGHT against an accuracy number, and nothing should: swap these numbers
for Peter's the moment they exist, that's a one-line change, not a redesign.

Canonical EEG bands (0.5-45 Hz is this pipeline's passband, see
preprocess.py) are the standard clinical breakdown, used both as
ARTIFACT_BANDS entries and as ENDPOINT_BAND choices (e.g. FAA needs alpha).
"""

from __future__ import annotations

Band = tuple[float, float]

# ---- canonical EEG bands (Hz) -----------------------------------------
EEG_BANDS: dict[str, Band] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# ---- which band each Step 3 detector's artifact lives in --------------
# eog/motion: slow deflection, energy concentrated below ~4 Hz (blink/saccade
#   potentials and gross movement artifact are both low-frequency dominant).
# emg: (8.0, 45.0), widened from the detector's own (20, 45) severity window
#   per Goncharova, McFarland, Vaughan & Wolpaw (2003), Clin. Neurophysiol.
#   114(9), 1580-1593 -- EMG is broadband, peaking 20-30 Hz frontally, with
#   measurable power extending down into alpha under weak contraction.
#   Peter-approved (2026-08-22): this is a domain/physics call, not fit to
#   any accuracy number -- see module docstring. detect_emg()'s own 20-45 Hz
#   severity-measurement window is intentionally left unchanged for now
#   (widening *what counts as detected EMG* is a separate question from
#   widening *where its energy overlaps alpha*, and changes what every other
#   endpoint band sees, not just alpha -- flagged for a separate Peter
#   conversation, not bundled into this change).
# pop: an abrupt single-sample step is a broadband transient (in principle
#   flat across the whole passband) -> spans the full 0.5-45 Hz recording band.
# line_noise: matches detect_line_noise's 49-51 Hz band.
# cardiac: matches detect_cardiac's 0.8-2.0 Hz resting-heart-rate band.
ARTIFACT_BANDS: dict[str, Band] = {
    "eog": (0.5, 4.0),
    "emg": (8.0, 45.0),
    "pop": (0.5, 45.0),
    "line_noise": (49.0, 51.0),
    "motion": (0.5, 45.0),
    "cardiac": (0.8, 2.0),
}

# ---- PLACEHOLDER weights — replace with Peter's derived numbers ------
# These are NOT hyperparameters to fit. Equal weighting is the neutral
# placeholder until the real physics-derived values land.
WEIGHT: dict[str, float] = {name: 1.0 for name in ARTIFACT_BANDS}


def spectral_overlap(artifact_band: Band, endpoint_band: Band) -> float:
    """Fraction of `artifact_band`'s width that falls inside `endpoint_band`.

    0.0 = the artifact's energy and the endpoint band never overlap (e.g. a
    49-51 Hz line-noise hit scored against the alpha band). 1.0 = the
    artifact band is fully contained in the endpoint band. This is what
    makes the same detected artifact carry a different penalty depending on
    which band we're measuring -- see quality_index.py.

    Deliberately asymmetric (normalized by the artifact's own bandwidth, not
    the endpoint's or the union): the question being asked is "how much of
    this artifact's energy lands where we're measuring", not how similar the
    two bands are in general.
    """
    a_lo, a_hi = artifact_band
    e_lo, e_hi = endpoint_band
    width = a_hi - a_lo
    if width <= 0:
        return 0.0
    intersection = min(a_hi, e_hi) - max(a_lo, e_lo)
    return max(0.0, min(1.0, intersection / width))


MIN_WELCH_RESOLUTION_HZ = 0.5  # see welch_nperseg


def welch_nperseg(sfreq: float, n_samples: int, target_resolution_hz: float = MIN_WELCH_RESOLUTION_HZ) -> int:
    """Pick a Welch nperseg that won't silently zero out a narrow band.

    A fixed nperseg=512 (this pipeline's original default, still fine at the
    256/500 Hz sample rates it was tuned against) gives frequency resolution
    sfreq/nperseg -- at 2048 Hz that's 4 Hz/bin, coarse enough that the
    narrowest bands here (cardiac's 0.8-2.0 Hz band and its 0.3-0.8 Hz
    surround sub-band, line_noise's 49-51 Hz band) can land entirely between
    two bins and select zero of them. `psd[:, :, empty_mask].mean()` is then
    a mean of an empty slice -> NaN, which silently propagates through the
    quality index and FAA all the way to a whole recording scoring "grade=F,
    quality=nan%" -- confirmed against a real 2048 Hz recording (ds007615),
    not a hypothetical.

    Fix: only scale nperseg *up* from 512 when the sample rate demands finer
    resolution than 512 alone provides, targeting `target_resolution_hz`
    (default 0.5 Hz -- half the width of the narrowest band above, chosen
    empirically so every band here keeps at least one real bin at 256, 500,
    and 2048 Hz, see the module's test coverage). Never exceeds n_samples
    (Welch can't use a window bigger than the data), and never goes below
    512 -- this must be a no-op for the 256/500 Hz recordings the detectors'
    variance/false-positive behavior was already tuned against.
    """
    target = int(sfreq / target_resolution_hz)
    return min(n_samples, max(512, target))
