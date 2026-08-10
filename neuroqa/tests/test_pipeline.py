"""Hand-check the pipeline against synthetic data before trusting it on
real uploads: manifest validation on a synthesized raw file, and the
endpoint-aware scorer's response to a clean vs. an artifact-injected
recording. Not a golden-value regression suite (there's no real dataset in
this environment to regress against) -- these assert *directional*
sanity: clean scores higher than dirty, FAA sign follows planted asymmetry,
detectors fire on what they're supposed to detect.

Run: python -m pytest neuroqa/tests/test_pipeline.py -v
  (or: python neuroqa/tests/test_pipeline.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from artifact_detectors import run_all
from bands import EEG_BANDS
from faa import compute_faa
from manifest import STANDARD_1020, ValidationResult, validate_recording
from quality_index import compute_quality

mne.set_log_level("ERROR")

CH_NAMES = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
            "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
SFREQ = 256.0
DURATION_SEC = 120.0


def make_synthetic_raw(alpha_amp_f3: float = 10.0, alpha_amp_f4: float = 10.0,
                        seed: int = 0) -> mne.io.RawArray:
    """19-channel synthetic resting EEG: broadband 1/f-ish noise plus a
    per-channel alpha (10Hz) component, with independently controllable
    alpha amplitude at F3/F4 so FAA's sign is a known, planted quantity."""
    rng = np.random.default_rng(seed)
    n_samples = int(DURATION_SEC * SFREQ)
    t = np.arange(n_samples) / SFREQ
    data = rng.normal(0, 3e-6, size=(len(CH_NAMES), n_samples))  # ~3uV broadband noise
    for i, ch in enumerate(CH_NAMES):
        amp = alpha_amp_f3 if ch == "F3" else alpha_amp_f4 if ch == "F4" else 6.0
        phase = rng.uniform(0, 2 * np.pi)
        data[i] += (amp * 1e-6) * np.sin(2 * np.pi * 10.0 * t + phase)
    info = mne.create_info(CH_NAMES, sfreq=SFREQ, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose=False)


def inject_blink_and_emg(raw: mne.io.RawArray, seed: int = 1) -> mne.io.RawArray:
    """Plant a clearly-artifactual version: large low-frequency frontal
    deflections (blink-like) and elevated 20-45Hz broadband power on every
    channel (muscle-like) — should tank delta/theta/beta/gamma quality far
    more than alpha, and should be flagged by detect_eog/detect_emg."""
    raw = raw.copy()
    rng = np.random.default_rng(seed)
    data = raw.get_data()
    n_samples = data.shape[1]
    t = np.arange(n_samples) / SFREQ
    frontal = {"Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"}
    for i, ch in enumerate(raw.ch_names):
        if ch in frontal:
            data[i] += 150e-6 * np.sin(2 * np.pi * 0.5 * t + rng.uniform(0, 1))  # blink-like, <4Hz
        data[i] += rng.normal(0, 15e-6, size=n_samples)  # broadband incl. 20-45Hz -> EMG-like
    raw._data = data
    return raw


def epoch(raw: mne.io.RawArray) -> np.ndarray:
    epochs = mne.make_fixed_length_epochs(raw, duration=4.0, overlap=2.0, preload=True, verbose=False)
    return epochs.get_data() * 1e6  # -> uV, matches the rest of the pipeline's convention


def test_manifest_validation_accepts_clean_synthetic_file(tmp_path):
    raw = make_synthetic_raw()
    path = tmp_path / "sub01.fif"
    raw.save(str(path), verbose=False)
    result = validate_recording(path, {"diagnosis_raw": "healthy", "severity_raw": "3"})
    assert result.ok, result.reasons
    assert result.diagnosis == "healthy"
    assert result.severity == 3.0
    assert {"F3", "F4"}.issubset(set(result.channels_found))


def test_manifest_validation_rejects_missing_f3_f4(tmp_path):
    info = mne.create_info(["Fp1", "Fp2", "O1", "O2"], sfreq=SFREQ, ch_types="eeg")
    raw = mne.io.RawArray(np.zeros((4, int(SFREQ * 120))), info, verbose=False)
    path = tmp_path / "no_frontal.fif"
    raw.save(str(path), verbose=False)
    result = validate_recording(path, {"diagnosis_raw": "healthy", "severity_raw": "1"})
    assert not result.ok
    assert any("F3" in r or "F4" in r for r in result.reasons)


def test_manifest_validation_rejects_bad_label():
    result = validate_recording(Path("nonexistent.edf"), {"diagnosis_raw": "maybe", "severity_raw": "1"})
    assert not result.ok
    assert any("unrecognized diagnosis" in r for r in result.reasons)


def test_manifest_validation_rejects_no_manifest_row():
    result = validate_recording(Path("nonexistent.edf"), None)
    assert not result.ok
    assert any("no manifest row" in r for r in result.reasons)


def test_quality_index_penalizes_dirty_more_than_clean_outside_alpha():
    clean = epoch(make_synthetic_raw())
    dirty = epoch(inject_blink_and_emg(make_synthetic_raw()))

    clean_detectors = run_all(clean, CH_NAMES, SFREQ)
    dirty_detectors = run_all(dirty, CH_NAMES, SFREQ)

    clean_delta = compute_quality(clean, CH_NAMES, SFREQ, EEG_BANDS["delta"], clean_detectors)["quality"].mean()
    dirty_delta = compute_quality(dirty, CH_NAMES, SFREQ, EEG_BANDS["delta"], dirty_detectors)["quality"].mean()
    assert dirty_delta < clean_delta, "planted blink/EMG contamination should tank delta-band quality"

    # eog detector should fire much harder on the blink-injected recording
    assert dirty_detectors["eog"].mean() > clean_detectors["eog"].mean()
    assert dirty_detectors["emg"].mean() > clean_detectors["emg"].mean()


def test_quality_index_is_endpoint_aware_not_generic():
    """The whole point: the SAME detected artifacts should cost alpha-band
    quality much less than delta-band quality, because a blink's energy
    sits below alpha and this scorer is supposed to know that."""
    dirty = epoch(inject_blink_and_emg(make_synthetic_raw()))
    detectors = run_all(dirty, CH_NAMES, SFREQ)

    alpha_quality = compute_quality(dirty, CH_NAMES, SFREQ, EEG_BANDS["alpha"], detectors)["quality"].mean()
    delta_quality = compute_quality(dirty, CH_NAMES, SFREQ, EEG_BANDS["delta"], detectors)["quality"].mean()

    assert alpha_quality > delta_quality, (
        f"alpha quality ({alpha_quality:.1f}) should exceed delta quality ({delta_quality:.1f}) "
        "on the same recording -- a blink shouldn't cost as much when measuring alpha as when "
        "measuring delta, that conditional weighting is the entire point of this scorer"
    )


def test_faa_sign_follows_planted_asymmetry():
    louder_f4 = epoch(make_synthetic_raw(alpha_amp_f3=6.0, alpha_amp_f4=14.0))
    result = compute_faa(louder_f4, CH_NAMES, SFREQ)
    assert result["faa"] > 0, "F4 alpha louder than F3 should give positive FAA (ln(F4) - ln(F3))"

    louder_f3 = epoch(make_synthetic_raw(alpha_amp_f3=14.0, alpha_amp_f4=6.0))
    result2 = compute_faa(louder_f3, CH_NAMES, SFREQ)
    assert result2["faa"] < 0


def test_faa_quality_weighting_pulls_toward_the_cleaner_epochs():
    """When epochs are quality-weighted (the 'ours' pipeline), an epoch's
    influence on the aggregate FAA should scale with its own weight."""
    data = epoch(make_synthetic_raw())
    n_epochs = data.shape[0]
    all_weight_on_first = np.zeros(n_epochs)
    all_weight_on_first[0] = 1.0
    flat = compute_faa(data, CH_NAMES, SFREQ)
    weighted = compute_faa(data, CH_NAMES, SFREQ, weights_f3=all_weight_on_first, weights_f4=all_weight_on_first)
    # first epoch alone should reproduce exactly if isolated
    assert np.isclose(weighted["faa"], weighted["faa_per_epoch"][0], atol=1e-6)
    assert not np.isclose(flat["faa"], weighted["faa"]) or n_epochs == 1


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
