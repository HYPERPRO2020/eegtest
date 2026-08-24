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


def test_manifest_validation_accepts_missing_severity_with_warning(tmp_path):
    """A dataset that genuinely ships no clinical severity (e.g. Mumtaz/HUSM's
    public deposit -- diagnosis label only) must still be usable for Study A
    and Study B's non-severity analyses, just flagged, not hard-rejected."""
    raw = make_synthetic_raw()
    path = tmp_path / "sub_no_severity.fif"
    raw.save(str(path), verbose=False)
    result = validate_recording(path, {"diagnosis_raw": "healthy", "severity_raw": ""})
    assert result.ok, result.reasons
    assert result.severity is None
    assert any("severity" in w for w in result.warnings)


def test_manifest_validation_rejects_garbage_severity(tmp_path):
    """Missing severity is fine (see above); a present-but-unparseable value
    (typo, wrong column) is still a real data problem and a hard rejection."""
    raw = make_synthetic_raw()
    path = tmp_path / "sub_bad_severity.fif"
    raw.save(str(path), verbose=False)
    result = validate_recording(path, {"diagnosis_raw": "healthy", "severity_raw": "not-a-number"})
    assert not result.ok
    assert any("not a number" in r for r in result.reasons)


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


def test_emg_band_now_overlaps_alpha_and_penalizes_it():
    """ARTIFACT_BANDS["emg"] was (20, 45) Hz -- zero overlap with alpha
    (8-13 Hz), so detected EMG contributed nothing to alpha-endpoint
    quality no matter how severe, in tension with the brief's own
    hypothesis that muscle noise shares alpha's frequency band. Widened to
    (8, 45) per Goncharova et al. (2003) (Peter-approved, 2026-08-22, see
    bands.py). This must now measurably cost alpha-band quality -- the
    entire reason for the change was to let EMG bite the endpoint that
    matters for FAA, not just delta/beta/gamma."""
    from bands import ARTIFACT_BANDS, spectral_overlap

    assert spectral_overlap(ARTIFACT_BANDS["emg"], EEG_BANDS["alpha"]) > 0.0

    clean = epoch(make_synthetic_raw())
    dirty = epoch(inject_blink_and_emg(make_synthetic_raw()))
    clean_detectors = run_all(clean, CH_NAMES, SFREQ)
    dirty_detectors = run_all(dirty, CH_NAMES, SFREQ)

    clean_alpha = compute_quality(clean, CH_NAMES, SFREQ, EEG_BANDS["alpha"], clean_detectors)["quality"].mean()
    dirty_alpha = compute_quality(dirty, CH_NAMES, SFREQ, EEG_BANDS["alpha"], dirty_detectors)["quality"].mean()
    assert dirty_alpha < clean_alpha, (
        f"dirty alpha quality ({dirty_alpha:.1f}) should now be measurably below clean "
        f"({clean_alpha:.1f}) -- planted EMG must cost the alpha endpoint something now"
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


def test_welch_nperseg_keeps_narrow_bands_non_empty_at_high_sample_rates():
    """A fixed nperseg=512 gives 4 Hz/bin resolution at 2048 Hz -- coarse
    enough that the cardiac band (0.8-2.0 Hz) and its 0.3-0.8 Hz surround
    sub-band, and line_noise's 49-51 Hz band, can select zero PSD bins,
    producing mean-of-empty-slice NaN. Confirmed against a real 2048 Hz
    recording (ds007615) before this fix -- not a hypothetical. Every
    sample rate this pipeline has actually seen (256/500/2048 Hz) must keep
    at least one real bin in the narrowest band used anywhere."""
    from bands import welch_nperseg
    from scipy.signal import welch as scipy_welch

    narrow_bands = [(0.8, 2.0), (0.3, 0.8), (2.0, 3.0), (49.0, 51.0)]
    for sfreq in (256.0, 500.0, 2048.0):
        n_samples = int(sfreq * 4.0)  # this pipeline's epoch length
        nperseg = welch_nperseg(sfreq, n_samples)
        freqs, _ = scipy_welch(np.zeros(n_samples), fs=sfreq, nperseg=nperseg)
        for lo, hi in narrow_bands:
            n_bins = int(((freqs >= lo) & (freqs <= hi)).sum())
            assert n_bins > 0, f"sfreq={sfreq}: band ({lo},{hi}) has zero bins at nperseg={nperseg}"
    # must be a no-op at 256 Hz specifically -- nperseg=512 already gives
    # 0.5 Hz resolution there, exactly this function's target, so the
    # detectors' variance/false-positive behavior at that rate (see
    # artifact_detectors.detect_cardiac's docstring) is unaffected.
    assert welch_nperseg(256.0, 1024) == 512
    # never below the original 512 floor, and never above what's available
    assert welch_nperseg(500.0, 2000) >= 512
    assert welch_nperseg(2048.0, 8192) <= 8192


def test_score_and_faa_is_deterministic(tmp_path):
    """Reproducibility: the same uploaded recording must score identically
    on repeat runs (no unseeded randomness in the default scoring path)."""
    from pipeline import score_and_faa

    raw = make_synthetic_raw()
    path = tmp_path / "sub_repro.fif"
    raw.save(str(path), verbose=False)

    first = score_and_faa(path)
    second = score_and_faa(path)
    assert first == second


def test_study_b_regression_3_is_deterministic():
    """AutoReject/ICA/CV-split randomness is all seeded (see pipeline.SEED
    and study_b.CV_SEED) -- the classifier's CV predictions, and therefore
    its accuracy/AUC, must be bit-identical across repeat runs on the same
    rows, not just close."""
    from study_b import regression_3, rows_to_frame

    rng = np.random.default_rng(7)
    rows = [
        {"file": f"s{i}", "group": "healthy" if i % 2 == 0 else "depressed",
         "quality_alpha_pct": float(rng.uniform(60, 100)),
         "clinical_severity": float(rng.uniform(0, 63)), "faa": float(rng.normal())}
        for i in range(12)
    ]
    df = rows_to_frame(rows, "quality_alpha_pct")
    first = regression_3(df, "quality_alpha_pct")
    second = regression_3(df, "quality_alpha_pct")
    assert first == second


def _severity_rows(rng, n=12, with_severity=True):
    return [
        {"file": f"s{i}", "group": "healthy" if i % 2 == 0 else "depressed",
         "quality_alpha_pct": float(rng.uniform(60, 100)),
         "quality_alpha_frontal_pct": float(rng.uniform(60, 100)),
         "clinical_severity": float(rng.uniform(0, 63)) if with_severity else None,
         "faa": float(rng.normal())}
        for i in range(n)
    ]


def test_study_b_regression_1_uses_clinical_severity_not_artifact_severity():
    """Regression 1 must regress on the manifest's clinical severity (BDI/
    HAM-D-like, external to the recording), not an EEG-derived quantity --
    using the latter would make quality~severity circular, since quality is
    itself computed from artifact severity (see study_b.py docstring)."""
    from study_b import run_study_b

    rng = np.random.default_rng(3)
    rows = _severity_rows(rng)
    result = run_study_b(rows)
    assert result["regression_1_quality_on_group_severity"] is not None
    assert result["regression_1_skipped_reason"] is None
    assert "clinical_severity" in result["regression_1_quality_on_group_severity"]["params"]
    assert result["n_with_severity"] == len(rows)


def test_study_b_finding_reports_real_signal_when_group_significant_and_no_leakage():
    """A batch where group significantly predicts FAA (controlling for
    quality) and quality-alone doesn't leak group must report the
    real-signal finding -- regardless of whether quality's OWN coefficient
    in that same regression happens to be significant. A prior version
    required both group AND quality coefficients to be significant, which
    contradicted the finding text it printed ("quality alone doesn't
    predict group above chance" describes the B.2 leakage check, already
    handled by the branch above -- not regression 2's quality coefficient)
    and meant a real, non-confounded group-FAA relationship (confirmed on
    real data, ds007615) was reported as "inconclusive"."""
    from study_b import run_study_b

    rng = np.random.default_rng(11)
    n = 40
    group = ["healthy" if i % 2 == 0 else "depressed" for i in range(n)]
    quality = rng.uniform(85, 100, size=n)  # unrelated to group by construction
    # FAA carries a real, large group effect and is NOT related to quality
    faa = [(-1.0 if g == "healthy" else 1.0) + rng.normal(0, 0.05) for g in group]
    rows = [{"file": f"s{i}", "group": group[i], "quality_alpha_pct": float(quality[i]),
             "quality_alpha_frontal_pct": float(quality[i]),
             "clinical_severity": float(rng.uniform(0, 40)), "faa": float(faa[i])}
            for i in range(n)]

    result = run_study_b(rows)
    assert not result["regression_3_quality_classifies_group"]["leakage_flag"]
    assert result["regression_2_faa_on_group_quality"]["pvalues"]["group_mdd"] < 0.05
    assert "real signal" in result["finding"], result["finding"]


def test_study_b_regression_1_skips_gracefully_without_severity():
    """A batch with no severity data at all (e.g. Mumtaz/HUSM) must still
    produce regressions 2/3, with regression 1 explicitly None + a reason --
    not a crash, and not silently dropped rows."""
    from study_b import run_study_b

    rng = np.random.default_rng(4)
    rows = _severity_rows(rng, with_severity=False)
    result = run_study_b(rows)
    assert result["regression_1_quality_on_group_severity"] is None
    assert result["regression_1_skipped_reason"] is not None
    assert result["n_with_severity"] == 0
    # regressions 2/3 still ran over every row -- severity's absence didn't
    # drop them from the rest of Study B.
    assert result["n"] == len(rows)
    assert result["regression_2_faa_on_group_quality"]["nobs"] == len(rows)
    assert result["regression_3_quality_classifies_group"]["n"] == len(rows)


def test_faa_classifiers_one_independent_result_per_pipeline():
    """Peter's ask: 4-5 classifiers, one per Study A pipeline, that don't
    share data -- each pipeline's result must come only from that
    pipeline's own FAA values, and a pipeline with too little data reports
    an explicit error rather than a fabricated number."""
    from faa_classifiers import MIN_N, classify_by_pipeline
    from study_a import PIPELINES

    rng = np.random.default_rng(5)
    rows = []
    for i in range(12):
        group = "healthy" if i % 2 == 0 else "depressed"
        for pipeline in PIPELINES:
            # plant a real signal only in "ours" so we can check pipelines
            # differ from each other, not just echo the same number
            bump = 0.5 if (pipeline == "ours" and group == "depressed") else 0.0
            rows.append({"file": f"s{i}", "group": group, "pipeline": pipeline,
                         "reference": "original", "faa": float(rng.normal()) + bump})

    result = classify_by_pipeline(rows)
    assert set(result.keys()) == set(PIPELINES)
    for pipeline in PIPELINES:
        r = result[pipeline]
        assert "error" not in r, r
        assert r["n"] == 12

    # too-small subsample -> explicit error, not a silently-computed number
    tiny = [r for r in rows if r["file"] in ("s0", "s1")]
    assert len(tiny) // len(PIPELINES) < MIN_N  # sanity: this really is too small
    result_tiny = classify_by_pipeline(tiny)
    assert all("error" in v for v in result_tiny.values())


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
