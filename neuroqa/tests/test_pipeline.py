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


def test_score_and_faa_reports_raw_faa_alongside_quality_weighted_faa(tmp_path):
    """score_and_faa must return both the quality-weighted `faa` and an
    unweighted `faa_raw`, computed from the same data -- study_b.py's
    regression_2_raw needs faa_raw as a DV independent of the quality
    weights that also feed the quality predictor (see study_b.py docstring
    on why regression_2 alone is partly circular)."""
    from pipeline import score_and_faa

    raw = make_synthetic_raw()
    path = tmp_path / "sub_rawfaa.fif"
    raw.save(str(path), verbose=False)

    result = score_and_faa(path)
    assert "faa_raw" in result
    assert isinstance(result["faa_raw"], float)


def test_study_b_regression_2_raw_uses_unweighted_faa_as_dv():
    """regression_2_raw's dependent variable must be faa_raw, not the
    quality-weighted faa regression_2 uses -- otherwise it isn't actually an
    independent robustness check (see study_b.py module docstring)."""
    from study_b import regression_2_raw, rows_to_frame

    rng = np.random.default_rng(5)
    n = 20
    rows = [
        {"file": f"s{i}", "group": "healthy" if i % 2 == 0 else "depressed",
         "quality_alpha_pct": float(rng.uniform(60, 100)),
         "faa": 999.0,  # deliberately garbage -- must NOT be what gets fit
         "faa_raw": float(rng.normal())}
        for i in range(n)
    ]
    df = rows_to_frame(rows, "quality_alpha_pct")
    df["faa_raw"] = [r["faa_raw"] for r in rows]
    m = regression_2_raw(df, "quality_alpha_pct")
    # If faa (999.0 constant-ish) had leaked in, the fit would be degenerate/
    # near-perfect on a constant; fitting real (small, ~N(0,1)) faa_raw
    # values instead gives residual variance in a normal range.
    assert 0.01 < m.mse_resid < 10.0


def test_study_b_mediation_analysis_reports_acme_ade_total_with_cis():
    """mediation_analysis must return ACME (indirect, quality-mediated
    effect), ADE (direct effect), and total effect, each with a bootstrapped
    CI, plus proportion_mediated -- the purpose-built tool for the
    mediator-vs-confounder question regression_2's bare coefficient check
    can't resolve (Table 2 Fallacy, see module docstring)."""
    from study_b import mediation_analysis, rows_to_frame

    rng = np.random.default_rng(9)
    n = 40
    group = ["healthy" if i % 2 == 0 else "depressed" for i in range(n)]
    # quality is caused by group (mediator setup), and FAA is caused by
    # quality (not directly by group) -- ACME should dominate, ADE should
    # be small/non-significant, ~fully mediated.
    quality = [rng.normal(90, 3) if g == "healthy" else rng.normal(70, 3) for g in group]
    faa = [0.02 * q + rng.normal(0, 0.05) for q in quality]
    rows = [{"file": f"s{i}", "group": group[i], "quality_alpha_pct": float(quality[i]),
             "faa": float(faa[i])} for i in range(n)]
    df = rows_to_frame(rows, "quality_alpha_pct")

    result = mediation_analysis(df, "quality_alpha_pct", n_boot=200, seed=1)
    for key in ("acme", "ade", "total_effect"):
        assert set(result[key].keys()) == {"point", "ci_lo", "ci_hi"}
    assert result["n_boot_valid"] > 0
    # Constructed as a near-full-mediation scenario: ACME should be the
    # dominant component of the total effect.
    assert abs(result["acme"]["point"]) > abs(result["ade"]["point"])


def test_study_b_leakage_flag_uses_ci_not_point_estimate_threshold():
    """leakage_flag must be derived from auc_ci_lo > 0.5 (the bootstrap CI
    already computed), not a bare `auc > 0.65` point-estimate cutoff -- see
    study_b.py's regression_3 comment for why 0.65 is an arbitrary,
    chance-disconnected threshold that both under- and over-flags relative
    to the CI. Both directions of the old bug are checked with fixed seeds
    against real (non-mocked) data so this regresses if anyone reverts to
    the point-estimate rule."""
    from study_b import regression_3, rows_to_frame

    # Case 1: real, moderate quality-group separation at n=200 -- point AUC
    # lands at 0.6433 (old rule: NOT flagged, since 0.6433 < 0.65) but the
    # CI's lower bound (0.568) clears 0.5 comfortably (new rule: flagged).
    # The old threshold would have missed a real, CI-confirmed leak here.
    rng = np.random.default_rng(0)
    n = 200
    group = ["healthy" if i % 2 == 0 else "depressed" for i in range(n)]
    quality = [rng.normal(80, 8) if g == "healthy" else rng.normal(83, 8) for g in group]
    rows = [{"file": f"s{i}", "group": group[i], "quality_alpha_pct": float(quality[i]),
             "faa": float(rng.normal())} for i in range(n)]
    df = rows_to_frame(rows, "quality_alpha_pct")
    r = regression_3(df, "quality_alpha_pct")
    assert r["auc"] < 0.65
    assert r["auc_ci_lo"] > 0.5
    assert r["leakage_flag"] is True

    # Case 2: quality independent of group by construction, small n=14 --
    # point AUC drifts to 0.735 by chance (old rule: flagged, since
    # 0.735 > 0.65) but the CI spans [0.4, 1.0], comfortably crossing 0.5
    # (new rule: not flagged). The old threshold would have raised a false
    # leakage warning here.
    rng2 = np.random.default_rng(4)
    n2 = 14
    group2 = ["healthy" if i % 2 == 0 else "depressed" for i in range(n2)]
    quality2 = [rng2.normal(80, 8) for _ in group2]
    rows2 = [{"file": f"s{i}", "group": group2[i], "quality_alpha_pct": float(quality2[i]),
              "faa": float(rng2.normal())} for i in range(n2)]
    df2 = rows_to_frame(rows2, "quality_alpha_pct")
    r2 = regression_3(df2, "quality_alpha_pct")
    assert r2["auc"] > 0.65
    assert r2["auc_ci_lo"] < 0.5
    assert r2["leakage_flag"] is False


def test_study_b_regression_1_enforces_total_n_floor_not_just_per_group():
    """A batch with only MIN_PER_GROUP_FOR_SEVERITY (2) per group technically
    passes the old per-group-only check but leaves a 3-parameter OLS fit
    (const, group, severity) with 1 residual df on 4 total observations --
    too thin to trust. regression_1 must now skip this case via the added
    total-n floor (MIN_N_FOR_OLS), not silently return a p-value from it."""
    from study_b import MIN_N_FOR_OLS, run_study_b

    rng = np.random.default_rng(6)
    rows = [
        {"file": f"s{i}", "group": "healthy" if i < 2 else "depressed",
         "quality_alpha_pct": float(rng.uniform(60, 100)),
         "quality_alpha_frontal_pct": float(rng.uniform(60, 100)),
         "clinical_severity": float(rng.uniform(0, 63)), "faa": float(rng.normal())}
        for i in range(4)  # 2 per group -- passes MIN_PER_GROUP_FOR_SEVERITY, fails MIN_N_FOR_OLS
    ]
    assert len(rows) < MIN_N_FOR_OLS
    result = run_study_b(rows)
    assert result["regression_1_quality_on_group_severity"] is None
    assert result["regression_1_skipped_reason"] is not None


def test_study_c_inject_artifact_is_noop_at_dose_zero_and_confined_to_frontal_channels():
    """dose=0.0 must return data unchanged (the no-injection baseline point
    dose_response fits against), and any dose>0 must only touch
    FRONTAL_INJECT_CHANNELS -- a non-frontal channel like O1 has no business
    changing when Study C injects contamination at F3/F4/Fp1/Fp2."""
    from study_c import inject_artifact

    rng = np.random.default_rng(2)
    sfreq = 256.0
    ch_names = ["F3", "F4", "Fp1", "Fp2", "O1"]
    data = rng.normal(0, 3.0, size=(20, len(ch_names), int(4 * sfreq)))

    zero = inject_artifact(data, ch_names, sfreq, "eog", 0.0, seed=1)
    assert np.array_equal(zero, data)

    out = inject_artifact(data, ch_names, sfreq, "eog", 2.0, seed=1)
    o1 = ch_names.index("O1")
    f3 = ch_names.index("F3")
    assert np.array_equal(out[:, o1, :], data[:, o1, :])
    assert not np.array_equal(out[:, f3, :], data[:, f3, :])


def test_study_c_inject_artifact_is_deterministic():
    """Same call, same seed, must reproduce bit-identical output -- Study C
    results need to be reproducible the same way the rest of the pipeline is
    (see pipeline.py's module docstring on seeded randomness)."""
    from study_c import inject_artifact

    rng = np.random.default_rng(2)
    sfreq = 256.0
    ch_names = ["F3", "F4", "Fp1", "Fp2"]
    data = rng.normal(0, 3.0, size=(10, len(ch_names), int(4 * sfreq)))

    first = inject_artifact(data, ch_names, sfreq, "emg_alpha_overlap", 2.0, seed=1)
    second = inject_artifact(data, ch_names, sfreq, "emg_alpha_overlap", 2.0, seed=1)
    assert np.array_equal(first, second)


def test_study_c_dose_scales_one_fixed_noise_shape_not_a_fresh_draw_per_dose():
    """Injected noise at a given channel must be the SAME waveform shape
    across every dose level for a fixed seed -- only its amplitude should
    scale with dose. Otherwise each dose point in a dose-response curve
    would sit on an independent random realization instead of tracing a
    function of dose, which is what a "dose-response" claim requires."""
    from study_c import inject_artifact

    rng = np.random.default_rng(3)
    sfreq = 256.0
    ch_names = ["F3", "F4"]
    data = rng.normal(0, 3.0, size=(5, len(ch_names), int(4 * sfreq)))
    f4 = ch_names.index("F4")

    at_1x = inject_artifact(data, ch_names, sfreq, "eog", 1.0, seed=7)
    at_2x = inject_artifact(data, ch_names, sfreq, "eog", 2.0, seed=7)
    added_1x = at_1x[:, f4, :] - data[:, f4, :]
    added_2x = at_2x[:, f4, :] - data[:, f4, :]
    # If the shape is reused, added_2x must be (near) exactly 2x added_1x,
    # sample for sample -- not just "bigger" in some general sense.
    ratio = added_2x / np.where(added_1x != 0, added_1x, np.nan)
    assert np.nanstd(ratio) < 1e-6
    assert abs(np.nanmean(ratio) - 2.0) < 1e-6


def test_study_c_spectral_specificity_alpha_overlap_vs_no_overlap():
    """The whole point of including emg_no_overlap as a control condition:
    alpha-overlapping injection (8-13Hz) must raise measured alpha-band
    power far more than equal-dose non-overlapping injection (20-45Hz) does,
    on the same near-zero baseline. If this weren't true, Study C's
    spectral-specificity comparison wouldn't actually test what its module
    docstring claims it tests."""
    from faa import alpha_power
    from study_c import inject_artifact

    rng = np.random.default_rng(8)
    sfreq = 256.0
    ch_names = ["F3", "F4", "Fp1", "Fp2"]
    data = rng.normal(0, 1.0, size=(10, len(ch_names), int(4 * sfreq)))  # small, near-flat baseline

    overlap = inject_artifact(data, ch_names, sfreq, "emg_alpha_overlap", dose=3.0, seed=5)
    no_overlap = inject_artifact(data, ch_names, sfreq, "emg_no_overlap", dose=3.0, seed=5)

    baseline_alpha = alpha_power(data, ch_names, sfreq).mean()
    overlap_alpha = alpha_power(overlap, ch_names, sfreq).mean()
    no_overlap_alpha = alpha_power(no_overlap, ch_names, sfreq).mean()

    assert overlap_alpha > baseline_alpha * 5  # alpha-overlapping injection clearly raises alpha power
    assert no_overlap_alpha < overlap_alpha / 5  # non-overlapping injection raises it far less


def test_study_c_dose_response_matches_faa_at_dose_zero_and_is_deterministic():
    """dose_response's dose=0.0 point must equal an ordinary unweighted
    compute_faa call on the untouched data (the baseline point everything
    else is compared against), and repeat calls with the same seed must be
    bit-identical."""
    from faa import compute_faa
    from study_c import dose_response

    rng = np.random.default_rng(4)
    sfreq = 256.0
    ch_names = ["F3", "F4", "Fp1", "Fp2"]
    data = rng.normal(0, 3.0, size=(15, len(ch_names), int(4 * sfreq)))

    result = dose_response(data, ch_names, sfreq, "eog", doses=(0.0, 1.0, 2.0), seed=2)
    expected_baseline = compute_faa(data, ch_names, sfreq)["faa"]
    assert abs(result["points"][0]["faa"] - expected_baseline) < 1e-9

    result2 = dose_response(data, ch_names, sfreq, "eog", doses=(0.0, 1.0, 2.0), seed=2)
    assert result == result2


def test_run_study_c_summarizes_across_recordings(tmp_path):
    """run_study_c must run every injection kind across every recording and
    summarize each kind's population-level slope with a subject-level
    bootstrap CI -- an end-to-end smoke test of the module's actual entry
    point, not just its internal pieces."""
    from study_c import INJECTION_BANDS, run_study_c

    paths = []
    for i in range(2):
        raw = make_synthetic_raw(seed=i)
        path = tmp_path / f"study_c_sub{i}.fif"
        raw.save(str(path), verbose=False)
        paths.append(path)

    result = run_study_c(paths, doses=(0.0, 1.0, 2.0), n_boot=50)
    assert result["n_recordings"] == 2
    assert len(result["per_subject"]) == 2
    for kind in INJECTION_BANDS:
        s = result["summary"][kind]
        assert s["n_subjects"] == 2
        assert {"mean_slope", "slope_ci_lo", "slope_ci_hi", "nonzero_slope"} <= s.keys()


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


def test_spectral_composition_detects_planted_peak_and_no_peak():
    """A synthetic spectrum with a clear planted alpha oscillation must be
    fit with peak_present=True, a CF near the planted frequency, and an
    oscillatory_share strictly between 0 and 1; a pure-aperiodic spectrum
    (no oscillation at all) must report peak_present=False, all periodic
    fields NaN, and oscillatory_share exactly 0.0 (not NaN -- see module
    docstring: these two use deliberately different no-peak conventions)."""
    from fooof.sim.gen import gen_power_spectrum
    from spectral_composition import fit_fooof

    freqs, powers = gen_power_spectrum([1, 40], [0, 1], [10, 0.4, 1.2])
    with_peak = fit_fooof(freqs, powers, aperiodic_mode="fixed")
    assert with_peak["peak_present"] is True
    assert abs(with_peak["alpha_cf"] - 10.0) < 1.0
    assert 0.0 < with_peak["oscillatory_share"] < 1.0
    assert with_peak["low_quality_fit"] is False  # clean synthetic spectrum, should fit well

    freqs2, powers2 = gen_power_spectrum([1, 40], [0, 1], [])
    no_peak = fit_fooof(freqs2, powers2, aperiodic_mode="fixed")
    assert no_peak["peak_present"] is False
    assert np.isnan(no_peak["alpha_cf"]) and np.isnan(no_peak["alpha_pw"]) and np.isnan(no_peak["alpha_bw"])
    assert no_peak["oscillatory_share"] == 0.0


def test_spectral_composition_flags_low_quality_fit():
    """A spectrum that genuinely doesn't follow an aperiodic+peaks shape
    (confirmed on real ds003478 recordings -- EMG bleed near 40 Hz and
    filter-edge effects near 1 Hz both routinely break the power-law
    assumption over the spec's full 1-40 Hz fit range) must be flagged
    low_quality_fit=True via MIN_R_SQUARED, not silently reported as if the
    decomposition were trustworthy (spec Sec. 5: 'flag ... and report how
    many were excluded and why')."""
    from spectral_composition import MIN_R_SQUARED, fit_fooof

    rng = np.random.default_rng(0)
    freqs = np.arange(1, 40, 0.5)
    # Deliberately non-power-law: flat-ish with a rising tail at the high
    # end (mimics the real EMG-bleed pattern) plus a dip-then-rise at the
    # low end (mimics real filter-edge effects) -- neither a clean 1/f decay.
    powers = np.full_like(freqs, 500.0)
    powers[freqs < 4] = np.array([1500, 1700, 1350, 700, 130, 90])[: (freqs < 4).sum()]
    powers[freqs > 34] = np.linspace(30, 1100, (freqs > 34).sum())
    powers += rng.normal(0, 20, size=freqs.shape)
    powers = np.clip(powers, 1.0, None)

    r = fit_fooof(freqs, powers, aperiodic_mode="fixed")
    assert r["r_squared"] < MIN_R_SQUARED, r["r_squared"]
    assert r["low_quality_fit"] is True


def test_spectral_composition_both_aperiodic_modes_recover_planted_exponent():
    """Both aperiodic_mode='fixed' and 'knee' (spec Sec. 6 robustness check)
    must run without crashing and recover an exponent close to the planted
    value on a clean synthetic spectrum."""
    from fooof.sim.gen import gen_power_spectrum
    from spectral_composition import fit_fooof

    freqs, powers = gen_power_spectrum([1, 40], [0, 1.5], [10, 0.4, 1.2])
    for mode in ("fixed", "knee"):
        r = fit_fooof(freqs, powers, aperiodic_mode=mode)
        assert abs(r["exponent"] - 1.5) < 0.3, (mode, r["exponent"])
        assert r["r_squared"] > 0.9


def test_faa_variants_classic_faa_sign_follows_planted_asymmetry():
    """classic_faa's sign must follow the planted F3/F4 alpha-amplitude
    asymmetry, matching faa.compute_faa's existing directional-sanity
    convention (test_faa_sign_follows_planted_asymmetry above)."""
    from spectral_composition import faa_variants, spectral_composition_for_recording

    raw_f4_bigger = make_synthetic_raw(alpha_amp_f3=6.0, alpha_amp_f4=14.0, seed=2)
    data = epoch(raw_f4_bigger)
    comp = spectral_composition_for_recording(data, raw_f4_bigger.ch_names, SFREQ)
    variants = faa_variants(comp["F3"], comp["F4"])
    assert variants["classic_faa"] > 0  # F4 louder alpha -> positive classic FAA

    raw_f3_bigger = make_synthetic_raw(alpha_amp_f3=14.0, alpha_amp_f4=6.0, seed=2)
    data2 = epoch(raw_f3_bigger)
    comp2 = spectral_composition_for_recording(data2, raw_f3_bigger.ch_names, SFREQ)
    variants2 = faa_variants(comp2["F3"], comp2["F4"])
    assert variants2["classic_faa"] < 0


def test_faa_variants_periodic_only_faa_is_nan_not_zero_when_no_peak():
    """periodic_only_faa must be NaN (not 0.0) whenever either channel lacks
    a detected peak -- spec Sec. 5: 'do not impute 0'. Constructed directly
    from fit_fooof-shaped dicts rather than real data, since reliably
    forcing a real synthetic recording to have zero detected peaks at both
    channels is not the point of this test -- the branch logic is."""
    from spectral_composition import faa_variants

    has_peak = {"peak_present": True, "alpha_pw": 0.5, "exponent": 1.0, "offset": 0.0, "classic_band_power": 1.0}
    no_peak = {"peak_present": False, "alpha_pw": float("nan"), "exponent": 1.0, "offset": 0.0, "classic_band_power": 1.0}

    v = faa_variants(no_peak, has_peak)
    assert v["both_peaks_present"] is False
    assert np.isnan(v["periodic_only_faa"])

    v2 = faa_variants(has_peak, has_peak)
    assert v2["both_peaks_present"] is True
    assert v2["periodic_only_faa"] == 0.0  # equal PW on both sides


def test_spectral_composition_for_recording_only_targets_f3_f4_by_default():
    """v1 targets F3/F4 only (spec Sec. 3: 'keep all channels stored for
    later; F3/F4 are the v1 target') -- the default `channels` argument
    must not silently expand to every channel in the recording."""
    from spectral_composition import spectral_composition_for_recording

    raw = make_synthetic_raw(seed=3)
    data = epoch(raw)
    comp = spectral_composition_for_recording(data, raw.ch_names, SFREQ)
    assert set(comp.keys()) == {"F3", "F4"}


def test_quality_weighted_psd_downweights_a_contaminated_epoch():
    """The whole reason quality_weighted_psd replaced a flat average (see
    module docstring): a handful of badly-contaminated epochs must not be
    allowed to distort the averaged spectrum the way a flat mean would.
    Plant one epoch with a huge broadband amplitude blowout and confirm the
    quality-weighted PSD stays close to the clean epochs' own spectrum,
    while a flat mean is pulled noticeably further away."""
    from spectral_composition import avg_psd, quality_weighted_psd

    raw = make_synthetic_raw(seed=5)
    data = epoch(raw)  # (n_epochs, n_channels, n_samples), uV
    contaminated = data.copy()
    blown_epoch = 0
    # A constant DC offset would be neutralized by welch()'s own per-segment
    # detrending (its default removes the mean before computing the
    # periodogram) -- broadband noise isn't, and is what a real muscle/
    # movement burst actually looks like spectrally.
    rng = np.random.default_rng(6)
    contaminated[blown_epoch] += rng.normal(0, 200.0, size=contaminated.shape[1:])

    freqs, flat_psd = avg_psd(contaminated, raw.ch_names, SFREQ)  # flat mean, no weights
    freqs_w, weighted_psd = quality_weighted_psd(contaminated, raw.ch_names, SFREQ)
    freqs_clean, clean_psd = avg_psd(data, raw.ch_names, SFREQ)  # ground truth: no contamination at all

    i3 = raw.ch_names.index("F3")
    # Low-frequency bins carry most of a broadband amplitude blowout's added
    # power -- compare there, where the distortion should be starkest.
    low = (freqs >= 1) & (freqs <= 4)
    flat_dist = np.abs(np.log(flat_psd[i3][low]) - np.log(clean_psd[i3][low])).mean()
    weighted_dist = np.abs(np.log(weighted_psd[i3][low]) - np.log(clean_psd[i3][low])).mean()
    assert weighted_dist < flat_dist


def test_theta_alpha_ratio_reflects_relative_band_power():
    """theta_alpha_ratio must be small when alpha dominates theta, and grow
    when theta power is boosted relative to alpha -- a direct sanity check
    on the arousal-proxy formula (spec Sec. 4.C), not just that it runs."""
    from spectral_composition import avg_psd, theta_alpha_ratio

    raw = make_synthetic_raw(alpha_amp_f3=20.0, alpha_amp_f4=20.0, seed=7)
    data = epoch(raw)
    freqs, psd = avg_psd(data, raw.ch_names, SFREQ)
    i3 = raw.ch_names.index("F3")
    ratio_alpha_dominant = theta_alpha_ratio(freqs, psd[i3])

    # Boost this same recording's theta (4-8Hz) content directly in the PSD
    # (cheaper and more targeted than re-synthesizing a whole raw signal)
    # and confirm the ratio increases as expected.
    boosted = psd[i3].copy()
    theta_mask = (freqs >= 4) & (freqs <= 8)
    boosted[theta_mask] *= 50.0
    ratio_theta_boosted = theta_alpha_ratio(freqs, boosted)

    assert ratio_theta_boosted > ratio_alpha_dominant


def test_alpha_power_trend_sign_follows_planted_within_recording_decline():
    """alpha_power_trend's slope must be negative when alpha power is
    planted to genuinely decline across the recording, and near-zero for a
    stationary signal -- the arousal-decline proxy (spec Sec. 4.C) needs to
    actually track a real within-recording change, not just return a
    number."""
    from spectral_composition import alpha_power_trend

    rng = np.random.default_rng(8)
    n_samples = int(DURATION_SEC * SFREQ)
    t = np.arange(n_samples) / SFREQ
    ch_names = ["F3", "F4"]
    # Alpha amplitude ramps from strong to weak across the recording --
    # a real within-recording decline, not epoch-to-epoch noise.
    envelope = np.linspace(1.0, 0.05, n_samples)
    raw_data = rng.normal(0, 3e-6, size=(len(ch_names), n_samples))
    for i in range(len(ch_names)):
        raw_data[i] += 15e-6 * envelope * np.sin(2 * np.pi * 10.0 * t)
    import mne
    info = mne.create_info(ch_names, sfreq=SFREQ, ch_types="eeg")
    raw = mne.io.RawArray(raw_data, info, verbose=False)
    data = epoch(raw)

    trend = alpha_power_trend(data, ch_names, SFREQ)
    assert trend["F3"] < 0 and trend["F4"] < 0

    stationary_raw = make_synthetic_raw(seed=9)
    stationary_data = epoch(stationary_raw)
    stationary_trend = alpha_power_trend(stationary_data, stationary_raw.ch_names, SFREQ)
    assert abs(stationary_trend["F3"]) < abs(trend["F3"])


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
