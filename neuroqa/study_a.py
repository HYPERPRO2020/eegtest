"""NeuroQA Study A — how much does FAA move with the preprocessing choice?

Computes FAA for the same recordings under 5 preprocessing pipelines x 2
reference schemes (10 combinations per recording), then reports the spread
of FAA across those 10 choices, per subject. The point isn't which pipeline
is "best" -- it's how much an analysis conclusion (FAA) would change
depending on an arbitrary-looking preprocessing choice, and where "ours"
(the endpoint-aware quality index used as continuous epoch weights, instead
of a hard reject/keep decision) lands relative to the other four.

Pipelines:
  raw       -- filter only, no rejection, flat average over all epochs.
  ica       -- ICA-clean ocular components (no real EOG channel in this
               dataset, so mne.preprocessing.ICA.find_bads_eog uses Fp1/Fp2
               as frontal proxies), then flat average over all epochs.
  generic   -- fixed 150 uV peak-to-peak epoch rejection (same order of
               magnitude threshold as v1/src/features.py and
               artifact_detectors.py's own placeholder thresholds), flat
               average over kept epochs.
  autoreject -- autoreject.AutoReject, per-channel/epoch adaptive rejection
               and interpolation, flat average over the cleaned epochs.
  ours      -- no rejection; every epoch kept but weighted by
               quality_index.py's alpha-endpoint quality at F3/F4
               respectively (continuous weighting instead of a hard cutoff).

Reference schemes:
  original  -- whatever reference the recording ships with (this dataset's
               channels are named "...-LE", i.e. linked-ears reference
               applied at acquisition -- see ingest.py's clean_channel_name).
  average   -- re-referenced to the average of the 19 picked channels.

Runs on a fixed-size stratified subsample (N_PER_GROUP recordings per
group, EC condition only, deduplicated by md5) rather than the full 120
recordings -- ICA + autoreject are the slow steps here (~10-20s each), and
this question (how much does FAA move across pipeline choices) is about
within/across-subject spread, not statistical power, so a bounded subsample
is the right tradeoff. Raise N_PER_GROUP to widen it.

Usage (from repo root, after ingest.py):
    .venv/Scripts/python.exe neuroqa/study_a.py
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd
from autoreject import AutoReject

from bands import EEG_BANDS
from faa import compute_faa
from preprocess import CHANNEL_ORDER, EPOCH_SEC, H_FREQ, L_FREQ, LINE_FREQ
from quality_index import compute_quality

mne.set_log_level("ERROR")

OUT_DIR = Path(__file__).resolve().parent / "outputs"
FIG_DIR = OUT_DIR / "figures"
OVERLAP_SEC = 2.0

N_PER_GROUP = 6  # recordings per group -- see module docstring for why bounded
PIPELINES = ["raw", "ica", "generic", "autoreject", "ours"]
REFERENCES = ["original", "average"]
GENERIC_REJECT_UV = 150.0

# autoreject needs real channel positions to do its spatial interpolation.
# This dataset's channel names use the older 10-20 labels (T3/T4/T5/T6);
# MNE's standard_1020 montage uses the modern equivalents (T7/T8/P7/P8).
# Renamed locally, only within this script, so the rest of the pipeline
# (preprocess.py/score.py/quality_index.py, which never touch autoreject)
# is untouched -- F3/F4 (all FAA needs) and FRONTAL_CHANNELS (Fp1/Fp2/F7/F8)
# aren't affected either way.
RENAME_1020 = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
STUDY_CH_ORDER = [RENAME_1020.get(ch, ch) for ch in CHANNEL_ORDER]


def load_referenced_raw(path: str, reference: str) -> mne.io.Raw:
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    from ingest import clean_channel_name
    raw.rename_channels({ch: clean_channel_name(ch) for ch in raw.ch_names})
    raw.pick(CHANNEL_ORDER)
    raw.reorder_channels(CHANNEL_ORDER)
    raw.rename_channels(RENAME_1020)
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"),
                     match_case=False, on_missing="warn", verbose=False)
    if reference == "average":
        raw.set_eeg_reference("average", verbose=False)
    # "original": no set_eeg_reference call -- keep the recording's own
    # (linked-ears, "-LE") reference as loaded.
    raw.notch_filter(LINE_FREQ, verbose=False)
    raw.filter(L_FREQ, H_FREQ, verbose=False)
    return raw


def make_epochs(raw: mne.io.Raw) -> mne.Epochs:
    return mne.make_fixed_length_epochs(
        raw, duration=EPOCH_SEC, overlap=OVERLAP_SEC, preload=True, verbose=False,
    )


def ica_clean(raw: mne.io.Raw) -> mne.io.Raw:
    """ICA-clean ocular components. No real EOG channel in this dataset, so
    find_bads_eog uses Fp1/Fp2 as frontal proxies (MNE supports this
    directly via the ch_name argument)."""
    ica = mne.preprocessing.ICA(n_components=15, random_state=97, max_iter="auto", verbose=False)
    ica.fit(raw, verbose=False)
    try:
        eog_idx, _ = ica.find_bads_eog(raw, ch_name=["Fp1", "Fp2"], verbose=False)
        ica.exclude = eog_idx
    except Exception:
        ica.exclude = []
    cleaned = raw.copy()
    ica.apply(cleaned, verbose=False)
    return cleaned


def faa_pipeline_raw(epochs: mne.Epochs, sfreq: float) -> float:
    data_uv = epochs.get_data() * 1e6
    return compute_faa(data_uv, STUDY_CH_ORDER, sfreq)["faa"]


def faa_pipeline_ica(raw: mne.io.Raw, sfreq: float) -> float:
    raw_clean = ica_clean(raw)
    epochs = make_epochs(raw_clean)
    return faa_pipeline_raw(epochs, sfreq)


def faa_pipeline_generic(epochs: mne.Epochs, sfreq: float) -> float:
    kept = epochs.copy().drop_bad(reject=dict(eeg=GENERIC_REJECT_UV * 1e-6), verbose=False)
    if len(kept) == 0:
        kept = epochs  # degenerate fallback: everything rejected, use unrejected set
    return faa_pipeline_raw(kept, sfreq)


def faa_pipeline_autoreject(epochs: mne.Epochs, sfreq: float) -> float:
    ar = AutoReject(random_state=11, n_jobs=1, verbose=False)
    kept = ar.fit_transform(epochs.copy())
    if len(kept) == 0:
        kept = epochs
    return faa_pipeline_raw(kept, sfreq)


def faa_pipeline_ours(epochs: mne.Epochs, sfreq: float) -> float:
    data_uv = epochs.get_data() * 1e6
    quality = compute_quality(data_uv, STUDY_CH_ORDER, sfreq, EEG_BANDS["alpha"])["quality"]
    i3, i4 = STUDY_CH_ORDER.index("F3"), STUDY_CH_ORDER.index("F4")
    return compute_faa(data_uv, STUDY_CH_ORDER, sfreq,
                        weights_f3=quality[:, i3], weights_f4=quality[:, i4])["faa"]


def run_all_pipelines(path: str) -> dict[tuple[str, str], float]:
    out = {}
    for reference in REFERENCES:
        raw = load_referenced_raw(path, reference)
        sfreq = raw.info["sfreq"]
        epochs = make_epochs(raw)
        out[("raw", reference)] = faa_pipeline_raw(epochs, sfreq)
        out[("ica", reference)] = faa_pipeline_ica(raw, sfreq)
        out[("generic", reference)] = faa_pipeline_generic(epochs, sfreq)
        out[("autoreject", reference)] = faa_pipeline_autoreject(epochs, sfreq)
        out[("ours", reference)] = faa_pipeline_ours(epochs, sfreq)
    return out


def select_subsample(manifest: pd.DataFrame) -> pd.DataFrame:
    eligible = manifest[
        manifest.load_error.isna() & manifest.channels_ok & manifest.sfreq_ok
        & (manifest.condition == "EC") & (~manifest.is_duplicate)
    ]
    eligible = eligible.drop_duplicates(subset="md5")
    return pd.concat(
        [eligible[eligible.group == g].head(N_PER_GROUP) for g in sorted(eligible.group.unique())]
    ).reset_index(drop=True)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(OUT_DIR / "ingest_manifest.csv")
    subsample = select_subsample(manifest)
    print(f"running Study A on {len(subsample)} recordings "
          f"({subsample.group.value_counts().to_dict()})")

    rows = []
    for _, r in subsample.iterrows():
        print(f"  {r['file']} ...")
        results = run_all_pipelines(r["path"])
        for (pipeline, reference), faa_val in results.items():
            rows.append({
                "file": r["file"], "group": r["group"], "subject": r["subject"],
                "pipeline": pipeline, "reference": reference, "faa": faa_val,
            })

    long_df = pd.DataFrame(rows)
    long_path = OUT_DIR / "study_a_faa_long.csv"
    long_df.to_csv(long_path, index=False)

    wide = long_df.pivot_table(index=["file", "group", "subject"],
                                columns=["pipeline", "reference"], values="faa")
    wide.columns = [f"{p}_{r}" for p, r in wide.columns]
    wide["faa_range"] = wide.max(axis=1) - wide.min(axis=1)
    wide["faa_std"] = wide.std(axis=1)
    wide_path = OUT_DIR / "study_a_faa_wide.csv"
    wide.reset_index().to_csv(wide_path, index=False)

    print(f"\nwrote {long_path}")
    print(f"wrote {wide_path}")
    print("\nFAA spread across the 10 pipeline x reference combinations, per subject:")
    print(wide[["faa_range", "faa_std"]].describe().to_string())

    # Parallel-coordinates plot: one line per subject across the 10 combos.
    combo_cols = [c for c in wide.columns if c not in ("faa_range", "faa_std")]
    fig, ax = plt.subplots(figsize=(11, 5))
    wide_reset = wide.reset_index()
    for _, row in wide_reset.iterrows():
        color = "tab:blue" if row["group"] == "H" else "tab:red"
        ax.plot(combo_cols, [row[c] for c in combo_cols], color=color, alpha=0.5, marker="o", markersize=3)
    ax.set_xticklabels(combo_cols, rotation=45, ha="right")
    ax.set_ylabel("FAA")
    ax.set_title("FAA per subject across 5 pipelines x 2 reference schemes\n(blue=H, red=MDD)")
    fig.tight_layout()
    fig_path = FIG_DIR / "study_a_faa_spread.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
