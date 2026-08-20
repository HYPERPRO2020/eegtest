"""Peter's sanity-check demo: raw / cleaned / artifacts-only, all from the
SAME real recording (not synthetic, not mixed datasets) -- does the
endpoint-aware quality index actually score them the way it should?

Split used: hard amplitude rejection at the same 150uV threshold Study A's
"generic" pipeline already uses (study_a.GENERIC_REJECT_UV). "Clean" = the
epochs that pass; "artifacts-only" = the epochs that get rejected --
concatenated on their own. A first attempt used ICA-component subtraction
(raw minus ICA-with-ocular-components-removed) as "artifacts-only" instead;
that produced a genuinely informative negative result -- the single excluded
component's sensor-space contribution scored as *cleaner* than the raw
signal under our current amplitude-threshold detectors, not dirtier (see
git history / the run log for those numbers) -- but it's a confusing demo
picture, not what Peter's asking to see. The reject/keep split below is a
much more direct, unambiguous same-dataset unclean/clean/artifacts-only
comparison, and also matches a pipeline this repo already runs (Study A's
"generic" pipeline), so it's not a new, one-off definition of "clean."

Endpoint-aware means the SAME artifact content should score very differently
depending which band you ask about: blink/EOG energy sits in delta
(0.5-4 Hz, bands.ARTIFACT_BANDS["eog"]) and broadband artifacts (pop/motion)
span the whole passband including alpha -- so watch both bands move, not
just one.

Usage: python scripts/build_validation_demo.py
Writes outputs/figures/validation_raw_clean_artifacts.png and prints the
quality-score table.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "neuroqa"))

import matplotlib.pyplot as plt
import mne
import numpy as np

from bands import EEG_BANDS
from quality_index import compute_quality
from study_a import GENERIC_REJECT_UV, load_referenced_raw, make_epochs

mne.set_log_level("ERROR")

RECORDING = REPO_ROOT / "data" / "ds003478" / "sub-001_task-Rest_run-01_eeg.set"
DEMO_CHANNEL = "Fp1"  # closest to the eyes -- where blink/EOG artifact is most visible


def quality_pct(data_uv: np.ndarray, sfreq: float, band_name: str) -> float:
    q = compute_quality(data_uv, ch_names, sfreq, EEG_BANDS[band_name])["quality"]
    return float(q.mean())


if __name__ == "__main__":
    raw, ch_names = load_referenced_raw(str(RECORDING), "original")
    epochs = make_epochs(raw)
    sfreq = epochs.info["sfreq"]

    reject_criteria = dict(eeg=GENERIC_REJECT_UV * 1e-6)
    kept = epochs.copy().drop_bad(reject=reject_criteria, verbose=False)
    bad_idx = [i for i, r in enumerate(kept.drop_log) if r]
    rejected_data = epochs.get_data()[bad_idx] if bad_idx else np.empty((0, *epochs.get_data().shape[1:]))

    n_total, n_kept, n_rejected = len(epochs), len(kept), len(bad_idx)
    print(f"Recording: {RECORDING.name}  (ds003478, real data)")
    print(f"Channels: {ch_names}")
    print(f"Epochs: {n_total} total, {n_kept} kept (<{GENERIC_REJECT_UV:.0f}uV), {n_rejected} rejected (>={GENERIC_REJECT_UV:.0f}uV)\n")

    versions_uv = {
        "unclean (all epochs)": epochs.get_data() * 1e6,
        "clean (kept epochs only)": kept.get_data() * 1e6,
        "artifacts-only (rejected epochs)": rejected_data * 1e6,
    }

    rows = []
    for label, data_uv in versions_uv.items():
        row = {"version": label}
        for band in ["delta", "alpha"]:
            row[f"quality_{band}_pct"] = quality_pct(data_uv, sfreq, band)
        rows.append(row)
        print(f"{label:38s}  quality[delta]={row['quality_delta_pct']:6.2f}%   "
              f"quality[alpha]={row['quality_alpha_pct']:6.2f}%")

    print("\nExpected pattern if the scorer is working: artifacts-only should score\n"
          "lowest, unclean in between, clean highest -- in BOTH bands, since these\n"
          "epochs were rejected on a broadband amplitude criterion, not a band-\n"
          "specific one (pop/motion detectors, which is what a 150uV spike mostly\n"
          "trips, span the whole 0.5-45Hz passband including alpha).")

    # --- waveform figure: one 4s epoch per version, one frontal channel ---
    ch_idx = ch_names.index(DEMO_CHANNEL)
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    example_epochs = {
        "unclean (all epochs)": epochs.get_data()[n_total // 2],  # an arbitrary cross-section, kept or rejected either way
        "clean (kept epochs only)": kept.get_data()[n_kept // 2] if n_kept else None,
        "artifacts-only (rejected epochs)": rejected_data[n_rejected // 2] if n_rejected else None,
    }
    t = np.arange(epochs.get_data().shape[-1]) / sfreq
    for ax, (label, ep) in zip(axes, example_epochs.items()):
        if ep is None:
            ax.text(0.5, 0.5, "(no epochs in this category)", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.plot(t, ep[ch_idx] * 1e6, linewidth=0.7)
        ax.set_ylabel(f"{label}\n(uV)", fontsize=8)
    axes[-1].set_xlabel("time within one 4s epoch (s)")
    fig.suptitle(f"{DEMO_CHANNEL}, {RECORDING.stem} -- one example epoch per category")
    fig.tight_layout()

    out_dir = REPO_ROOT / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "validation_raw_clean_artifacts.png", dpi=130)
    print(f"\nwrote {out_dir / 'validation_raw_clean_artifacts.png'}")
