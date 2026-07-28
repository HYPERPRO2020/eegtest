"""NeuroQA Step 2 — preprocessing.

Filters each ingested recording and segments it into fixed-length epochs,
ready for the (currently blocked) Step 3 artifact detectors to score. This
step does **not** reject anything — rejection is Step 3's job, once the
scientist's taxonomy doc has real thresholds. Step 2 only filters and cuts.

Pipeline per recording:
    pick the 19 10-20 channels (drop A2-A1 reference / extra aux channels)
    -> 50 Hz notch (line noise; this dataset is Malaysian mains, 50 Hz not 60 Hz)
    -> 0.5-45 Hz bandpass
    -> 4-second epochs, 50% overlap (2-second step)

Usage (from repo root, after ingest.py):
    .venv/Scripts/python.exe neuroqa/preprocess.py
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np

from ingest import EXPECTED_CHANNELS, clean_channel_name

# pandas is only needed by main()'s manifest CSV I/O, not by preprocess_file()
# itself -- imported lazily there so the webapp's import of CHANNEL_ORDER/
# EPOCH_SEC/etc. doesn't drag pandas (67MB) in.

mne.set_log_level("ERROR")

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent / "outputs"
EPOCH_DIR = OUT_DIR / "epochs"

L_FREQ, H_FREQ = 0.5, 45.0
LINE_FREQ = 50.0  # Malaysia mains — NOT 60 Hz, see task-sheet correction
EPOCH_SEC = 4.0
OVERLAP_SEC = 2.0  # 50% overlap

# Fixed channel order so every recording's epoch array lines up the same way.
CHANNEL_ORDER = sorted(EXPECTED_CHANNELS)


def preprocess_file(path: str) -> tuple[np.ndarray, float]:
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    raw.rename_channels({ch: clean_channel_name(ch) for ch in raw.ch_names})
    raw.pick(CHANNEL_ORDER)  # drops A2-A1 reference + any extra aux channels
    raw.reorder_channels(CHANNEL_ORDER)
    raw.notch_filter(LINE_FREQ, verbose=False)
    raw.filter(L_FREQ, H_FREQ, verbose=False)

    epochs = mne.make_fixed_length_epochs(
        raw, duration=EPOCH_SEC, overlap=OVERLAP_SEC, preload=True, verbose=False,
    )
    data_uv = epochs.get_data() * 1e6  # MNE returns volts; store microvolts (QC thresholds are in uV)
    return data_uv, raw.info["sfreq"]


def main():
    import pandas as pd

    manifest_path = OUT_DIR / "ingest_manifest.csv"
    if not manifest_path.exists():
        raise SystemExit("run neuroqa/ingest.py first — outputs/ingest_manifest.csv is missing")
    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest.load_error.isna()] if "load_error" in manifest else manifest

    EPOCH_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, r in manifest.iterrows():
        try:
            data, sfreq = preprocess_file(r["path"])
        except Exception as e:
            print(f"  [ERROR] {r['file']}: {e}")
            rows.append({"file": r["file"], "group": r["group"], "subject": r["subject"],
                         "condition": r["condition"], "n_epochs": 0, "error": str(e)})
            continue

        out_name = Path(r["file"]).stem + ".npz"
        np.savez_compressed(
            EPOCH_DIR / out_name,
            data=data.astype(np.float32),  # microvolts
            ch_names=np.array(CHANNEL_ORDER),
            sfreq=sfreq,
        )
        rows.append({
            "file": r["file"], "group": r["group"], "subject": r["subject"],
            "condition": r["condition"], "n_epochs": data.shape[0],
            "epoch_sec": EPOCH_SEC, "overlap_sec": OVERLAP_SEC,
            "n_channels": data.shape[1], "sfreq": sfreq, "error": None,
        })
        print(f"  {r['file']:28s} -> {data.shape[0]:4d} epochs "
              f"({data.shape[0] * (EPOCH_SEC - OVERLAP_SEC) + OVERLAP_SEC:.0f}s covered)")

    summary = pd.DataFrame(rows)
    out_path = OUT_DIR / "preprocess_summary.csv"
    summary.to_csv(out_path, index=False)

    n_ok = int((summary.n_epochs > 0).sum())
    print(f"\npreprocessed {n_ok}/{len(summary)} recordings, "
          f"{int(summary.n_epochs.sum())} epochs total")
    print(f"epoch arrays -> {EPOCH_DIR}/  (one .npz per recording)")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
