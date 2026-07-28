"""NeuroQA Step 1 — ingestion.

Loads every resting-state .edf in the HUSM (Mumtaz) dataset, confirms channel
count and sampling rate, and parses group (H/MDD) and condition (EC/EO) from
the filename. TASK files (P300, not resting-state) are skipped, per spec.

The label lives in the filename, so there is no separate label file:
    H S1 EC.edf    -> group=H,   subject=1, condition=EC
    MDD S3 EO.edf  -> group=MDD, subject=3, condition=EO

Usage (from repo root):
    .venv/Scripts/python.exe neuroqa/ingest.py
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import mne
import pandas as pd

mne.set_log_level("ERROR")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "v1" / "Data"
OUT_DIR = Path(__file__).resolve().parent / "outputs"

EXPECTED_SFREQ = 256.0

# The 19 electrodes of the 10-20 system every recording is expected to carry.
# Raw files actually report 20-22 raw channels (the 19 below, plus a linked-ear
# reference "A2-A1" and, on some files, two extra unlabeled channels "23A"/"24A")
# so the correct check is "these 19 names are present", not "channel count == 19".
EXPECTED_CHANNELS = {
    "Fp1", "F3", "C3", "P3", "O1", "F7", "T3", "T5", "Fz",
    "Fp2", "F4", "C4", "P4", "O2", "F8", "T4", "T6", "Cz", "Pz",
}


def clean_channel_name(ch: str) -> str:
    # "EEG Fp1-LE" -> "Fp1"
    return ch.replace("EEG ", "").replace("-LE", "").strip()

# Matches "H S1 EC.edf", "MDD S24 EC.edf" (also tolerates a numeric
# acquisition-id prefix and doubled spaces seen in a few files on disk).
FILENAME_RE = re.compile(
    r"^(?:\d+_)?(?P<group>H|MDD)\s+S(?P<subject>\d+)\s+(?P<condition>EC|EO|TASK)\.edf$",
    re.IGNORECASE,
)


def parse_filename(name: str) -> dict | None:
    m = FILENAME_RE.match(name)
    if not m:
        return None
    return {
        "group": m.group("group").upper(),
        "subject": int(m.group("subject")),
        "condition": m.group("condition").upper(),
    }


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def ingest() -> pd.DataFrame:
    files = sorted(DATA_DIR.glob("*.edf"))
    rows = []
    for path in files:
        parsed = parse_filename(path.name)
        if parsed is None:
            print(f"  [skip] unparseable filename: {path.name}")
            continue
        if parsed["condition"] == "TASK":
            continue  # P300 task, not resting-state — out of scope per spec

        row = {"file": path.name, **parsed, "path": str(path)}
        try:
            raw = mne.io.read_raw_edf(str(path), preload=False, verbose=False)
        except Exception as e:
            row.update(load_error=str(e), n_channels=None, sfreq=None,
                       channels_ok=False, missing_channels="", sfreq_ok=False,
                       duration_sec=None, md5=None)
            rows.append(row)
            print(f"  [ERROR] {path.name}: could not load — {e}")
            continue

        n_channels = len(raw.ch_names)
        sfreq = float(raw.info["sfreq"])
        clean_names = {clean_channel_name(ch) for ch in raw.ch_names}
        missing = EXPECTED_CHANNELS - clean_names
        row.update(
            load_error=None,
            n_channels=n_channels,
            sfreq=sfreq,
            channels_ok=(len(missing) == 0),
            missing_channels=",".join(sorted(missing)) if missing else "",
            sfreq_ok=(sfreq == EXPECTED_SFREQ),
            duration_sec=round(raw.n_times / sfreq, 1),
            md5=md5sum(path),
        )
        rows.append(row)
        flag = "" if (row["channels_ok"] and row["sfreq_ok"]) else "  <-- MISMATCH"
        print(f"  {path.name:28s} {n_channels:2d} raw ch  {sfreq:.0f} Hz{flag}")

    df = pd.DataFrame(rows)
    df["is_duplicate"] = df["md5"].duplicated(keep=False) & df["md5"].notna()
    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"scanning {DATA_DIR} ...")
    df = ingest()

    n_total = len(df)
    n_ok = int((df.channels_ok & df.sfreq_ok).sum())
    n_dupe = int(df.is_duplicate.sum())

    print(f"\ningested {n_total} resting-state recordings "
          f"(H={int((df.group == 'H').sum())}, MDD={int((df.group == 'MDD').sum())})")
    print(f"  channel/sfreq mismatches : {n_total - n_ok}")
    print(f"  byte-identical duplicates: {n_dupe} files across "
          f"{df[df.is_duplicate].md5.nunique()} hash groups")
    if n_dupe:
        print("  NOTE: this dataset is known to contain duplicate recordings shared\n"
              "        across different subject labels (see v1/README.md, 'The data\n"
              "        problem found first'). Deduplicate by md5 before treating rows\n"
              "        in the output CSV as independent recordings for any aggregate\n"
              "        (e.g. per-group) statistic.")

    out_path = OUT_DIR / "ingest_manifest.csv"
    df.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
