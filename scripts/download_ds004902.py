"""Download ds004902 ("A Resting-state EEG Dataset for Sleep Deprivation",
Xiang et al. 2024, OpenNeuro, CC0) -- the spec's optional arousal-test
extension (Neureidos_Tool_Build_Spec_v1 Sec. 2: "get only if time allows").
71 participants, each with two BIDS sessions (ses-1/ses-2): one normal-sleep
(NS) and one sleep-deprivation (SD) recording, matched within-subject.

Session-to-condition mapping is NOT fixed (ses-1 isn't always NS) -- it's
counterbalanced per participant and resolved via participants.tsv's
SessionOrder column, whose own data dictionary (participants.json) defines
it explicitly: "NS->SD" = "First normal sleep session, then sleep
deprivation session", "SD->NS" = the reverse. Since ses-1/ses-2 are BIDS's
standard chronological-visit numbering (confirmed: no other per-session
metadata field distinguishes condition in this dataset's eeg.json sidecars),
SessionOrder="NS->SD" means ses-1=NS/ses-2=SD, and "SD->NS" means the
reverse -- NOT guessed, read directly from the dataset's own documented
metadata. 41/71 participants are NS->SD, 30/71 are SD->NS (confirmed real
counterbalancing, not just a theoretical possibility).

Eyes-closed only (scope-limited vs. the dataset's own eyes-closed+eyes-open
design -- this extension is optional per the spec, and EC is the condition
most comparable to how the other three datasets are primarily used here).
Downloads a bounded subsample (--limit, default 25) rather than all 71 --
each recording is a real EEG file (~35MB .fdt), same "bounded for runtime,
not manipulated for effect" reasoning as Study A's subsample (see
pipeline.STUDY_A_N_PER_GROUP).

This dataset's manifest is a distinct, simpler shape (filename, subject_id,
session_condition) rather than Phase 1's diagnosis/severity schema --
manifest.py's diagnosis validator only recognizes healthy/depressed
synonyms, and NS/SD isn't a depression diagnosis, so this intentionally
does NOT go through manifest.parse_manifest_csv/validate_batch. See
scripts/run_ds004902_arousal.py, which reads this manifest.csv directly.

Usage: python scripts/download_ds004902.py [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import urllib.request
from pathlib import Path

PARTICIPANTS_URL = "https://s3.amazonaws.com/openneuro.org/ds004902/participants.tsv"
S3_BASE = "https://s3.amazonaws.com/openneuro.org/ds004902"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "ds004902"


def fetch(url: str, dest: Path, retries: int = 3) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
                f.write(r.read())
            return True
        except Exception as e:
            print(f"    [retry {attempt}/{retries}] {dest.name}: {e}", file=sys.stderr)
            time.sleep(2 * attempt)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    participants_path = OUT_DIR / "participants.tsv"
    fetch(PARTICIPANTS_URL, participants_path)
    rows = list(csv.DictReader(participants_path.read_text().splitlines(), delimiter="\t"))
    print(f"{len(rows)} total participants; SessionOrder counts: "
          f"NS->SD={sum(1 for r in rows if r['SessionOrder']=='NS->SD')}, "
          f"SD->NS={sum(1 for r in rows if r['SessionOrder']=='SD->NS')}")

    subjects = rows[: args.limit] if args.limit else rows
    print(f"downloading {len(subjects)} subjects (--limit {args.limit})")

    manifest_rows = []
    n_ok, n_fail = 0, 0
    for i, row in enumerate(subjects, 1):
        subj = row["participant_id"]
        order = row["SessionOrder"]
        if order == "NS->SD":
            ses_condition = {"ses-1": "NS", "ses-2": "SD"}
        elif order == "SD->NS":
            ses_condition = {"ses-1": "SD", "ses-2": "NS"}
        else:
            print(f"  [skip] {subj}: unrecognized SessionOrder {order!r}")
            continue

        subj_ok = True
        for ses, condition in ses_condition.items():
            base = f"{subj}_{ses}_task-eyesclosed_eeg"
            set_name, fdt_name = f"{base}.set", f"{base}.fdt"
            ok = (fetch(f"{S3_BASE}/{subj}/{ses}/eeg/{set_name}", OUT_DIR / set_name)
                  and fetch(f"{S3_BASE}/{subj}/{ses}/eeg/{fdt_name}", OUT_DIR / fdt_name))
            if ok:
                manifest_rows.append({"filename": set_name, "subject_id": subj, "session_condition": condition})
            else:
                subj_ok = False
        status = "ok" if subj_ok else "FAILED (partial)"
        print(f"[{i}/{len(subjects)}] {subj} ({order}): {status}")
        n_ok += 1 if subj_ok else 0
        n_fail += 0 if subj_ok else 1

    manifest_path = OUT_DIR / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "subject_id", "session_condition"])
        w.writeheader()
        w.writerows(manifest_rows)

    print(f"\ndone: {n_ok} subjects fully ok, {n_fail} partial/failed")
    print(f"wrote {manifest_path} ({len(manifest_rows)} rows)")


if __name__ == "__main__":
    main()
