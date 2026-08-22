"""Download ds007615 ("LDAEP and resting-state EEG in healthy women",
OpenNeuro, CC0) eyes-closed resting recordings for every subject matching
the SAME BDI thresholds used for ds003478 (>13 depressed, <7 healthy) --
kept consistent across datasets rather than picking new cutoffs per
dataset. Real per-subject BDI-II total score (phenotype/bdi.tsv). No
login needed -- plain HTTPS from the dataset's public S3 bucket, same
mechanism as ds003478.

A genuinely independent cohort from ds003478/ds003474 (University of Oslo,
not the University of Arizona Cavanagh-lab cohort those two datasets share)
-- found via a background dataset search, see the session's other commits.

Usage: python scripts/download_ds007615.py [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import urllib.request
from pathlib import Path

BDI_URL = "https://s3.amazonaws.com/openneuro.org/ds007615/phenotype/bdi.tsv"
S3_BASE = "https://s3.amazonaws.com/openneuro.org/ds007615"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "ds007615"

HBDI_CUTOFF = 13  # BDI > 13 -> depressed (same cutoff as ds003478, for consistency)
CTL_CUTOFF = 7    # BDI < 7  -> healthy


def fetch(url: str, dest: Path, retries: int = 3) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(dest, "wb") as f:
                f.write(r.read())
            return True
        except Exception as e:
            print(f"    [retry {attempt}/{retries}] {dest.name}: {e}", file=sys.stderr)
            time.sleep(2 * attempt)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bdi_path = OUT_DIR / "bdi.tsv"
    fetch(BDI_URL, bdi_path)

    rows = list(csv.DictReader(bdi_path.read_text().splitlines(), delimiter="\t"))
    eligible = []
    for row in rows:
        try:
            bdi = float(row["bdi_total"])
        except (KeyError, ValueError):
            continue
        if bdi > HBDI_CUTOFF:
            diagnosis = "depressed"
        elif bdi < CTL_CUTOFF:
            diagnosis = "healthy"
        else:
            continue
        eligible.append((row["participant_id"], bdi, diagnosis))

    print(f"{len(eligible)} subjects match BDI(>{HBDI_CUTOFF})/BDI(<{CTL_CUTOFF}) groups "
          f"out of {len(rows)} total participants")
    if args.limit:
        eligible = eligible[: args.limit]
        print(f"--limit {args.limit}: downloading only the first {len(eligible)}")

    manifest_rows = []
    n_ok, n_fail = 0, 0
    for i, (subj, bdi, diagnosis) in enumerate(eligible, 1):
        vhdr_name = f"{subj}_task-rest_acq-ec_eeg.vhdr"
        eeg_name = f"{subj}_task-rest_acq-ec_eeg.eeg"
        vmrk_name = f"{subj}_task-rest_acq-ec_eeg.vmrk"
        ok = all(
            fetch(f"{S3_BASE}/{subj}/eeg/{name}", OUT_DIR / name)
            for name in (vhdr_name, eeg_name, vmrk_name)
        )
        status = "ok" if ok else "FAILED"
        print(f"[{i}/{len(eligible)}] {subj} (BDI={bdi:g}, {diagnosis}): {status}")
        if ok:
            n_ok += 1
            manifest_rows.append({"filename": vhdr_name, "diagnosis": diagnosis, "severity": bdi})
        else:
            n_fail += 1

    manifest_path = OUT_DIR / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "diagnosis", "severity"])
        w.writeheader()
        w.writerows(manifest_rows)

    print(f"\ndone: {n_ok} ok, {n_fail} failed")
    print(f"wrote {manifest_path} ({len(manifest_rows)} rows)")


if __name__ == "__main__":
    main()
