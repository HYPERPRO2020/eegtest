"""Download ds003478 ("EEG: Depression rest", OpenNeuro, CC0) run-01 recordings
for every subject matching the dataset's own published BDI groups (hBDI>13 /
CTL<7), plus build a NeuroQA manifest.csv from participants.tsv's real BDI
scores. No login needed -- plain HTTPS from the dataset's public S3 bucket.

Usage: python scripts/download_ds003478.py [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import urllib.request
from pathlib import Path

PARTICIPANTS_URL = "https://raw.githubusercontent.com/OpenNeuroDatasets/ds003478/master/participants.tsv"
S3_BASE = "https://s3.amazonaws.com/openneuro.org/ds003478"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "ds003478"

HBDI_CUTOFF = 13  # BDI > 13 -> depressed (matches dataset's own published hBDI group)
CTL_CUTOFF = 7    # BDI < 7  -> healthy   (matches dataset's own published CTL group)


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
    parser.add_argument("--limit", type=int, default=None, help="only download the first N eligible subjects (for a quick smoke test)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    participants_path = OUT_DIR / "participants.tsv"
    fetch(PARTICIPANTS_URL, participants_path)

    rows = list(csv.DictReader(participants_path.read_text().splitlines(), delimiter="\t"))
    eligible = []
    for row in rows:
        bdi_raw = row.get("BDI", "").strip()
        try:
            bdi = float(bdi_raw)
        except ValueError:
            continue  # NaN / INVALID PARTICIPANT (e.g. sub-038/544) -- excluded
        if bdi > HBDI_CUTOFF:
            diagnosis = "depressed"
        elif bdi < CTL_CUTOFF:
            diagnosis = "healthy"
        else:
            continue  # in the ambiguous middle band -- not part of either published group
        eligible.append((row["participant_id"], bdi, diagnosis))

    print(f"{len(eligible)} subjects match the published hBDI(>{HBDI_CUTOFF})/CTL(<{CTL_CUTOFF}) groups "
          f"out of {len(rows)} total participants")
    if args.limit:
        eligible = eligible[: args.limit]
        print(f"--limit {args.limit}: downloading only the first {len(eligible)}")

    manifest_rows = []
    n_ok, n_fail = 0, 0
    for i, (subj, bdi, diagnosis) in enumerate(eligible, 1):
        set_name = f"{subj}_task-Rest_run-01_eeg.set"
        fdt_name = f"{subj}_task-Rest_run-01_eeg.fdt"
        set_url = f"{S3_BASE}/{subj}/eeg/{set_name}"
        fdt_url = f"{S3_BASE}/{subj}/eeg/{fdt_name}"
        set_dest = OUT_DIR / set_name
        fdt_dest = OUT_DIR / fdt_name

        ok = fetch(set_url, set_dest) and fetch(fdt_url, fdt_dest)
        status = "ok" if ok else "FAILED"
        print(f"[{i}/{len(eligible)}] {subj} (BDI={bdi:g}, {diagnosis}): {status}")
        if ok:
            n_ok += 1
            manifest_rows.append({"filename": set_name, "diagnosis": diagnosis, "severity": bdi})
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
