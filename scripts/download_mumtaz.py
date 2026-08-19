"""Download the Mumtaz/HUSM 'MDD Patients and Healthy Controls EEG Data (New)'
dataset (figshare 4244171, CC BY 4.0) -- used here as a secondary replication
check (no per-subject severity scores in this deposit, so only Study A and
Study B's non-severity analyses run against it, see ARCHITECTURE.md). No
login needed -- figshare's public API serves direct download URLs.

Only the eyes-closed (EC) resting recordings are pulled -- one per subject,
matching the standard resting-state FAA paradigm; EO and TASK (P300) files
are out of scope for this study. Diagnosis comes straight from the filename
convention ('H ...' vs 'MDD ...'), which is this deposit's only labeling.

Usage: python scripts/download_mumtaz.py [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

API_URL = "https://api.figshare.com/v2/articles/4244171"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "mumtaz"


def fetch(url: str, dest: Path, retries: int = 3) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
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
    req = urllib.request.Request(API_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        meta = json.load(r)

    ec_files = [f for f in meta["files"] if f["name"].upper().endswith("EC.EDF")]
    print(f"{len(ec_files)} eyes-closed files found (out of {len(meta['files'])} total)")
    if args.limit:
        ec_files = ec_files[: args.limit]

    manifest_rows = []
    n_ok, n_fail = 0, 0
    for i, f in enumerate(ec_files, 1):
        name = f["name"]  # e.g. "H S1 EC.edf" / "MDD S1 EC.edf"
        diagnosis = "healthy" if name.upper().startswith("H ") else (
            "depressed" if name.upper().startswith("MDD ") else None)
        if diagnosis is None:
            print(f"  [skip] unrecognized filename pattern: {name}")
            continue
        dest_name = name.replace(" ", "_")
        dest = OUT_DIR / dest_name
        ok = fetch(f["download_url"], dest)
        status = "ok" if ok else "FAILED"
        print(f"[{i}/{len(ec_files)}] {name}: {status}")
        if ok:
            n_ok += 1
            # No per-subject severity in this deposit (see module docstring) --
            # leave severity blank; manifest.py's validator requires a numeric
            # severity, so run_local.py's caller must special-case this dataset
            # (see notes in ARCHITECTURE.md / the run script) rather than
            # inventing a fake number here.
            manifest_rows.append({"filename": dest_name, "diagnosis": diagnosis, "severity": ""})
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
