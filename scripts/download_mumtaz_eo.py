"""Download the Mumtaz/HUSM dataset's eyes-open (EO) resting recordings,
into a separate directory from the eyes-closed (EC) ones download_mumtaz.py
already fetched -- per Neureidos_Tool_Build_Spec_v1 Sec. 2/6: "Keep
eyes-closed and eyes-open separate throughout -- never pool."

Usage: python scripts/download_mumtaz_eo.py [--limit N]
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
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "mumtaz_eo"


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

    eo_files = [f for f in meta["files"] if f["name"].upper().endswith("EO.EDF")]
    print(f"{len(eo_files)} eyes-open files found (out of {len(meta['files'])} total)")
    if args.limit:
        eo_files = eo_files[: args.limit]

    manifest_rows = []
    n_ok, n_fail = 0, 0
    for i, f in enumerate(eo_files, 1):
        name = f["name"]  # e.g. "H S1 EO.edf" / "MDD S1 EO.edf"
        diagnosis = "healthy" if name.upper().startswith("H ") else (
            "depressed" if name.upper().startswith("MDD ") else None)
        if diagnosis is None:
            print(f"  [skip] unrecognized filename pattern: {name}")
            continue
        dest_name = name.replace(" ", "_")
        dest = OUT_DIR / dest_name
        ok = fetch(f["download_url"], dest)
        status = "ok" if ok else "FAILED"
        print(f"[{i}/{len(eo_files)}] {name}: {status}")
        if ok:
            n_ok += 1
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
