"""One-off extraction of TDBRAIN's ses-1 resting-state EC/EO recordings for the
subset of subjects with a pure ("uncomorbid") MDD or HEALTHY indication label
in TDBRAIN_participants_V3.xlsx.

Reads scripts/tdbrain_availability.json (built by an earlier ad-hoc pass over
the participants sheet + the zip's own file listing -- see chat history) to
know which subjects have complete EC/EO/both, and pulls only those .bdf files
out of the password-protected zip (selective extraction -- never unpacks the
whole 15GB archive). Writes data/tdbrain/EC/ and data/tdbrain/EO/ each with
their own manifest.csv (filename,diagnosis,severity) in the format
neuroqa/manifest.py expects -- severity is each subject's BDI_pre score from
the participants sheet (scripts/tdbrain_severity.json), blank when TDBRAIN
didn't collect one for that subject.
"""

from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path

ZIP_PATH = Path(r"C:\Users\RyanC\Downloads\TDBRAIN_Dataset_V3_1_Encr.zip")
AVAIL_PATH = Path(__file__).parent / "tdbrain_availability.json"
SEVERITY_PATH = Path(__file__).parent / "tdbrain_severity.json"
OUT_ROOT = Path(__file__).parent.parent / "data" / "tdbrain"

PASSWORD = os.environ.get("TDBRAIN_ZIP_PASSWORD")
if not PASSWORD:
    sys.exit("set TDBRAIN_ZIP_PASSWORD env var before running this script")


def main():
    avail = json.loads(AVAIL_PATH.read_text())
    severity = json.loads(SEVERITY_PATH.read_text())
    z = zipfile.ZipFile(ZIP_PATH)

    for cond_key, cond_dir in [("ec", "EC"), ("eo", "EO")]:
        out_dir = OUT_ROOT / cond_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_rows = [("filename", "diagnosis", "severity")]

        subjects = [(sid, v) for sid, v in avail.items() if v[cond_key]]
        print(f"[{cond_dir}] extracting {len(subjects)} recordings ...")
        for i, (sid, v) in enumerate(subjects, 1):
            name_in_zip = f"TDBRAIN_Dataset_V3_1/{sid}/ses-1/eeg/{sid}_ses-1_task-rest{cond_dir}_eeg.bdf"
            out_name = f"{sid}_ses-1_task-rest{cond_dir}_eeg.bdf"
            out_path = out_dir / out_name
            if not out_path.exists():
                data = z.read(name_in_zip, pwd=PASSWORD.encode())
                out_path.write_bytes(data)
            manifest_rows.append((out_name, v["group"], str(severity.get(sid, ""))))
            if i % 50 == 0 or i == len(subjects):
                print(f"  [{cond_dir}] {i}/{len(subjects)}")

        manifest_path = out_dir / "manifest.csv"
        with manifest_path.open("w", newline="") as f:
            for row in manifest_rows:
                f.write(",".join(row) + "\n")
        print(f"[{cond_dir}] wrote {manifest_path} ({len(manifest_rows)-1} rows)")


if __name__ == "__main__":
    main()
