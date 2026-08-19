"""Reconstruct outputs/<dataset>/quality_faa_summary.csv from a completed
run_local.py console log, for the two real-data runs that finished just
before run_local.py started writing this CSV itself (see run_local.py's
`quality_faa_summary.csv` output, added afterward). Deterministic + seeded
run (pipeline.SEED=0), so parsing the log's own printed numbers reproduces
exactly what score_and_faa returned -- not a re-estimate.

A from-scratch re-run of run_local.py no longer needs this script; it
writes the CSV directly now. Kept only as a record of how these two numbers
were recovered without repeating ~20 minutes of MNE scoring.

Usage: python scripts/build_summary_csv_from_log.py <log_file> <manifest.csv> <out_csv>
"""
import csv
import re
import sys
from pathlib import Path

LINE_RE = re.compile(
    r"^\s*(?P<file>\S+)\s+grade=(?P<grade>\S+)\s+quality\[alpha\]=\s*(?P<quality>[\d.]+)%\s+FAA=(?P<faa>[+-][\d.]+)\s*$"
)


def main():
    log_path, manifest_path, out_path = (Path(p) for p in sys.argv[1:4])

    manifest = {}
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            manifest[row["filename"]] = row

    rows = []
    for line in log_path.read_text(errors="replace").splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        filename = m.group("file")
        man_row = manifest.get(filename, {})
        rows.append({
            "file": filename,
            "group": man_row.get("diagnosis", ""),
            "clinical_severity": man_row.get("severity", ""),
            "grade": m.group("grade"),
            "quality_alpha_pct": m.group("quality"),
            "faa": m.group("faa"),
        })

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "group", "clinical_severity", "grade", "quality_alpha_pct", "faa"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
