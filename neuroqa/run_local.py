"""NeuroQA — local batch CLI: run the full upload -> validate -> score ->
Study A -> Study B -> results pipeline against a local folder, without
needing a live Vercel deploy.

This exercises exactly the same functions (manifest.py, pipeline.py,
study_a.py, study_b.py) the Vercel API routes call per-recording -- it's a
single-process stand-in for the fan-out job model (see ARCHITECTURE.md),
useful for local testing/dev and for anyone who'd rather not use the
hosted UI at all.

Usage:
    python neuroqa/run_local.py <folder-of-recordings> <manifest.csv> [--out outputs/]

manifest.csv needs filename/diagnosis/severity columns (or common synonyms,
see manifest.parse_manifest_csv) -- filename values must match files in
<folder-of-recordings> exactly.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from joblib import Parallel, delayed

from manifest import parse_manifest_csv, validate_batch
from pipeline import aggregate, score_and_faa, select_study_a_subsample, study_a_for_recording, validation_result_to_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("manifest_csv", type=Path)
    parser.add_argument("--out", type=Path, default=Path("neuroqa/outputs"))
    parser.add_argument("--study-a-n-per-group", type=int, default=None)
    parser.add_argument("--line-freq", type=float, default=None,
                         help="mains hum frequency (Hz) for the notch filter -- default 50.0 "
                              "(preprocess.LINE_FREQ); pass 60.0 for US-mains recordings, e.g. ds003478")
    parser.add_argument("--n-jobs", type=int, default=1,
                         help="parallel workers for the Study A pipeline sweep (joblib); "
                              "-1 uses all cores. Each recording's 10-combo sweep is independent "
                              "of every other recording's, so this only speeds up Study A, not "
                              "the (already cheap) base-scoring pass.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_rows = parse_manifest_csv(args.manifest_csv.read_text())
    paths = sorted(p for p in args.data_dir.iterdir() if p.is_file())

    print(f"validating {len(paths)} files against {len(manifest_rows)} manifest rows ...")
    batch = validate_batch(paths, manifest_rows)
    print(f"  accepted: {len(batch.accepted)}   rejected: {len(batch.rejected)}")
    for r in batch.rejected:
        print(f"    [rejected] {r.filename}: {'; '.join(r.reasons)}")
    for r in batch.accepted:
        if r.warnings:
            print(f"    [warning]  {r.filename}: {'; '.join(r.warnings)}")
    if batch.duplicate_groups:
        print(f"  duplicate groups (by md5): {batch.duplicate_groups}")

    (args.out / "validation.json").write_text(json.dumps({
        "accepted": [validation_result_to_dict(r) for r in batch.accepted],
        "rejected": [validation_result_to_dict(r) for r in batch.rejected],
        "duplicate_groups": batch.duplicate_groups,
        "group_ok": batch.group_ok,
        "group_reasons": batch.group_reasons,
    }, indent=2))

    if not batch.group_ok:
        raise SystemExit(f"batch is not a usable group: {'; '.join(batch.group_reasons)}")

    path_by_name = {p.name: p for p in paths}

    print(f"\nscoring {len(batch.accepted)} accepted recordings ...")
    base_rows = []
    for r in batch.accepted:
        try:
            row = score_and_faa(path_by_name[r.filename], line_freq=args.line_freq)
        except Exception as e:
            print(f"  [ERROR] {r.filename}: {e}")
            continue
        row.update(file=r.filename, group=r.diagnosis, clinical_severity=r.severity)
        base_rows.append(row)
        print(f"  {r.filename:28s} grade={row['grade']}  quality[alpha]={row['quality_alpha_pct']:5.1f}%  FAA={row['faa']:+.4f}")

    kwargs = {}
    if args.study_a_n_per_group is not None:
        kwargs["n_per_group"] = args.study_a_n_per_group
    subsample = select_study_a_subsample(batch.accepted, **kwargs)
    print(f"\nrunning Study A pipeline sweep on {len(subsample)} recordings ({args.n_jobs} job(s)) ...")

    def _run_one(r):
        print(f"  {r.filename} ...")
        combos = study_a_for_recording(path_by_name[r.filename], line_freq=args.line_freq)
        return r, combos

    outcomes = Parallel(n_jobs=args.n_jobs)(delayed(_run_one)(r) for r in subsample)
    study_a_rows = []
    for r, combos in outcomes:
        for combo_key, faa_val in combos.items():
            pipeline, reference = combo_key.rsplit("_", 1)
            study_a_rows.append({"file": r.filename, "group": r.diagnosis,
                                  "pipeline": pipeline, "reference": reference, "faa": faa_val})

    print("\naggregating Study A + Study B ...")
    results = aggregate(base_rows, study_a_rows)
    (args.out / "results.json").write_text(json.dumps(results, indent=2))

    # Per-recording quality/FAA table -- the brief's "results table" deliverable
    # and what notebooks/phase1_findings.ipynb loads instead of recomputing
    # (same cache convention its own intro cell documents: reuse outputs/*.csv
    # when present, delete to force a from-scratch recompute).
    summary_cols = ["file", "group", "clinical_severity", "grade", "quality_alpha_pct",
                     "quality_alpha_frontal_pct", "raw_severity_mean", "faa", "faa_raw", "n_epochs", "n_channels"]
    with open(args.out / "quality_faa_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(base_rows)

    print(f"\nwrote {args.out / 'validation.json'}")
    print(f"wrote {args.out / 'results.json'}")
    print(f"wrote {args.out / 'quality_faa_summary.csv'}")
    print(f"\nfinding: {results['study_b'].get('finding', '(no Study B result)')}")


if __name__ == "__main__":
    main()
