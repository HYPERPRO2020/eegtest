"""Run Study C (synthetic artifact-injection dose-response, see
neuroqa/study_c.py) against one dataset's Study-A subsample, writing
outputs/<dataset>/study_c.json.

Reuses the exact same accepted-batch validation and subsample-selection
logic run_local.py/pipeline.py use for Study A, so Study C runs on the same
recordings Study A already reports on -- directly comparable, not a
different sample.

Usage: python scripts/run_study_c.py <data_dir> <manifest.csv> --out outputs/<dataset> [--line-freq 60.0] [--n-per-group 15]
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "neuroqa"))

from manifest import parse_manifest_csv, validate_batch  # noqa: E402
from pipeline import select_study_a_subsample  # noqa: E402
from study_c import run_study_c  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("manifest_csv", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--line-freq", type=float, default=None)
    parser.add_argument("--n-per-group", type=int, default=15)
    args = parser.parse_args()

    manifest_rows = parse_manifest_csv(args.manifest_csv.read_text())
    paths = sorted(p for p in args.data_dir.iterdir() if p.is_file())
    batch = validate_batch(paths, manifest_rows)
    print(f"accepted: {len(batch.accepted)} rejected: {len(batch.rejected)}")

    subsample = select_study_a_subsample(batch.accepted, n_per_group=args.n_per_group)
    path_by_name = {p.name: p for p in paths}
    subsample_paths = [path_by_name[r.filename] for r in subsample]
    print(f"running Study C on {len(subsample_paths)} recordings (same subsample as Study A) ...")

    result = run_study_c(subsample_paths, line_freq=args.line_freq)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "study_c.json").write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out / 'study_c.json'}")
    for kind, s in result["summary"].items():
        print(f"  {kind:20s} n={s['n_subjects']:3d} mean_slope={s['mean_slope']:+.5f} "
              f"CI=[{s['slope_ci_lo']:+.5f}, {s['slope_ci_hi']:+.5f}] nonzero={s['nonzero_slope']}")


if __name__ == "__main__":
    main()
