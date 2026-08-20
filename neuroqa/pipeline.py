"""NeuroQA — orchestration: validated upload -> per-recording scoring ->
Study A / Study B aggregation.

This is the shared core both the Vercel API routes (api/*.py, one recording
or one aggregation per invocation -- see ARCHITECTURE.md's job model) and
the local CLI (run_local.py, for offline testing without a live Vercel
deploy) call into. Nothing here does file I/O beyond reading the recording
itself -- callers own uploading/fetching files and persisting results
(to Vercel Blob or local disk).

Fixed SEED below drives every source of randomness in the pipeline (ICA,
AutoReject, cross-validation splits) so the same uploaded batch produces
the same result every run -- see project brief, "Reproducibility."
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from faa_classifiers import classify_by_pipeline
from manifest import BatchValidationResult, ValidationResult
from score import score_recording
from study_a import PIPELINES, REFERENCES, run_all_pipelines, spread_stats
from study_b import run_study_b

SEED = 0
STUDY_A_N_PER_GROUP = 6  # recordings per group in the pipeline-sweep subsample --
# ICA + autoreject are the slow steps (~10-20s each); this question (how much
# does FAA move across pipeline choices) is about within/across-subject
# spread, not statistical power, so a bounded subsample is the right
# tradeoff -- same reasoning the previous fixed-dataset version used.


def score_and_faa(path: str | Path) -> dict:
    """Base per-recording scoring: quality across all bands + FAA under the
    endpoint-aware quality-weighted ("ours") pipeline. Cheap (no ICA/
    autoreject) -- this is what runs for every accepted upload, feeding
    Study B. Returns a JSON-safe dict.
    """
    from bands import EEG_BANDS
    from faa import compute_faa
    from preprocess import preprocess_file
    from quality_index import compute_quality

    data_uv, ch_names, sfreq = preprocess_file(path)
    row, channel_detail = score_recording(data_uv, ch_names, sfreq)

    quality = compute_quality(data_uv, ch_names, sfreq, EEG_BANDS["alpha"])["quality"]
    i3, i4 = ch_names.index("F3"), ch_names.index("F4")
    faa_result = compute_faa(data_uv, ch_names, sfreq,
                              weights_f3=quality[:, i3], weights_f4=quality[:, i4])

    return {
        **row,
        "quality_alpha_pct": row["quality_alpha_pct"],
        "faa": round(faa_result["faa"], 4),
        "n_channels_used": len(ch_names),
        "channels_used": ch_names,
        "channel_detail": channel_detail,
    }


def select_study_a_subsample(accepted: list[ValidationResult],
                              n_per_group: int = STUDY_A_N_PER_GROUP) -> list[ValidationResult]:
    """Deterministic subsample for Study A's pipeline sweep: sort accepted
    recordings by filename (stable given the same upload), dedupe by md5,
    take the first `n_per_group` per diagnosis label."""
    by_md5_seen: set[str] = set()
    deduped = []
    for r in sorted(accepted, key=lambda r: r.filename):
        if r.md5 in by_md5_seen:
            continue
        by_md5_seen.add(r.md5)
        deduped.append(r)

    out = []
    for label in sorted({r.diagnosis for r in deduped}):
        out.extend([r for r in deduped if r.diagnosis == label][:n_per_group])
    return out


def study_a_for_recording(path: str | Path, seed: int = SEED) -> dict[str, float]:
    """Study A's 10-combo (5 pipelines x 2 references) FAA sweep for one
    recording. Expensive (ICA + AutoReject per combo) -- only called for
    `select_study_a_subsample`'s output, not every accepted recording.
    Returns {"pipeline_reference": faa_value, ...} (flat, JSON-safe keys).
    """
    results = run_all_pipelines(str(path), seed=seed)
    return {f"{pipeline}_{reference}": round(v, 4) for (pipeline, reference), v in results.items()}


def aggregate(base_rows: list[dict], study_a_rows: list[dict]) -> dict:
    """Final aggregation: Study A spread stats + Study B regressions/
    classifier, over whatever recordings the caller already scored.

    base_rows: one dict per accepted recording, each score_and_faa()'s
      output plus {"file", "group"} merged in by the caller.
    study_a_rows: one dict per Study-A-subsample recording, each
      {"file", "group", "pipeline", "reference", "faa"} (long format --
      caller flattens study_a_for_recording()'s combo dict into this shape).
    """
    study_a_result = spread_stats(study_a_rows) if study_a_rows else {"long": [], "wide": [], "combo_cols": []}
    study_b_result = run_study_b(base_rows) if base_rows else {"n": 0}
    # Peter's ask: 4-5 independent FAA-only classifiers, one per Study A
    # pipeline, that don't share data with each other -- see
    # faa_classifiers.py's module docstring.
    faa_classifiers_result = classify_by_pipeline(study_a_rows) if study_a_rows else {}
    return {
        "seed": SEED,
        "n_recordings": len(base_rows),
        "n_study_a_recordings": len({r["file"] for r in study_a_rows}) if study_a_rows else 0,
        "study_a": study_a_result,
        "study_b": study_b_result,
        "faa_classifiers_by_pipeline": faa_classifiers_result,
    }


def validation_result_to_dict(r: ValidationResult) -> dict:
    return asdict(r)
