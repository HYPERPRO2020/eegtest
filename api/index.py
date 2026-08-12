"""NeuroQA study-batch API — one Flask app, one Vercel Python function.

Was originally four separate api/*.py files (one @vercel/python build
each). Consolidated into a single function after a live deploy hit a
Vercel build-infra bug (`ENOENT ... lstat ... pycache ...`, a doubled/
mangled path) that's been reported when several @vercel/python builds in
one project independently pip-install overlapping heavy dependencies
(mne/scipy/numpy/etc. here) — see ARCHITECTURE.md's post-deploy notes.
Cutting Python builds from five (webapp.py + 4 api routes) to two
(webapp.py + this file) directly shrinks that risk, on top of being
simpler and faster to cold-start (one shared mne import instead of four).

Routes (unchanged from the four-file version, so neuroqa/templates/
study.html's fetch() calls didn't need to change):
    POST /api/create_job
    POST /api/process_recording
    GET  /api/job_status
    POST /api/aggregate

See each function's docstring below for its contract; see ARCHITECTURE.md
for the job model these implement.
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "neuroqa"))

from flask import Flask, jsonify, request

import blob_client
from manifest import SUPPORTED_SUFFIXES, parse_manifest_csv, validate_batch
from pipeline import aggregate as run_aggregate
from pipeline import score_and_faa, select_study_a_subsample, study_a_for_recording

app = Flask(__name__)
app.json.compact = True


def _new_job_id() -> str:
    import uuid
    return uuid.uuid4().hex[:12]


def _error(message: str, status: int = 400):
    return jsonify({"error": message}), status


@app.post("/api/create_job")
def create_job():
    """Start a new upload batch.

    Body: {"manifest_csv": "<csv text>", "files": {"<filename>": "<blob url>", ...}}
    (files are already sitting in Blob -- the browser uploaded them there
    directly via the client-upload token handshake, see
    api/blob-upload-token.js). Downloads every uploaded file into one
    shared temp directory (so BrainVision .vhdr/.eeg/.vmrk triplets sit
    next to each other), validates the batch against the manifest,
    persists manifest/files/validation to Blob under a new job_id, and
    returns the validation result plus which accepted recordings are in
    the Study A pipeline-sweep subsample.
    """
    body = request.get_json(silent=True) or {}
    manifest_csv = body.get("manifest_csv")
    files = body.get("files")
    if not manifest_csv or not files:
        return _error("request body needs 'manifest_csv' and 'files'")

    try:
        manifest_rows = parse_manifest_csv(manifest_csv)
    except ValueError as e:
        return _error(str(e))

    tmp_dir = None
    local_paths = []
    try:
        for filename, url in files.items():
            dest = blob_client.download_to_tmp(url, filename)
            if tmp_dir is None:
                tmp_dir = dest.parent
            else:
                shared_dest = tmp_dir / filename
                dest.rename(shared_dest)
                dest = shared_dest
            local_paths.append(dest)
    except Exception as e:
        return _error(f"couldn't fetch uploaded file(s) from Blob: {e}", 502)

    validate_paths = [p for p in local_paths if p.suffix.lower() in SUPPORTED_SUFFIXES]
    batch = validate_batch(validate_paths, manifest_rows)

    job_id = _new_job_id()
    blob_client.write_manifest(job_id, manifest_csv)
    files_map_url = blob_client.write_files_map(job_id, files)

    subsample = select_study_a_subsample(batch.accepted)
    subsample_names = {r.filename for r in subsample}

    validation_payload = {
        "job_id": job_id,
        "files_map_url": files_map_url,
        "accepted": [asdict(r) for r in batch.accepted],
        "rejected": [asdict(r) for r in batch.rejected],
        "duplicate_groups": batch.duplicate_groups,
        "group_ok": batch.group_ok,
        "group_reasons": batch.group_reasons,
        "study_a_subsample": sorted(subsample_names),
    }
    blob_client.write_validation(job_id, validation_payload)
    return jsonify(validation_payload)


@app.post("/api/process_recording")
def process_recording():
    """Score exactly one recording.

    Body: {"job_id", "filename", "diagnosis", "severity", "run_study_a": bool,
    "files_map_url"}. One recording per invocation, per ARCHITECTURE.md's
    job model -- the browser fans this out in parallel across every
    accepted recording right after create_job returns (run_study_a=true
    only for the Study A subsample -- that sweep runs ICA + AutoReject per
    pipeline/reference combo and is too slow for every recording). Writes
    the result to jobs/{job_id}/status/{filename}.json in Blob and also
    returns it directly.
    """
    body = request.get_json(silent=True) or {}
    job_id = body.get("job_id")
    filename = body.get("filename")
    diagnosis = body.get("diagnosis")
    severity = body.get("severity")
    run_study_a = bool(body.get("run_study_a"))
    if not job_id or not filename or not diagnosis:
        return _error("request body needs 'job_id', 'filename', 'diagnosis'")

    files_map_url = body.get("files_map_url")
    if not files_map_url:
        return _error("request body needs 'files_map_url' (from create_job's response)")

    try:
        files = blob_client.get_json(files_map_url)
        local_path = blob_client.download_group_to_tmp(files, filename)
    except Exception as e:
        status = {"filename": filename, "group": diagnosis, "severity": severity,
                  "ok": False, "error": f"couldn't fetch file from Blob: {e}",
                  "base": None, "study_a": None}
        blob_client.write_recording_status(job_id, filename, status)
        return jsonify(status), 502

    try:
        base = score_and_faa(local_path)
    except Exception as e:
        status = {"filename": filename, "group": diagnosis, "severity": severity,
                  "ok": False, "error": str(e), "base": None, "study_a": None}
        blob_client.write_recording_status(job_id, filename, status)
        return jsonify(status)  # a per-recording failure isn't a request failure

    study_a_result = None
    if run_study_a:
        try:
            study_a_result = study_a_for_recording(local_path)
        except Exception as e:
            study_a_result = {"error": str(e)}

    status = {
        "filename": filename, "group": diagnosis, "severity": severity,
        "ok": True, "error": None, "base": base, "study_a": study_a_result,
    }
    blob_client.write_recording_status(job_id, filename, status)
    return jsonify(status)


@app.get("/api/job_status")
def job_status():
    """Poll progress for a batch: every jobs/{job_id}/status/{filename}.json
    blob written so far by process_recording invocations. Used for the
    frontend's progress bar and to recover state after a browser reload."""
    job_id = request.args.get("job_id")
    if not job_id:
        return _error("query param 'job_id' is required")

    statuses = blob_client.read_all_recording_statuses(job_id)
    n_ok = sum(1 for s in statuses if s.get("ok"))
    n_error = sum(1 for s in statuses if not s.get("ok"))
    return jsonify({
        "job_id": job_id, "n_done": len(statuses), "n_ok": n_ok,
        "n_error": n_error, "statuses": statuses,
    })


@app.post("/api/aggregate")
def aggregate():
    """Final Study A / Study B aggregation for a batch.

    Body: {"job_id"}. Reads every recording's status from Blob, splits
    into Study B's base rows and Study A's long-format pipeline-sweep
    rows, runs pipeline.aggregate(), persists to
    jobs/{job_id}/results.json, and returns it.
    """
    body = request.get_json(silent=True) or {}
    job_id = body.get("job_id")
    if not job_id:
        return _error("request body needs 'job_id'")

    statuses = blob_client.read_all_recording_statuses(job_id)
    if not statuses:
        return _error(f"no scored recordings found for job {job_id}", 404)

    base_rows = []
    study_a_rows = []
    for s in statuses:
        if not s.get("ok"):
            continue
        base = dict(s["base"])
        base["file"] = s["filename"]
        base["group"] = s["group"]
        base_rows.append(base)

        study_a = s.get("study_a")
        if study_a and "error" not in study_a:
            for combo_key, faa_val in study_a.items():
                pipeline_name, reference = combo_key.rsplit("_", 1)
                study_a_rows.append({
                    "file": s["filename"], "group": s["group"],
                    "pipeline": pipeline_name, "reference": reference, "faa": faa_val,
                })

    try:
        results = run_aggregate(base_rows, study_a_rows)
    except Exception as e:
        return _error(f"aggregation failed: {e}", 500)

    blob_client.write_results(job_id, results)
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
