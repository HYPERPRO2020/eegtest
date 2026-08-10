"""POST /api/create_job — start a new upload batch.

Body: {"manifest_csv": "<csv text>", "files": {"<filename>": "<blob url>", ...}}
(files are already sitting in Blob -- the browser uploaded them there
directly via the client-upload token handshake, see api/blob-upload-token.js
and ARCHITECTURE.md's job model; nothing here ever sees raw file bytes in
the request body).

Downloads every uploaded file into one shared temp directory (so BrainVision
.vhdr/.eeg/.vmrk triplets sit next to each other, same requirement
manifest.py's reader has locally), validates the batch against the manifest,
persists manifest/files/validation to Blob under a new job_id, and returns
the validation result plus which accepted recordings are in the Study A
pipeline-sweep subsample (so the browser knows which process_recording
calls to flag with run_study_a=true).
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from flask import Flask, request

from _common import error_response, new_job_id

import blob_client
from manifest import SUPPORTED_SUFFIXES, parse_manifest_csv, validate_batch
from pipeline import select_study_a_subsample

app = Flask(__name__)


@app.post("/api/create_job")
def create_job():
    body = request.get_json(silent=True) or {}
    manifest_csv = body.get("manifest_csv")
    files = body.get("files")
    if not manifest_csv or not files:
        return error_response("request body needs 'manifest_csv' and 'files'")

    try:
        manifest_rows = parse_manifest_csv(manifest_csv)
    except ValueError as e:
        return error_response(str(e))

    tmp_dir = None
    local_paths = []
    try:
        for filename, url in files.items():
            dest = blob_client.download_to_tmp(url, filename)
            if tmp_dir is None:
                tmp_dir = dest.parent
            else:
                # reuse one shared directory so BrainVision sidecars sit
                # next to their .vhdr -- download_to_tmp makes a fresh dir
                # per call, so move into the first one.
                shared_dest = tmp_dir / filename
                dest.rename(shared_dest)
                dest = shared_dest
            local_paths.append(dest)
    except Exception as e:
        return error_response(f"couldn't fetch uploaded file(s) from Blob: {e}", 502)

    # Sidecar-only files (BrainVision .eeg/.vmrk) aren't independently
    # validated -- they're picked up implicitly when their .vhdr is read.
    validate_paths = [p for p in local_paths if p.suffix.lower() in SUPPORTED_SUFFIXES]
    batch = validate_batch(validate_paths, manifest_rows)

    job_id = new_job_id()
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

    return validation_payload


# Vercel's Python runtime looks for a WSGI-compatible `app` in this file.
if __name__ == "__main__":
    app.run(debug=True, port=5001)
