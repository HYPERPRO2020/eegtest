"""POST /api/process_recording — score exactly one recording.

Body: {"job_id", "filename", "diagnosis", "severity", "run_study_a": bool}.
One recording per invocation, per ARCHITECTURE.md's job model -- the
browser fans this out in parallel across every accepted recording in the
batch right after create_job returns (run_study_a=true only for the
recordings create_job flagged as the Study A subsample; that sweep runs
ICA + AutoReject per pipeline/reference combo and is too slow to run for
every recording in a large batch).

Writes the result to jobs/{job_id}/status/{filename}.json in Blob (so a
browser reload/job_status poll can recover it) and also returns it directly
in the response.
"""

from __future__ import annotations

from flask import Flask, request

from _common import error_response

import blob_client
from pipeline import score_and_faa, study_a_for_recording

app = Flask(__name__)


@app.post("/api/process_recording")
def process_recording():
    body = request.get_json(silent=True) or {}
    job_id = body.get("job_id")
    filename = body.get("filename")
    diagnosis = body.get("diagnosis")
    severity = body.get("severity")
    run_study_a = bool(body.get("run_study_a"))
    if not job_id or not filename or not diagnosis:
        return error_response("request body needs 'job_id', 'filename', 'diagnosis'")

    # Blob URLs are host-per-store, not derivable from the pathname alone,
    # so the caller passes back the files.json URL create_job's PUT returned
    # rather than this route trying to reconstruct it.
    files_map_url = body.get("files_map_url")
    if not files_map_url:
        return error_response("request body needs 'files_map_url' (from create_job's response)")

    try:
        files = blob_client.get_json(files_map_url)
        local_path = blob_client.download_group_to_tmp(files, filename)
    except Exception as e:
        status = {"filename": filename, "group": diagnosis, "severity": severity,
                  "ok": False, "error": f"couldn't fetch file from Blob: {e}",
                  "base": None, "study_a": None}
        blob_client.write_recording_status(job_id, filename, status)
        return status, 502

    try:
        base = score_and_faa(local_path)
    except Exception as e:
        status = {"filename": filename, "group": diagnosis, "severity": severity,
                  "ok": False, "error": str(e), "base": None, "study_a": None}
        blob_client.write_recording_status(job_id, filename, status)
        return status, 200  # a per-recording failure isn't a request failure

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
    return status


if __name__ == "__main__":
    app.run(debug=True, port=5002)
