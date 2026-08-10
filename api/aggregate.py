"""POST /api/aggregate — final Study A / Study B aggregation for a batch.

Body: {"job_id"}. Reads every recording's status from Blob (not from the
request body -- Blob is the single source of truth for job state, see
ARCHITECTURE.md), splits them into Study B's base rows (every successfully
scored recording) and Study A's long-format pipeline-sweep rows (only the
subsample recordings process_recording ran with run_study_a=true), runs
pipeline.aggregate(), persists the result to jobs/{job_id}/results.json,
and returns it. Cheap and fast (pure numpy/pandas/statsmodels/sklearn over
already-computed numbers, no mne) -- meant to be called once, after
job_status shows every recording is done.
"""

from __future__ import annotations

from flask import Flask, request

from _common import error_response

import blob_client
from pipeline import aggregate as run_aggregate

app = Flask(__name__)


@app.post("/api/aggregate")
def aggregate():
    body = request.get_json(silent=True) or {}
    job_id = body.get("job_id")
    if not job_id:
        return error_response("request body needs 'job_id'")

    statuses = blob_client.read_all_recording_statuses(job_id)
    if not statuses:
        return error_response(f"no scored recordings found for job {job_id}", 404)

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
        return error_response(f"aggregation failed: {e}", 500)

    blob_client.write_results(job_id, results)
    return results


if __name__ == "__main__":
    app.run(debug=True, port=5004)
