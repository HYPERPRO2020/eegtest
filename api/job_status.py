"""GET /api/job_status?job_id=... — poll progress for a batch.

Reads every jobs/{job_id}/status/{filename}.json blob written so far by
process_recording invocations. Used both for the frontend's progress bar
and, if the browser reloads mid-run, to recover which recordings are
already done instead of re-processing them.
"""

from __future__ import annotations

from flask import Flask, jsonify, request

from _common import error_response

import blob_client

app = Flask(__name__)


@app.get("/api/job_status")
def job_status():
    job_id = request.args.get("job_id")
    if not job_id:
        return error_response("query param 'job_id' is required")

    statuses = blob_client.read_all_recording_statuses(job_id)
    n_ok = sum(1 for s in statuses if s.get("ok"))
    n_error = sum(1 for s in statuses if not s.get("ok"))
    return jsonify({
        "job_id": job_id,
        "n_done": len(statuses),
        "n_ok": n_ok,
        "n_error": n_error,
        "statuses": statuses,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5003)
