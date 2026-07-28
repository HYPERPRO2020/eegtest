"""NeuroQA — upload-and-grade web UI.

One page: drop in a .edf, it runs the same endpoint-aware pipeline as
score.py on that single file, and shows a breakdown of exactly where it
lost points -- by artifact type, by channel, and over time -- for whichever
EEG band you're measuring (a live selector, since the whole point of the
quality index is that the score depends on the band). A waveform viewer
lets you scroll the actual filtered multi-channel trace and jump straight
to any flagged moment.

Stateless by design: /analyze does everything in one request/response, no
server-side session between requests (see analyze.py's waveform payload) --
this is a Flask app for local use, but it's also meant to survive being
deployed as a serverless function (e.g. Vercel), where a later request has
no guarantee of hitting the same process, let alone the same memory.

Run:
    .venv/Scripts/python.exe neuroqa/webapp.py
then open http://127.0.0.1:5000
"""

from __future__ import annotations

import tempfile
import traceback
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from analyze import AnalysisError, analyze_edf

MAX_UPLOAD_MB = 100  # local Flask use; Vercel's own request-size limit (much
# smaller, a few MB depending on plan) applies first when deployed there --
# this just caps it for direct/local use, it doesn't raise Vercel's ceiling.

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
# Flask indent/newline-pretty-prints JSON by default -- fine for a small
# payload, but the waveform array has hundreds of thousands of numbers, so
# one-value-per-line more than doubles the response size. Force compact.
app.json.compact = True


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/analyze")
def analyze():
    upload = request.files.get("edf")
    if upload is None or upload.filename == "":
        return jsonify({"error": "no file uploaded"}), 400

    filename = secure_filename(upload.filename)
    if not filename.lower().endswith(".edf"):
        return jsonify({"error": "please upload a .edf file"}), 400

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / filename
        upload.save(tmp_path)
        try:
            result = analyze_edf(tmp_path, filename=upload.filename)
        except AnalysisError as e:
            return jsonify({"error": str(e)}), 422
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": f"unexpected error while scoring: {e}"}), 500

    return jsonify(result)


# Vercel's Python runtime (@vercel/python) looks for a WSGI-compatible `app`
# in this file -- Flask's `app` object already is one, nothing extra needed.
if __name__ == "__main__":
    app.run(debug=True, port=5000)
