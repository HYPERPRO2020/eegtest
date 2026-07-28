"""NeuroQA — upload-and-grade web UI.

One page: drop in a .edf, it runs the same endpoint-aware pipeline as
score.py on that single file, and shows a breakdown of exactly where it
lost points -- by artifact type, by channel, and over time -- for whichever
EEG band you're measuring (a live selector, since the whole point of the
quality index is that the score depends on the band). A waveform viewer
lets you scroll the actual filtered multi-channel trace and jump straight
to any flagged moment.

Run:
    .venv/Scripts/python.exe neuroqa/webapp.py
then open http://127.0.0.1:5000
"""

from __future__ import annotations

import tempfile
import traceback
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from analyze import AnalysisError, analyze_edf

MAX_UPLOAD_MB = 100
MAX_CACHED_RECORDINGS = 5  # in-memory waveform cache, evicts oldest past this
MAX_WAVEFORM_WINDOW_SEC = 30.0  # cap per /waveform request so payloads stay small

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# analysis_id -> {"data": (n_channels, n_samples) uV ndarray, "sfreq", "channels", "duration_sec"}
# In-memory only: fine for a single-user local tool, resets if the dev server
# reloads. Bounded size so repeated uploads in one session can't grow unbounded.
_waveform_cache: "OrderedDict[str, dict]" = OrderedDict()


def _cache_waveform(waveform: dict) -> str:
    analysis_id = uuid.uuid4().hex
    _waveform_cache[analysis_id] = waveform
    while len(_waveform_cache) > MAX_CACHED_RECORDINGS:
        _waveform_cache.popitem(last=False)
    return analysis_id


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
            result, waveform = analyze_edf(tmp_path, filename=upload.filename)
        except AnalysisError as e:
            return jsonify({"error": str(e)}), 422
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": f"unexpected error while scoring: {e}"}), 500

    result["analysis_id"] = _cache_waveform(waveform)
    result["duration_sec"] = round(waveform["duration_sec"], 2)
    return jsonify(result)


@app.get("/waveform/<analysis_id>")
def waveform(analysis_id: str):
    cached = _waveform_cache.get(analysis_id)
    if cached is None:
        return jsonify({"error": "no cached recording for this id -- re-upload the file"}), 404

    try:
        start_sec = max(0.0, float(request.args.get("start", 0.0)))
        duration_sec = float(request.args.get("duration", 10.0))
    except ValueError:
        return jsonify({"error": "start/duration must be numbers"}), 400
    duration_sec = min(max(duration_sec, 0.5), MAX_WAVEFORM_WINDOW_SEC)

    data, sfreq = cached["data"], cached["sfreq"]
    n_samples = data.shape[1]
    start_idx = min(int(start_sec * sfreq), n_samples)
    end_idx = min(start_idx + int(duration_sec * sfreq), n_samples)
    window = data[:, start_idx:end_idx]

    return jsonify({
        "start_sec": round(start_idx / sfreq, 4),
        "duration_sec": round((end_idx - start_idx) / sfreq, 4),
        "sfreq": sfreq,
        "channels": cached["channels"],
        "recording_duration_sec": round(n_samples / sfreq, 2),
        "samples": np.round(window, 1).tolist(),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
