# NeuroQA

Endpoint-aware EEG quality scoring: existing artifact detectors (blink, muscle,
electrode pop, line noise, motion, cardiac) weighted by how much each detected
artifact's frequency band overlaps the band you're actually measuring, instead
of a single generic "clean vs. dirty" score.

## Run the web UI

```
pip install -r requirements.txt
python webapp.py
```

Then open http://127.0.0.1:5000 and drop in a resting-state `.edf` recorded
with the standard 19-channel 10-20 montage. It shows the grade/quality per
frequency band, a breakdown of which artifact type cost the most points, the
actual filtered waveform with the offending channel/time highlighted, and a
ranked list of the worst epoch x channel cells.

## Batch pipeline (scripts, in order)

1. `ingest.py` -- scans a directory of `.edf` files, validates channels/sample
   rate, writes `outputs/ingest_manifest.csv`.
2. `preprocess.py` -- filters + epochs each recording into `outputs/epochs/*.npz`.
3. `score.py` -- runs the endpoint-aware quality index over every recording,
   writes `outputs/quality_summary.csv`.
4. `faa.py` -- frontal alpha asymmetry per recording.
5. `causal_quality.py` -- proves the quality index is already causal (streaming
   replay vs. batch) and prototypes a causal per-channel baseline.
6. `study_a.py` / `study_b.py` -- FAA across preprocessing pipelines/reference
   schemes, and the group/quality/severity regressions.

`study_a.py`, `study_b.py`, and `causal_quality.py` need the extra packages
listed at the bottom of `requirements.txt`; the webapp itself does not.

## Module map

- `bands.py` -- frequency bands, artifact-type -> band mapping, and the
  placeholder `WEIGHT` dict (pending real physics-derived weights).
- `quality_index.py` -- the endpoint-aware penalty/quality computation itself.
- `artifact_detectors.py` -- the six Step 3 detectors (severity in [0, 1]).
- `analyze.py` -- runs the full pipeline on one uploaded file for the webapp.
- `webapp.py` -- Flask app: upload, `/analyze`, `/waveform/<id>`.
