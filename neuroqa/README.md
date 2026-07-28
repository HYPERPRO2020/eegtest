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
ranked list of the worst epoch x channel cells. `/analyze` is stateless --
everything (including the waveform, decimated if needed) comes back in one
response, nothing is held server-side between requests.

## Deploy to Vercel

`vercel.json` (repo root) points Vercel's Python runtime at `neuroqa/webapp.py`.
Connect the GitHub repo in the Vercel dashboard (Import Project) -- no CLI
needed, it builds on push.

Worth knowing before relying on this:
- **Dependency size.** `mne` + `scipy` + `numpy` (scipy is a hard dependency
  of mne's own filtering, not something this app pulls in directly) is
  roughly 170-180MB installed. Vercel's serverless Python functions have a
  size ceiling around 250MB -- this fits, but without much room to spare, so
  don't add new dependencies to `requirements.txt` without checking the size
  again.
- **Upload size.** Vercel's own request-size limit (a few MB, depends on
  plan) applies before `webapp.py`'s own `MAX_CONTENT_LENGTH` ever sees the
  request. A large recording that uploads fine locally may get rejected by
  the platform on Vercel.
- **Cold starts + execution time.** Importing mne/scipy from cold plus
  actually running the pipeline (a few seconds locally, warm) can be slow on
  a cold serverless invocation. `vercel.json` raises `maxDuration` to 60s as
  a hedge; whether that's enough -- and whether your plan even honors that
  setting -- depends on the plan.
- **Untested.** This config was written from established Vercel Python/Flask
  deployment patterns, not verified against an actual deploy -- there's no
  Vercel CLI/account in the environment this was built in. If it fails to
  build or times out, the error log from the Vercel dashboard is the next
  debugging step.

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

`ingest.py`/`preprocess.py`/`score.py`'s own CLI output and `study_a.py` /
`study_b.py` / `causal_quality.py` need the extra packages commented out at
the bottom of `requirements.txt` (`pip install pandas scikit-learn
statsmodels autoreject matplotlib`) -- the webapp itself does not, which is
why they're commented out rather than listed normally: keeping them out of
`pip install -r requirements.txt` is what keeps the deployed footprint small.

## Module map

- `bands.py` -- frequency bands, artifact-type -> band mapping, and the
  placeholder `WEIGHT` dict (pending real physics-derived weights).
- `quality_index.py` -- the endpoint-aware penalty/quality computation itself.
- `artifact_detectors.py` -- the six Step 3 detectors (severity in [0, 1]).
- `analyze.py` -- runs the full pipeline on one uploaded file for the webapp,
  including building the (possibly decimated) waveform payload.
- `webapp.py` -- Flask app: serves the page and the single `/analyze` route.
