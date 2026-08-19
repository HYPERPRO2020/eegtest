# NeuroQA

Testing whether frontal alpha asymmetry (FAA), a long-studied EEG depression
marker, is partly an artifact of muscle-noise contamination rather than a
real brain signal. The core piece is an **endpoint-aware quality scorer**:
existing artifact detectors (blink, muscle, electrode pop, line noise,
motion, cardiac) weighted by how much each detected artifact's frequency
band overlaps the band you're actually measuring, instead of a single
generic "clean vs. dirty" score. See `../ARCHITECTURE.md` for the Vercel
deployment design and what's/isn't verified.

## Two ways to use this

**1. Single-file quick grader** (`/`) — drop in one resting-state recording,
see its quality grade per band, a breakdown of which artifact type cost the
most points, the filtered waveform with offending moments highlighted, and
a ranked list of the worst epoch x channel cells.

**2. Study runner** (`/study`) — upload a *labeled batch* (diagnosis +
severity per recording via a manifest) and run the full pipeline: per-file
validation with a specific reason for any rejection, Study A (how much the
depressed-vs-healthy FAA difference moves across 5 preprocessing pipelines x
2 reference schemes), and Study B (is contamination itself tied to
diagnosis, does it confound FAA, and does quality alone leak group above
chance). This is the deliverable the project brief actually asks for; the
single-file grader is a smaller, useful side tool built on the same scorer.

Both accept any format MNE reads well: .edf, .bdf, .cnt, .set, .fif, or a
.vhdr/.eeg/.vmrk BrainVision triplet.

## Run locally

```
pip install -r requirements.txt
python webapp.py
```
Then open http://127.0.0.1:5000.

`/study`'s upload flow needs Vercel Blob (client uploads + job state, see
`../ARCHITECTURE.md`) so it only works end to end once deployed. To run the
same pipeline locally without Vercel at all:
```
python run_local.py <folder-of-recordings> <manifest.csv> --out outputs/
```
`manifest.csv` needs filename/diagnosis/severity columns (or common
synonyms — see `manifest.parse_manifest_csv`).

## Tests

```
pip install -r requirements.txt pytest
python -m pytest tests/ -v
```
Hand-checks the scorer against synthetic clean vs. artifact-injected
recordings (confirms it's genuinely endpoint-aware: the same planted
blink/EMG costs far less alpha-band quality than delta-band quality),
manifest validation, and reproducibility (same input -> bit-identical
output).

## Module map

- `bands.py` — frequency bands, artifact-type -> band mapping, and the
  placeholder `WEIGHT` dict (pending real physics-derived weights from
  Peter — see the module docstring, not tuned against any result here).
- `artifact_detectors.py` — the six Step 3 detectors (severity in [0, 1]).
- `quality_index.py` — the endpoint-aware penalty/quality computation.
- `faa.py` — frontal alpha asymmetry, optionally quality-weighted.
- `manifest.py` — upload validation: labeled, F3/F4 present, heuristic
  "still looks raw, not already cleaned/re-referenced" checks. Severity is
  soft-required (a present-but-bad value hard-fails; a genuinely absent one
  just warns and disables Test B.1 for that batch, see study_b.py).
- `preprocess.py` — channel-name canonicalization, filtering, epoching.
- `score.py` — runs the detectors + quality index across every band.
- `study_a.py` — the 5-pipeline x 2-reference FAA sweep, per recording.
- `study_b.py` — the three group/quality/severity/FAA regressions.
  Regression 1 (quality~group+severity) uses each recording's *clinical*
  severity from the manifest, not an EEG-derived quantity, and is skipped
  (not faked) for batches that have none.
- `pipeline.py` — ties the above together; imported by both `run_local.py`
  and `webapp.py`'s `/api/*` routes, so there's exactly one implementation
  of "how a recording gets scored," not one per entry point.
- `blob_client.py` — thin Vercel Blob wrapper for the job/state model.
- `analyze.py` — single-file quick-grader analysis logic.
- `webapp.py` — the whole app: quick grader (`/`, `/analyze`), study runner
  page (`/study`), and the upload-batch job model (`/api/*`) — one Flask
  app, one Vercel function (see `../ARCHITECTURE.md`'s post-deploy notes
  for why it's not split up).
- `run_local.py` — batch CLI, no Vercel needed.
- `tests/test_pipeline.py` — synthetic-data hand-checks.

`causal_quality.py` (the streaming/causal scorer prototype) was removed —
explicitly out of scope for this build per the project brief.
