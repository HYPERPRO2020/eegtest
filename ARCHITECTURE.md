# NeuroQA — architecture spike (upload-driven Phase 1)

This repo's prior state (see `git log`) built NeuroQA against a single fixed,
pre-downloaded dataset (`v1/Data`, Mumtaz/HUSM) with labels parsed out of
filenames — a single-file "upload and grade" webapp, plus batch scripts run
locally against that fixed dataset. The current brief instead requires
**users to upload their own labeled batch of recordings** through the web
UI, with everything (ingestion, scoring, Study A, Study B) run against
whatever they upload. This doc is the spike required before scaffolding that:
does it actually fit on Vercel, and what job/state model does it need.

## 1. Dependency size — measured, not guessed

The previous README estimated "mne + scipy + numpy ~170-180MB, no room to
spare," assumed a 250MB ceiling, and called the whole Vercel config
"untested." Both the estimate and the ceiling were wrong:

- **Real installed size**, measured with `pip install -t <dir>` (what
  Vercel's Python builder actually does), not just summing wheel sizes:
  - `mne + numpy + scipy + Flask + Werkzeug`: **348 MB**. mne 1.12.1 makes
    `matplotlib` an unconditional runtime dependency now (not just a plotting
    extra) — `mne.channels` imports `mne.bem` imports `mne.viz.misc` at
    module load time, so `import mne` hard-fails without matplotlib
    installed, confirmed empirically. matplotlib + pillow + fonttools +
    kiwisolver + contourpy account for ~95MB of that.
  - Adding `scikit-learn + statsmodels + autoreject + pandas` on top:
    **557 MB** raw, **398 MB** after stripping `__pycache__`/`.pyc` (pip
    compiles bytecode during install; a real Vercel build should be told not
    to, or should strip it, to keep the margin below honest).
- **The actual limit**: Vercel's function size limit is **500MB uncompressed
  for the Python runtime specifically** (confirmed from Vercel's current
  docs, fetched live — https://vercel.com/docs/functions/limitations,
  "For Python functions, the maximum uncompressed size is 500 MB"), not the
  250MB that applies to Node/other runtimes. The prior README's 250MB
  assumption was the wrong number for this runtime.

**Conclusion: everything fits in one function's dependency set**, with ~100MB
of headroom (398MB / 500MB) even including the full `autoreject` package,
`scikit-learn`, `statsmodels`, and `pandas` alongside `mne`. No need to trim
`autoreject`, reimplement it narrowly, or split scoring/study code across
functions with different dependency sets — the fallback the brief
anticipated ("if it doesn't fit, trim... or split") turned out not to be
required. One `requirements.txt`, one dependency set, several route files.
Action item carried into the build: strip `__pycache__` before deploy (or
disable pyc compilation in the build step) to keep the real 100MB margin
instead of the 557MB raw number's much thinner one.

## 2. Job model

Confirmed from the same docs fetch:
- Max function duration: 300s (Hobby) / up to 800s, 1800s extended-beta
  (Pro+). Plenty for one recording's ICA + autoreject pass (~10-20s warm,
  more on a cold start importing mne/scipy).
- Request/response body cap: **4.5MB** — confirms raw EEG files (tens of MB
  each, many per upload) cannot be routed through a function body and must
  go straight from the browser to Vercel Blob.
- No built-in queue on the base product. Given the size problem is solved
  (no need to split by dependency weight), the simplest reliable-enough job
  model for Phase 1: the **browser fans out**, firing one request per
  recording (parallel) at a `/api/process-recording` endpoint right after
  upload completes, instead of a single request processing the whole batch.
  Each invocation handles exactly one recording — matches "fan out per
  recording, not one giant call" from the brief. Downside, stated plainly:
  if the browser tab closes mid-run, in-flight recordings stall (no
  server-owned queue to resume them). Vercel does offer "Workflows" for
  durable, resumable multi-step jobs beyond a single invocation's duration —
  a better fit long-term, left as a noted future improvement rather than
  built now, to keep Phase 1's moving parts to what's actually been verified.
- **State**: Vercel Blob only, no separate KV — a small per-recording status
  JSON (`jobs/{job_id}/status/{recording_id}.json`) plus a manifest and
  final results JSON in the same job's Blob prefix. The status/results
  endpoint lists and reads these small JSON blobs. Picked Blob-only over
  provisioning Vercel KV/Redis because it's one storage primitive already
  needed for the files themselves, and this environment has no way to
  provision or test a real KV instance — one mechanism, used consistently,
  per the brief's own instruction.
- **Upload**: Vercel Blob client uploads (browser → Blob directly, bypassing
  the 4.5MB function body cap) require the official `@vercel/blob/client`
  token handshake, which is Node-SDK-coupled — not something safe to
  hand-roll in Python without a way to verify it against a real deployment.
  One small Node function (`api/blob-upload-token.js`) implements just that
  handshake using the real SDK; everything else stays Python.

## 3. What got built, and what's still unverified

Implemented per the plan above:
- `neuroqa/manifest.py` — upload validation (labeled, has severity, F3/F4
  present, heuristic "still looks raw" checks), replacing the old
  filename-parsed fixed-dataset `ingest.py` (deleted).
- `neuroqa/preprocess.py`, `score.py`, `faa.py`, `study_a.py`, `study_b.py`,
  `pipeline.py` — generalized to run on an arbitrary uploaded batch's
  channel subsets instead of one fixed 19-channel dataset montage, and
  restructured as importable functions rather than scripts that assume a
  local `outputs/ingest_manifest.csv`.
- `neuroqa/run_local.py` — runs the same functions the API routes call,
  against a local folder + manifest.csv, no Vercel needed. This is how the
  pipeline (including a real Study A pipeline sweep with actual ICA +
  AutoReject, and Study B's regressions/classifier) was hand-checked end to
  end in this environment, on synthetic multi-recording data
  (`neuroqa/tests/test_pipeline.py` plus an ad hoc synthetic-batch run) —
  the exact API route logic itself was additionally exercised against a
  fake in-memory Blob store standing in for `vercel_blob`.
- `neuroqa/webapp.py` — as of the third post-deploy fix below, this one
  Flask app is the entire backend: the quick grader, the study-runner page,
  *and* the fan-out job model (create_job/process_recording/job_status/
  aggregate), all as routes on one Vercel Python function. Uses
  `vercel_blob` (an unofficial but real, working PyPI package implementing
  Blob's REST protocol — there's no official Python SDK) instead of
  hand-rolling that protocol from memory.
- `api/blob-upload-token.js` — the one Node function, using the real
  `@vercel/blob/client` SDK for the client-upload token handshake.
- `neuroqa/templates/study.html` — the upload/manifest/progress/results UI,
  importing `@vercel/blob/client` from an ESM CDN (`esm.sh`) in the browser
  since there's no bundler in this project — also unverified, same caveat.
- All Study A/B randomness (ICA, AutoReject, cross-validation splits) is
  seeded from one constant (`pipeline.SEED = 0`); `run_local.py`'s "same
  upload -> same result" was hand-checked, plus two automated determinism
  tests in `test_pipeline.py`.

**Post-deploy fix**: the first real deploy attempt failed. `vercel.json` had
combined the legacy `builds` array with a top-level `functions` key — Vercel
rejects that combination outright ("The `functions` property cannot be used
in conjunction with the `builds` property"). Fixed by moving each function's
`maxDuration` into its own `builds` entry's `config` object instead, which
legacy mode does support. Also brought `process_recording.py`'s
`maxDuration` down from 800s to 300s: the 800s/1800s extended durations
need Pro/Enterprise ([confirmed in Vercel's docs](https://vercel.com/docs/functions/limitations#max-duration)
— Hobby is capped at 300s default *and* maximum), and this deploy's plan
tier isn't known from this environment. If you're on Pro+ and batches are
timing out on a slow Study A sweep (ICA + AutoReject per recording), raise
`neuroqa/webapp.py`'s `maxDuration` back up in `vercel.json`.

**Second post-deploy fix**: the next deploy attempt failed differently —
`ENOENT: no such file or directory, lstat '.../pycache/vercel/path0/.vercel/
python/.venv/lib/python3.12/site-packages/requests/_types.cpython-312.pyc'`
(the path is self-referential/duplicated, not a real path). This isn't
something in the app's code; it matches a reported class of Vercel Python
build-infra bug where the builder's own bytecode-cache path computation
breaks when several `@vercel/python` builds in one project independently
pip-install overlapping heavy dependencies (mne/scipy/numpy/... here) —
this project had five (`webapp.py` + 4 separate api/*.py routes). First
attempt: consolidated the four API routes into one Flask app (`api/
index.py`, one Vercel function), cutting Python builds from five to two.

**Third post-deploy fix**: the *same* ENOENT error recurred with just two
Python builds, which rules out "many builds" as the trigger and points more
specifically at two *concurrent* Python builds racing on Vercel's shared
bytecode-cache directory — a classic TOCTOU (two builds precompiling the
same package's `.pyc` into the same cache path at the same time, one
finishing and invalidating the path the other mid-flight `lstat`s). The
conclusive fix: merged `api/index.py` into `neuroqa/webapp.py` (same four
`/api/*` URL paths, so `study.html`'s `fetch()` calls didn't change) so
there is exactly **one** Python build in the whole project — removing the
concurrency removes the race by construction, not just by probability.
`api/blob-upload-token.js` stays separate since it has to be Node, and a
Node build can't race a Python build on a Python-specific bytecode cache.
If this project ever needs to split back into multiple Python functions
(e.g. it outgrows one function's 500MB/one-region budget), watch for this
error returning and treat it as a Vercel platform issue to raise with
their support, not a config mistake to keep re-guessing at.

Still unverified (no Vercel account/CLI in this environment, same
limitation the previous session hit): the actual deploy, `vercel.json`'s
legacy `builds`/`routes` wiring, cold-start behavior, and the real Blob
client-upload handshake end to end
(the mocked test above stands in for it, but a fake store can't catch a
wrong header name or a real auth failure). Every dependency-size and
job-model number is a real local measurement or Vercel's current published
docs, not a guess — the wiring on top of those numbers is not deploy-tested.

**Privacy note, not addressed by this build**: Vercel Blob objects are
public URLs by default, and this app uses that default (`access: 'public'`
in `blob-upload-token.js`) to keep the client-upload flow simple. Uploaded
recordings are paired with a diagnosis label — that's sensitive health
data sitting behind an unguessable-but-unauthenticated URL, not access
control. A real deployment handling real patient data would need Blob's
private-access mode plus signed download URLs (`vercel_blob`'s `head`/
`get` with a token, or the JS SDK's private-store equivalent) before going
anywhere near real recordings. Flagging this now rather than silently
shipping it, since it's a correctness-of-scope issue, not a nice-to-have.
