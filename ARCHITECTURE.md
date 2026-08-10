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

## 3. What's still unverified

There is no Vercel account/CLI in this environment (same limitation the
previous session hit). Every number above is either a real local
measurement (`pip install -t`) or Vercel's own current published docs
(fetched live during this session), not a guess — but the actual deploy,
cold-start behavior, and Blob/upload wiring have not been exercised against
a live Vercel project. Flagged the same way the previous session flagged
its own Flask deploy: follows documented patterns, not yet deploy-verified.
