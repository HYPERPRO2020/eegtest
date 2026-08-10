"""NeuroQA — thin wrapper around Vercel Blob for this app's job/state model.

Uses the `vercel_blob` package (an unofficial but working Python client for
Vercel Blob's REST API -- there's no official Python SDK, only JS) rather
than hand-rolling the REST protocol (auth header, API version header, etc.)
from memory with no way to verify it against a live account in this
environment. `vercel_blob` only needs requests/tqdm/certifi/urllib3, all
already pulled in transitively by mne (via pooch) -- effectively free.

Job layout in the Blob store (see ARCHITECTURE.md section 2 -- Blob-only
state, no separate KV):
    jobs/{job_id}/manifest.csv           -- the uploaded manifest, verbatim
    jobs/{job_id}/files.json             -- {filename: blob_url} for uploads
    jobs/{job_id}/validation.json        -- validate_batch() output
    jobs/{job_id}/status/{filename}.json -- one recording's process_recording result
    jobs/{job_id}/results.json           -- final aggregate() output

STATUS: not deploy-verified (no live Vercel account in this environment) --
same caveat as the rest of this repo's Vercel-facing code.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import requests
import vercel_blob

JOB_PREFIX = "jobs"


def _download_one(url: str, dest: Path) -> None:
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)


def download_to_tmp(url: str, filename: str) -> Path:
    """Stream a blob down to a fresh temp file, named after the original
    upload (MNE's readers dispatch on file extension) and return its path."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="neuroqa_"))
    dest = tmp_dir / filename
    _download_one(url, dest)
    return dest


def download_group_to_tmp(files: dict[str, str], primary_filename: str) -> Path:
    """Download `primary_filename` plus any same-stem sidecars present in
    `files` (BrainVision's .vhdr needs its .eeg/.vmrk siblings physically
    next to it -- see manifest.BRAINVISION_SIDECARS) into one shared temp
    directory. Returns the primary file's path."""
    from manifest import BRAINVISION_SIDECARS

    tmp_dir = Path(tempfile.mkdtemp(prefix="neuroqa_"))
    primary_dest = tmp_dir / primary_filename
    _download_one(files[primary_filename], primary_dest)

    stem = Path(primary_filename).stem
    for suffix in BRAINVISION_SIDECARS:
        sidecar_name = stem + suffix
        if sidecar_name in files:
            _download_one(files[sidecar_name], tmp_dir / sidecar_name)
    return primary_dest


def _job_path(job_id: str, *parts: str) -> str:
    return "/".join([JOB_PREFIX, job_id, *parts])


def put_json(path: str, obj: dict, add_random_suffix: bool = False) -> str:
    """Write JSON to `path` in the blob store, overwriting any existing blob
    at that exact path (allowOverwrite -- status files get re-written as a
    job progresses). Returns the blob's public URL."""
    data = json.dumps(obj).encode("utf-8")
    resp = vercel_blob.put(path, data, options={
        "addRandomSuffix": "true" if add_random_suffix else "false",
        "allowOverwrite": "true",
    })
    return resp["url"]


def put_text(path: str, text: str) -> str:
    resp = vercel_blob.put(path, text.encode("utf-8"), options={
        "addRandomSuffix": "false", "allowOverwrite": "true",
    })
    return resp["url"]


def get_json(url: str) -> dict:
    """Blobs are public URLs by default -- a plain unauthenticated GET reads
    them back, no token needed (see ARCHITECTURE.md's public-Blob caveat)."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_text(url: str) -> str:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def list_prefix(prefix: str) -> list[dict]:
    """List every blob under `prefix`, paginating through `cursor` until
    exhausted. Returns vercel_blob's raw blob dicts (each has at least
    "url" and "pathname")."""
    out: list[dict] = []
    cursor = None
    while True:
        options = {"prefix": prefix, "limit": "1000"}
        if cursor:
            options["cursor"] = cursor
        resp = vercel_blob.list(options=options)
        out.extend(resp.get("blobs", []))
        if not resp.get("hasMore"):
            break
        cursor = resp.get("cursor")
    return out


# ---- job-specific helpers -------------------------------------------------

def write_manifest(job_id: str, manifest_text: str) -> str:
    return put_text(_job_path(job_id, "manifest.csv"), manifest_text)


def write_files_map(job_id: str, files: dict[str, str]) -> str:
    return put_json(_job_path(job_id, "files.json"), files)


def write_validation(job_id: str, validation: dict) -> str:
    return put_json(_job_path(job_id, "validation.json"), validation)


def write_recording_status(job_id: str, filename: str, status: dict) -> str:
    safe_name = filename.replace("/", "_")
    return put_json(_job_path(job_id, "status", f"{safe_name}.json"), status)


def read_all_recording_statuses(job_id: str) -> list[dict]:
    blobs = list_prefix(_job_path(job_id, "status") + "/")
    return [get_json(b["url"]) for b in blobs]


def write_results(job_id: str, results: dict) -> str:
    return put_json(_job_path(job_id, "results.json"), results)
