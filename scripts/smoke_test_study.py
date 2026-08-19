"""One-off smoke test of the live /study upload-batch job model end to end
against the real production deployment: upload real files to Blob, then
drive create_job -> process_recording (x4) -> aggregate exactly like
study.html's browser JS would, over HTTP. Verifies the parts ARCHITECTURE.md
flagged as never having been checked against a live account: the
vercel_blob write path, and the Flask routes' handling of real Blob URLs.

Not meant to be kept long-term -- a throwaway verification script, not part
of the app itself.
"""
import json
import os
import sys
from pathlib import Path

import requests
import vercel_blob


def _load_env_local(path: Path) -> None:
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"'))


_load_env_local(Path(__file__).resolve().parent.parent / ".env.local")

BASE = "https://eegtest.vercel.app"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "mumtaz"
FILES = ["H_S1_EC.edf", "H_S2_EC.edf", "MDD_S10_EC.edf", "MDD_S13_EC.edf"]
MANIFEST_CSV = "filename,diagnosis,severity\n" + "\n".join(
    f"{f},{'healthy' if f.startswith('H_') else 'depressed'}," for f in FILES
)

token = os.environ.get("BLOB_READ_WRITE_TOKEN")
assert token, "BLOB_READ_WRITE_TOKEN not found in .env.local"

print("uploading files to Blob...")
blob_urls = {}
for f in FILES:
    data = (DATA_DIR / f).read_bytes()
    result = vercel_blob.put(f"smoketest/{f}", data, options={"token": token, "addRandomSuffix": "true"})
    blob_urls[f] = result["url"]
    print(f"  {f} -> {result['url']}")

print("\nPOST /api/create_job")
r = requests.post(f"{BASE}/api/create_job", json={"manifest_csv": MANIFEST_CSV, "files": blob_urls}, timeout=60)
r.raise_for_status()
job = r.json()
print(json.dumps({k: v for k, v in job.items() if k != "accepted"}, indent=2))
print(f"accepted: {[a['filename'] for a in job['accepted']]}")
if job["rejected"]:
    print(f"REJECTED: {job['rejected']}")
assert job["group_ok"], f"batch not usable: {job['group_reasons']}"

job_id = job["job_id"]
files_map_url = job["files_map_url"]
subsample = set(job["study_a_subsample"])

print(f"\nPOST /api/process_recording x{len(job['accepted'])}")
for rec in job["accepted"]:
    body = {
        "job_id": job_id, "filename": rec["filename"], "diagnosis": rec["diagnosis"],
        "severity": rec["severity"], "run_study_a": rec["filename"] in subsample,
        "files_map_url": files_map_url,
    }
    resp = requests.post(f"{BASE}/api/process_recording", json=body, timeout=120)
    resp.raise_for_status()
    status = resp.json()
    print(f"  {rec['filename']}: ok={status['ok']}  error={status.get('error')}  "
          f"study_a={'yes' if status.get('study_a') else 'no'}")
    if not status["ok"]:
        sys.exit(f"process_recording failed for {rec['filename']}: {status.get('error')}")

print("\nPOST /api/aggregate")
r = requests.post(f"{BASE}/api/aggregate", json={"job_id": job_id}, timeout=60)
r.raise_for_status()
results = r.json()
print(json.dumps(results, indent=2, default=str)[:3000])
print("\nSMOKE TEST PASSED")
