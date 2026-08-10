"""Shared setup for every api/*.py Vercel function: puts neuroqa/ on
sys.path (these functions live in a sibling directory, not a package under
neuroqa/) so `from manifest import ...` etc. work, same reasoning as
webapp.py's own sys.path.insert -- Vercel's Python loader imports each
function module by absolute path and does not walk up to add sibling
directories the way `python neuroqa/webapp.py` does locally.

Also a couple of tiny Flask helpers reused by every route: uniform JSON
error responses and a request-id-free job id generator.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "neuroqa"))

from flask import jsonify  # noqa: E402 (must follow the sys.path insert above)


def new_job_id() -> str:
    return uuid.uuid4().hex[:12]


def error_response(message: str, status: int = 400):
    return jsonify({"error": message}), status
