"""NeuroQA Step 1 — upload-driven ingestion + validation.

Replaces the previous ingest.py's core assumption (one fixed dataset on
disk, diagnosis parsed out of the filename) with the shape this build
actually needs: a user uploads a *batch* of raw recordings plus a manifest
(filename -> diagnosis -> severity), and every file is validated
independently against that manifest, with a specific reason attached to
each failure rather than a silent drop. Nothing here reads from a fixed
local dataset directory.

Requirements enforced (see project brief, "Upload requirements"):
  - labeled          -- a manifest row with a recognized diagnosis value
  - severity          -- soft-required: a numeric severity score if the
                         manifest row provides one; a missing value is a
                         WARNING (accepted, Test B.1 skipped for it in
                         study_b.py -- some real public datasets, e.g.
                         Mumtaz/HUSM's deposit, ship only a diagnosis label),
                         but a present, unparseable value is still a hard
                         rejection -- see validate_recording.
  - includes F3/F4    -- required for FAA; hard-checked after channel-name
                         canonicalization (raw files spell channels very
                         differently: "EEG F3-Ref", "F3.", "F3-LE", ...)
  - raw, not cleaned  -- continuous Raw data (not Epochs/Evoked), a real
                         multi-channel montage (not a handful of pre-derived
                         channels), long enough to epoch. See
                         `_reference_state_warnings` docstring for what this
                         check can and can't detect -- it's a heuristic,
                         not a certificate that a file was never touched.
  - a group           -- enforced at the batch level (validate_batch), not
                         per file: at least MIN_PER_GROUP accepted
                         recordings for each of at least 2 diagnosis labels.

STATUS: the "not already cleaned/re-referenced" check is a heuristic, not a
guarantee -- EDF/BDF/CNT/SET headers don't reliably carry "this was already
average-referenced" the way MNE's own custom_ref_applied flag does for .fif
files saved by MNE itself. Flagged as a WARNING (accepted, surfaced to the
user) rather than a hard rejection when the file's own header gives no
positive signal either way, per the brief's "reject/flag ... with a clear
reason" -- flagging is the honest answer when the data genuinely can't say.
"""

from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path

import mne

mne.set_log_level("ERROR")

# ---- what counts as "raw EEG this pipeline can read" -----------------------
READERS = {
    ".edf": mne.io.read_raw_edf,
    ".bdf": mne.io.read_raw_bdf,
    ".cnt": mne.io.read_raw_cnt,
    ".set": mne.io.read_raw_eeglab,
    ".fif": mne.io.read_raw_fif,
    ".vhdr": mne.io.read_raw_brainvision,
}
SUPPORTED_SUFFIXES = set(READERS)
# BrainVision ships as a triplet; only .vhdr is "the file" MNE opens, but
# .eeg/.vmrk must be present alongside it -- checked explicitly since a
# missing sidecar fails with a confusing low-level error otherwise.
BRAINVISION_SIDECARS = (".eeg", ".vmrk")

REQUIRED_CHANNELS = {"F3", "F4"}  # hard requirement: FAA can't be computed without both

# 10-20 names used only for the "does this look like a real multi-channel
# montage, not a handful of pre-derived channels" sanity check below --
# NOT a hard per-name requirement beyond F3/F4.
STANDARD_1020 = {
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "T7", "C3", "Cz", "C4",
    "T4", "T8", "T5", "P7", "P3", "Pz", "P4", "T6", "P8", "O1", "O2", "A1", "A2",
}
MIN_MONTAGE_CHANNELS = 8  # below this, "raw" is implausible even with F3/F4 present

MIN_DURATION_SEC = 60.0   # need several 4s epochs; a few seconds isn't a resting recording
MIN_SFREQ = 50.0          # alpha (8-13Hz) and the 20-45Hz EMG band need real bandwidth
MIN_PER_GROUP = 2         # "a group, not a single file" -- per label, at the batch level

_SUFFIX_STRIP = ("EEG ", "-REF", "-Ref", "-ref", "-LE", "-le", "-A1", "-A2")


def canonical_channel_name(raw_name: str) -> str:
    """Best-effort map of an arbitrary EEG file's channel label onto the
    standard 10-20 name it almost certainly means.

    Uploaded files spell channels very differently by vendor/format:
    "EEG F3-Ref", "F3.", "F3-LE", "F3 ". Strips common prefixes/suffixes,
    trailing separators, then matches case-insensitively against
    STANDARD_1020; returns the canonical-cased name on a match, otherwise
    the stripped-but-unmatched original (so an unrecognized channel still
    shows up under a readable name in validation messages, not silently
    dropped).
    """
    name = raw_name.strip()
    for token in _SUFFIX_STRIP:
        if name.startswith(token):
            name = name[len(token):]
        if name.endswith(token):
            name = name[: -len(token)]
    name = name.strip(" .-_").strip()
    for canonical in STANDARD_1020:
        if name.upper() == canonical.upper():
            return canonical
    return name


# ---- manifest parsing --------------------------------------------------

_DIAGNOSIS_SYNONYMS = {
    "healthy": {"healthy", "h", "control", "hc", "non-depressed", "nondepressed", "0", "no"},
    "depressed": {"depressed", "mdd", "d", "case", "patient", "1", "yes"},
}


def normalize_diagnosis(raw_value: str) -> str | None:
    """Map a free-text manifest diagnosis value onto {"healthy","depressed"}.

    Returns None if unrecognized -- caller turns that into a per-file
    validation error rather than guessing. Kept to a strict binary mapping
    (not an open label set) because Study A/B and the FAA group comparison
    are all binary group comparisons; see project brief Study B.
    """
    if raw_value is None:
        return None
    v = raw_value.strip().lower()
    for canonical, synonyms in _DIAGNOSIS_SYNONYMS.items():
        if v in synonyms:
            return canonical
    return None


_FILENAME_KEYS = {"filename", "file", "recording", "name"}
_DIAGNOSIS_KEYS = {"diagnosis", "label", "group", "dx", "class"}
_SEVERITY_KEYS = {"severity", "score", "severity_score", "bdi", "hamd", "hdrs"}


def parse_manifest_csv(text: str) -> dict[str, dict]:
    """Parse a manifest CSV into {filename: {"diagnosis_raw", "severity_raw"}}.

    Column names are matched case-insensitively against a few common
    synonyms per field (see _*_KEYS above) so "file,label,BDI" and
    "filename,diagnosis,severity" both work. Raises ValueError with a
    human-readable message if the required columns can't be identified.
    """
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise ValueError("manifest is empty")

    lower_to_actual = {f.strip().lower(): f for f in reader.fieldnames}

    def find_column(keys: set[str], what: str) -> str:
        for k in keys:
            if k in lower_to_actual:
                return lower_to_actual[k]
        raise ValueError(
            f"manifest is missing a {what} column -- expected one of "
            f"{sorted(keys)}, found columns {reader.fieldnames}"
        )

    file_col = find_column(_FILENAME_KEYS, "filename")
    diag_col = find_column(_DIAGNOSIS_KEYS, "diagnosis")
    sev_col = find_column(_SEVERITY_KEYS, "severity")

    rows: dict[str, dict] = {}
    for row in reader:
        filename = (row.get(file_col) or "").strip()
        if not filename:
            continue
        rows[filename] = {
            "diagnosis_raw": (row.get(diag_col) or "").strip(),
            "severity_raw": (row.get(sev_col) or "").strip(),
        }
    return rows


# ---- per-file validation -------------------------------------------------

@dataclass
class ValidationResult:
    filename: str
    ok: bool
    reasons: list[str] = field(default_factory=list)   # hard-fail reasons (ok is False iff non-empty)
    warnings: list[str] = field(default_factory=list)  # accepted-but-flagged
    diagnosis: str | None = None
    severity: float | None = None
    n_channels: int | None = None
    sfreq: float | None = None
    duration_sec: float | None = None
    channels_found: list[str] | None = None
    md5: str | None = None


def _read_raw_any(path: Path) -> mne.io.BaseRaw:
    suffix = path.suffix.lower()
    reader = READERS.get(suffix)
    if reader is None:
        raise ValueError(
            f"unsupported file type '{suffix}' -- expected one of {sorted(SUPPORTED_SUFFIXES)}"
        )
    if suffix == ".vhdr":
        missing_sidecars = [
            s for s in BRAINVISION_SIDECARS if not path.with_suffix(s).exists()
        ]
        if missing_sidecars:
            raise ValueError(
                f"BrainVision file missing sidecar(s) {missing_sidecars} "
                f"next to {path.name} -- .vhdr/.eeg/.vmrk must be uploaded together"
            )
    return reader(str(path), preload=False, verbose=False)


def _reference_state_warnings(raw: mne.io.BaseRaw, matched: dict[str, str]) -> list[str]:
    """Heuristic 'is this still recognizably raw' checks -- see module
    docstring STATUS note. Only flags what the header can positively
    support; absence of a flag is not proof the file is untouched raw data.
    """
    warnings: list[str] = []
    # MNE's own flag -- only meaningful for .fif files previously saved by
    # MNE after set_eeg_reference(); most raw formats (edf/bdf/cnt/set) never
    # set this, so its absence there means "unknown", not "confirmed raw".
    ref_applied = raw.info.get("custom_ref_applied", 0)
    if ref_applied:
        warnings.append(
            "file header reports a custom EEG reference was already applied "
            "(likely re-referenced/processed before upload, not raw acquisition state)"
        )
    n_montage_hits = sum(1 for name in matched if name in STANDARD_1020)
    if n_montage_hits < MIN_MONTAGE_CHANNELS:
        warnings.append(
            f"only {n_montage_hits} standard 10-20 channels recognized "
            f"(<{MIN_MONTAGE_CHANNELS}) -- looks more like a pre-extracted channel "
            "subset than a full raw montage; check this wasn't already reduced"
        )
    has_mastoid_or_ref = any(
        c.upper() in {"A1", "A2", "M1", "M2", "REF", "LM", "RM"} for c in raw.ch_names
    )
    if not has_mastoid_or_ref and n_montage_hits >= MIN_MONTAGE_CHANNELS:
        warnings.append(
            "no linked-ear/mastoid/reference channel found alongside the montage -- "
            "may already be re-referenced with the original reference channel dropped"
        )
    return warnings


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def validate_recording(path: Path, manifest_row: dict | None) -> ValidationResult:
    """Validate one uploaded file against one manifest row.

    Returns a ValidationResult with `ok=False` and specific `reasons` on any
    hard failure (unreadable, unlabeled, no severity, missing F3/F4, too
    short/low-rate to epoch); `ok=True` recordings may still carry
    `warnings` (see _reference_state_warnings).
    """
    result = ValidationResult(filename=path.name, ok=False)

    if manifest_row is None:
        result.reasons.append("no manifest row for this filename")
    else:
        diagnosis = normalize_diagnosis(manifest_row.get("diagnosis_raw"))
        if diagnosis is None:
            result.reasons.append(
                f"unrecognized diagnosis label '{manifest_row.get('diagnosis_raw')}' -- "
                "expected healthy/control/H or depressed/MDD/D (case-insensitive)"
            )
        else:
            result.diagnosis = diagnosis

        severity_raw = manifest_row.get("severity_raw", "")
        if not severity_raw:
            # Missing is different from garbage: a dataset that genuinely has
            # no clinical severity (e.g. Mumtaz/HUSM's public deposit, which
            # ships only a diagnosis label) is still usable for Study A and
            # Study B's non-severity analyses -- only Test B.1
            # (quality~group+severity) needs this value, and study_b.py skips
            # that one test gracefully when it's absent. A present-but-bogus
            # value (a typo, wrong column) is still a hard rejection below.
            result.warnings.append(
                "no severity score provided -- accepted, but Test B.1 "
                "(quality ~ group + severity) will be skipped for this recording"
            )
        else:
            try:
                result.severity = float(severity_raw)
            except (TypeError, ValueError):
                result.reasons.append(f"severity score '{severity_raw}' is not a number")

    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        result.reasons.append(
            f"unsupported file type '{path.suffix}' -- expected one of {sorted(SUPPORTED_SUFFIXES)}"
        )
        return result  # nothing more we can check without reading it

    try:
        raw = _read_raw_any(path)
    except Exception as e:
        result.reasons.append(f"couldn't read file: {e}")
        return result

    if not isinstance(raw, mne.io.BaseRaw):
        result.reasons.append("file is not continuous raw data (looks like Epochs/Evoked)")
        return result

    result.n_channels = len(raw.ch_names)
    result.sfreq = float(raw.info["sfreq"])
    result.duration_sec = round(raw.n_times / result.sfreq, 1) if result.sfreq else None
    matched = {canonical_channel_name(ch): ch for ch in raw.ch_names}
    result.channels_found = sorted(matched)
    result.md5 = md5sum(path)

    missing = REQUIRED_CHANNELS - set(matched)
    if missing:
        result.reasons.append(
            f"missing required channel(s) {sorted(missing)} "
            f"(found: {', '.join(sorted(matched)) or 'none recognized'})"
        )
    if result.sfreq < MIN_SFREQ:
        result.reasons.append(f"sampling rate {result.sfreq:.1f} Hz is below the {MIN_SFREQ:.0f} Hz minimum")
    if result.duration_sec is not None and result.duration_sec < MIN_DURATION_SEC:
        result.reasons.append(
            f"recording is only {result.duration_sec:.0f}s, need at least {MIN_DURATION_SEC:.0f}s"
        )

    if not result.reasons:
        result.warnings.extend(_reference_state_warnings(raw, matched))

    result.ok = not result.reasons
    return result


@dataclass
class BatchValidationResult:
    accepted: list[ValidationResult]
    rejected: list[ValidationResult]
    duplicate_groups: list[list[str]]  # filenames sharing an md5, among accepted files
    group_ok: bool
    group_reasons: list[str]


def validate_batch(paths: list[Path], manifest_rows: dict[str, dict]) -> BatchValidationResult:
    """Validate every uploaded file, then check the batch as a whole is
    actually a usable group: at least MIN_PER_GROUP accepted recordings
    per diagnosis label. Duplicate detection (by md5) is reported but does
    NOT reject a file on its own -- Study A/B's own aggregation dedupes
    before treating rows as independent, same as the old ingest.py did for
    the fixed dataset (some acquisition pipelines legitimately re-export
    the same recording under two filenames).
    """
    results = [validate_recording(p, manifest_rows.get(p.name)) for p in paths]
    accepted = [r for r in results if r.ok]
    rejected = [r for r in results if not r.ok]

    by_md5: dict[str, list[str]] = {}
    for r in accepted:
        by_md5.setdefault(r.md5, []).append(r.filename)
    duplicate_groups = [names for names in by_md5.values() if len(names) > 1]

    counts: dict[str, int] = {}
    for r in accepted:
        counts[r.diagnosis] = counts.get(r.diagnosis, 0) + 1

    group_reasons = []
    if len(counts) < 2:
        group_reasons.append(
            f"only {len(counts)} diagnosis label(s) present among accepted recordings "
            "-- need both healthy and depressed recordings to run a group comparison"
        )
    for label, n in counts.items():
        if n < MIN_PER_GROUP:
            group_reasons.append(
                f"only {n} accepted recording(s) labeled '{label}' "
                f"(<{MIN_PER_GROUP}) -- too few to treat as a group"
            )

    return BatchValidationResult(
        accepted=accepted,
        rejected=rejected,
        duplicate_groups=duplicate_groups,
        group_ok=not group_reasons,
        group_reasons=group_reasons,
    )
