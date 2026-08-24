"""NeuroQA Study A — how much does FAA move with the preprocessing choice?

Computes FAA for the same recording under 5 preprocessing pipelines x 2
reference schemes (10 combinations), then Study A's aggregation reports the
spread of FAA across those 10 choices, per subject. The point isn't which
pipeline is "best" -- it's how much an analysis conclusion (FAA) would
change depending on an arbitrary-looking preprocessing choice, and where
"ours" (the endpoint-aware quality index used as continuous epoch weights,
instead of a hard reject/keep decision) lands relative to the other four.

Pipelines:
  raw       -- filter only, no rejection, flat average over all epochs.
  ica       -- ICA-clean ocular components (no real EOG channel in typical
               resting-state uploads, so mne.preprocessing.ICA.find_bads_eog
               uses Fp1/Fp2 as frontal proxies), then flat average.
  generic   -- fixed 150 uV peak-to-peak epoch rejection, flat average over
               kept epochs.
  autoreject -- autoreject.AutoReject, per-channel/epoch adaptive rejection
               and interpolation, flat average over the cleaned epochs.
  ours      -- no rejection; every epoch kept but weighted by
               quality_index.py's alpha-endpoint quality at F3/F4
               respectively (continuous weighting instead of a hard cutoff).

Reference schemes:
  original  -- whatever reference the recording ships with, as uploaded.
  average   -- re-referenced to the average of whichever channels this
               recording has (see preprocess.STANDARD_1020_ORDER).

Generalized from the previous version (which assumed a fixed 19-channel
HUSM montage and read from a local dataset directory): channel selection and
montage now come from preprocess.py's canonicalization, and the function
below runs per recording, driven by whatever the caller (pipeline.py, for
an uploaded batch) passes in -- no fixed dataset path or manifest CSV.
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
from autoreject import AutoReject

from bands import EEG_BANDS
from faa import compute_faa
from manifest import _read_raw_any, canonical_channel_name
from preprocess import EPOCH_SEC, H_FREQ, L_FREQ, LINE_FREQ, STANDARD_1020_ORDER
from quality_index import compute_quality

mne.set_log_level("ERROR")

OVERLAP_SEC = 2.0
PIPELINES = ["raw", "ica", "generic", "autoreject", "ours"]
REFERENCES = ["original", "average"]
GENERIC_REJECT_UV = 150.0

# autoreject needs real channel positions to do its spatial interpolation.
# Uploaded files may use either the older 10-20 labels (T3/T4/T5/T6) or the
# modern equivalents MNE's standard_1020 montage expects (T7/T8/P7/P8);
# preprocess.py's canonicalization keeps the older labels (matches
# STANDARD_1020_ORDER), so rename to the modern ones locally, only for the
# montage/autoreject/ICA path in this module -- nothing else needs it.
RENAME_1020 = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


def load_referenced_raw(path: str, reference: str, line_freq: float = LINE_FREQ) -> mne.io.BaseRaw:
    raw = _read_raw_any(Path(path))
    raw.load_data()
    raw.rename_channels({ch: canonical_channel_name(ch) for ch in raw.ch_names})
    present = [ch for ch in STANDARD_1020_ORDER if ch in raw.ch_names]
    if "F3" not in present or "F4" not in present:
        raise ValueError(f"F3/F4 not found in {path} after canonicalization")
    raw.pick(present)
    raw.reorder_channels(present)
    # Only rename the keys actually present -- a recording that already uses
    # the modern labels (T7/T8/P7/P8, e.g. ds003478) has none of RENAME_1020's
    # old-style keys (T3/T4/T5/T6) in raw.ch_names at this point, and MNE's
    # rename_channels hard-fails on any mapping key it can't find rather than
    # skipping it (confirmed by a real crash on ds003478's sub-052: "Invalid
    # channel name(s) {'T3','T4','T5','T6'} are not present in info").
    raw.rename_channels({old: new for old, new in RENAME_1020.items() if old in raw.ch_names})
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"),
                     match_case=False, on_missing="warn", verbose=False)
    if reference == "average":
        raw.set_eeg_reference("average", verbose=False)
    # "original": no set_eeg_reference call -- keep whatever reference the
    # recording shipped with, as uploaded.
    raw.notch_filter(line_freq, verbose=False)
    raw.filter(L_FREQ, H_FREQ, verbose=False)
    return raw, [RENAME_1020.get(ch, ch) for ch in present]


def make_epochs(raw: mne.io.BaseRaw) -> mne.Epochs:
    return mne.make_fixed_length_epochs(
        raw, duration=EPOCH_SEC, overlap=OVERLAP_SEC, preload=True, verbose=False,
    )


def ica_clean(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """ICA-clean ocular components. No real EOG channel expected, so
    find_bads_eog uses Fp1/Fp2 as frontal proxies when present."""
    ica = mne.preprocessing.ICA(n_components=min(15, len(raw.ch_names) - 1),
                                 random_state=97, max_iter="auto", verbose=False)
    ica.fit(raw, verbose=False)
    proxy_channels = [ch for ch in ("Fp1", "Fp2") if ch in raw.ch_names]
    try:
        if proxy_channels:
            eog_idx, _ = ica.find_bads_eog(raw, ch_name=proxy_channels, verbose=False)
            ica.exclude = eog_idx
        else:
            ica.exclude = []
    except Exception:
        ica.exclude = []
    cleaned = raw.copy()
    ica.apply(cleaned, verbose=False)
    return cleaned


def faa_pipeline_raw(epochs: mne.Epochs, ch_names: list[str], sfreq: float) -> float:
    data_uv = epochs.get_data() * 1e6
    return compute_faa(data_uv, ch_names, sfreq)["faa"]


def faa_pipeline_ica(raw: mne.io.BaseRaw, ch_names: list[str], sfreq: float) -> float:
    raw_clean = ica_clean(raw)
    epochs = make_epochs(raw_clean)
    return faa_pipeline_raw(epochs, ch_names, sfreq)


def faa_pipeline_generic(epochs: mne.Epochs, ch_names: list[str], sfreq: float) -> float:
    kept = epochs.copy().drop_bad(reject=dict(eeg=GENERIC_REJECT_UV * 1e-6), verbose=False)
    if len(kept) == 0:
        kept = epochs  # degenerate fallback: everything rejected, use unrejected set
    return faa_pipeline_raw(kept, ch_names, sfreq)


def faa_pipeline_autoreject(epochs: mne.Epochs, ch_names: list[str], sfreq: float, seed: int) -> float:
    ar = AutoReject(random_state=seed, n_jobs=1, verbose=False)
    kept = ar.fit_transform(epochs.copy())
    if len(kept) == 0:
        kept = epochs
    return faa_pipeline_raw(kept, ch_names, sfreq)


def faa_pipeline_ours(epochs: mne.Epochs, ch_names: list[str], sfreq: float) -> float:
    data_uv = epochs.get_data() * 1e6
    quality = compute_quality(data_uv, ch_names, sfreq, EEG_BANDS["alpha"])["quality"]
    i3, i4 = ch_names.index("F3"), ch_names.index("F4")
    return compute_faa(data_uv, ch_names, sfreq,
                        weights_f3=quality[:, i3], weights_f4=quality[:, i4])["faa"]


def run_all_pipelines(path: str, seed: int = 0, line_freq: float = LINE_FREQ) -> dict[tuple[str, str], float]:
    """FAA for one recording under every (pipeline, reference) combination.

    Returns {(pipeline_name, reference_name): faa_value} — 10 entries
    (5 pipelines x 2 references) for a recording with a usable montage.
    """
    out: dict[tuple[str, str], float] = {}
    for reference in REFERENCES:
        raw, ch_names = load_referenced_raw(path, reference, line_freq=line_freq)
        sfreq = raw.info["sfreq"]
        epochs = make_epochs(raw)
        out[("raw", reference)] = faa_pipeline_raw(epochs, ch_names, sfreq)
        out[("ica", reference)] = faa_pipeline_ica(raw, ch_names, sfreq)
        out[("generic", reference)] = faa_pipeline_generic(epochs, ch_names, sfreq)
        out[("autoreject", reference)] = faa_pipeline_autoreject(epochs, ch_names, sfreq, seed)
        out[("ours", reference)] = faa_pipeline_ours(epochs, ch_names, sfreq)
    return out


def spread_stats(rows: list[dict]) -> dict:
    """Given a list of {"file", "group", "pipeline", "reference", "faa"} rows
    (long format, as pipeline.py accumulates them across an uploaded batch),
    compute the per-recording FAA range/std across the 10 combinations, and
    the wide table the client-side parallel-coordinates plot needs.

    Returns {"long": rows, "wide": [...], "combo_cols": [...]}.
    """
    by_file: dict[str, dict] = {}
    for r in rows:
        combo = f"{r['pipeline']}_{r['reference']}"
        entry = by_file.setdefault(r["file"], {"file": r["file"], "group": r["group"], "values": {}})
        entry["values"][combo] = r["faa"]

    combo_cols = sorted({f"{p}_{ref}" for p in PIPELINES for ref in REFERENCES})
    wide = []
    for entry in by_file.values():
        vals = [entry["values"][c] for c in combo_cols if c in entry["values"]]
        row = {"file": entry["file"], "group": entry["group"], **entry["values"]}
        if vals:
            row["faa_range"] = max(vals) - min(vals)
            row["faa_std"] = float(np.std(vals))
        wide.append(row)

    return {"long": rows, "wide": wide, "combo_cols": combo_cols}
