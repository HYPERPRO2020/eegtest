"""NeuroQA Step 2 — preprocessing.

Filters an uploaded recording and segments it into fixed-length epochs. This
step does **not** reject anything — rejection is Step 3's job. Step 2 only
canonicalizes channel names, filters, and cuts.

Pipeline per recording:
    read whatever MNE-supported format it is (see manifest.READERS)
    -> canonicalize channel names, keep only recognized 10-20 channels
       (F3/F4 guaranteed present -- manifest.validate_recording already
       checked that before this ever runs)
    -> 50 Hz notch (override with LINE_FREQ=60.0 for US-mains recordings)
    -> 0.5-45 Hz bandpass
    -> 4-second epochs, 50% overlap (2-second step)

Generalized from the previous version, which required a fixed 19-channel
HUSM-dataset montage and only read .edf. Different uploaded recordings can
legitimately carry different channel subsets (only F3/F4 is a hard
requirement, see manifest.py) -- callers index channels by name
(`ch_names.index(...)`), never by a fixed position, so a variable per-file
channel list is safe throughout the rest of the pipeline.
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np

from manifest import STANDARD_1020, _read_raw_any, canonical_channel_name

mne.set_log_level("ERROR")

L_FREQ, H_FREQ = 0.5, 45.0
LINE_FREQ = 50.0  # mains hum frequency; override to 60.0 for US-acquired recordings
EPOCH_SEC = 4.0
OVERLAP_SEC = 2.0  # 50% overlap

# Fixed reference ORDER (not a required set) -- when a recording carries a
# given channel, it always lands at this channel's relative position among
# the channels present, so eyeballing two recordings' channel lists side by
# side stays predictable. Everything downstream indexes by name, not
# position, so a recording missing some of these is fine as long as F3/F4
# survive (already enforced by manifest.validate_recording upstream).
STANDARD_1020_ORDER = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "T7", "C3", "Cz", "C4",
    "T4", "T8", "T5", "P7", "P3", "Pz", "P4", "T6", "P8", "O1", "O2",
]


def load_and_filter(path: str | Path, line_freq: float = LINE_FREQ) -> tuple[mne.io.BaseRaw, list[str]]:
    """Load an uploaded recording, canonicalize+select channels, filter.

    line_freq defaults to 50 Hz (this pipeline's original global default,
    still correct for most of the world) -- pass 60.0 for US-mains
    recordings. Matters beyond the notch filter itself:
    artifact_detectors.detect_line_noise() checks specifically 49-51 Hz vs.
    neighbors, so a 60 Hz recording notch-filtered (and detected) at 50 Hz
    both misses the real line-noise band and silently zeros out that
    detector's severity score. Confirmed real for ds003478 specifically --
    its own eeg.json reports PowerLineFrequency: 60 (Univ. of Arizona, US),
    not the 50 Hz this pipeline defaulted to for every dataset before this
    fix.

    Returns (raw, ch_names) with raw already notch+bandpass filtered and
    picked down to `ch_names` (a subset of STANDARD_1020_ORDER, always
    including F3/F4). Split out from preprocess_file() so callers that need
    the continuous (pre-epoching) signal too -- e.g. analyze.py's waveform
    viewer -- don't have to re-filter it themselves.
    """
    raw = _read_raw_any(Path(path))
    raw.load_data()
    raw.rename_channels({ch: canonical_channel_name(ch) for ch in raw.ch_names})

    # Two channels canonicalizing to the same name (rare — e.g. a duplicate
    # export) would break pick(); keep only the first occurrence of each.
    seen = set()
    keep = []
    for ch in raw.ch_names:
        if ch not in seen:
            seen.add(ch)
            keep.append(ch)
    if len(keep) != len(raw.ch_names):
        raw.pick(keep)

    present = [ch for ch in STANDARD_1020_ORDER if ch in raw.ch_names]
    if "F3" not in present or "F4" not in present:
        raise ValueError(
            f"F3/F4 not found after channel-name canonicalization "
            f"(recognized channels: {present or 'none'})"
        )
    raw.pick(present)
    raw.reorder_channels(present)

    raw.notch_filter(line_freq, verbose=False)
    raw.filter(L_FREQ, H_FREQ, verbose=False)
    return raw, present


def preprocess_file(path: str | Path, line_freq: float = LINE_FREQ) -> tuple[np.ndarray, list[str], float]:
    """Load, filter, and epoch one uploaded recording. See load_and_filter
    for line_freq (default 50 Hz, pass 60.0 for US-mains recordings).

    Returns (data_uv, ch_names, sfreq): data_uv is (n_epochs, n_channels,
    n_samples) in microvolts, ch_names is the canonicalized channel list
    actually present. Raises ValueError if F3/F4 don't survive
    canonicalization or the recording is too short to epoch.
    """
    raw, present = load_and_filter(path, line_freq=line_freq)
    epochs = mne.make_fixed_length_epochs(
        raw, duration=EPOCH_SEC, overlap=OVERLAP_SEC, preload=True, verbose=False,
    )
    if len(epochs) == 0:
        raise ValueError(f"recording too short to produce a single {EPOCH_SEC:.0f}s epoch")
    data_uv = epochs.get_data() * 1e6  # MNE returns volts; store microvolts (QC thresholds are in uV)
    return data_uv, present, float(raw.info["sfreq"])
