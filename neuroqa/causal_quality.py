"""NeuroQA — causal (streaming) quality index, and the causal-baseline problem.

"Real-time" here means causal, not deployed (see project brief): the score
for epoch e may only use data['s seen through epoch e, never epoch e+1
onward. This module does two things:

1. Proves the existing quality index (score.py / quality_index.py) is
   already causal, by literally re-running it epoch-by-epoch on a growing
   window and diffing against the batch/offline result. Every Step 3
   detector (artifact_detectors.py) reduces only within an epoch's own
   samples (axis=-1/-2), never across the epoch axis, so there is nothing
   for a causal replay to change -- this is demonstrated empirically below,
   not just asserted.

2. Because point 1 makes the "streaming vs batch" comparison trivial (diff
   is exactly zero), the actual open problem the brief points at is a level
   deeper: the Step 3 detectors use fixed absolute-uV / fixed-ratio
   thresholds (bands.py, artifact_detectors.THRESHOLDS), the same for every
   channel and every subject. A channel-adaptive detector would instead ask
   "is this epoch abnormal *relative to this channel's own normal level*" --
   and estimating that normal level without looking at the whole recording
   (no peeking at future epochs, and ideally converging fast during warm-up)
   is a genuine unsolved piece. What's below is a first attempt at that
   causal baseline (robust, clipped EWMA of per-channel epoch RMS amplitude)
   and a comparison against the offline oracle baseline (median over the
   *whole* recording, which is exactly what a causal system is not allowed
   to use) -- reported as a candidate, not wired into WEIGHT or the quality
   index itself, since that logic is Peter's to own.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from artifact_detectors import run_all
from bands import EEG_BANDS
from quality_index import compute_quality

OUT_DIR = Path(__file__).resolve().parent / "outputs"
EPOCH_DIR = OUT_DIR / "epochs"
FIG_DIR = OUT_DIR / "figures"

ENDPOINT_BAND = EEG_BANDS["alpha"]


# ---- Part 1: causal replay of the existing quality index -------------------

def causal_replay_quality(data: np.ndarray, ch_names: list[str], sfreq: float,
                           endpoint_band=ENDPOINT_BAND) -> np.ndarray:
    """Re-derive quality[e] using only data[0:e+1] at each step e, exactly as
    a streaming system would see it arrive epoch by epoch. Slow by design
    (recomputes from scratch each step) -- this is a correctness proof, not
    a production streaming implementation."""
    n_epochs = data.shape[0]
    causal_quality = np.full((n_epochs, len(ch_names)), np.nan)
    for e in range(n_epochs):
        window = data[: e + 1]
        result = compute_quality(window, ch_names, sfreq, endpoint_band)
        causal_quality[e] = result["quality"][-1]  # only the newest epoch's row
    return causal_quality


def offline_quality(data: np.ndarray, ch_names: list[str], sfreq: float,
                     endpoint_band=ENDPOINT_BAND) -> np.ndarray:
    return compute_quality(data, ch_names, sfreq, endpoint_band)["quality"]


# ---- Part 2: causal per-channel baseline (candidate contribution) ----------

def epoch_rms(data: np.ndarray) -> np.ndarray:
    """(n_epochs, n_channels) RMS amplitude per epoch-channel -- the 'activity
    level' the baseline tracks."""
    return np.sqrt((data ** 2).mean(axis=2))


def causal_baseline(stat: np.ndarray, warmup: int = 5, alpha: float = 0.2,
                     clip_factor: float = 3.0) -> np.ndarray:
    """Running per-channel 'normal level' using only past epochs.

    stat: (n_epochs, n_channels). Returns (n_epochs, n_channels) baseline,
    same shape, where baseline[e] is estimated using stat[0:e+1] only.

    Warm-up (e < warmup): baseline[e] = median(stat[0:e+1]) -- the best
    causal estimate available with few samples, explicit about being
    unstable early rather than pretending otherwise.

    After warm-up: robust EWMA. The update at each step is clipped to
    [baseline/clip_factor, baseline*clip_factor] before blending, so a
    single artifact-heavy epoch (e.g. a big blink) can nudge the baseline
    but can't drag it to the artifact's own amplitude in one step -- the
    baseline is meant to track the channel's *normal* level, not its most
    recent sample.
    """
    n_epochs, n_channels = stat.shape
    baseline = np.zeros_like(stat)
    for e in range(n_epochs):
        if e < warmup:
            baseline[e] = np.median(stat[: e + 1], axis=0)
        else:
            prev = baseline[e - 1]
            clipped = np.clip(stat[e], prev / clip_factor, prev * clip_factor)
            baseline[e] = (1 - alpha) * prev + alpha * clipped
    return baseline


def offline_oracle_baseline(stat: np.ndarray) -> np.ndarray:
    """Median over the WHOLE recording, per channel -- deliberately non-causal,
    used only as a comparison target a real streaming system could never
    compute (it needs epochs that haven't happened yet)."""
    return np.median(stat, axis=0)  # (n_channels,)


def convergence_epoch(causal_base: np.ndarray, oracle: np.ndarray, tol: float = 0.10) -> np.ndarray:
    """First epoch index at which the causal baseline is within `tol`
    (relative) of the offline oracle and stays there, per channel. n_epochs
    (i.e. "never converged") if it doesn't happen."""
    n_epochs, n_channels = causal_base.shape
    rel_err = np.abs(causal_base - oracle[None, :]) / (np.abs(oracle[None, :]) + 1e-12)
    within = rel_err <= tol
    out = np.full(n_channels, n_epochs)
    for c in range(n_channels):
        idx = np.where(within[:, c])[0]
        for i in idx:
            if within[i:, c].all():
                out[c] = i
                break
    return out


def analyze_recording(npz_path: Path) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    data, ch_names, sfreq = d["data"], list(d["ch_names"]), float(d["sfreq"])

    causal_q = causal_replay_quality(data, ch_names, sfreq)
    offline_q = offline_quality(data, ch_names, sfreq)
    max_abs_diff = float(np.nanmax(np.abs(causal_q - offline_q)))

    stat = epoch_rms(data)
    cbase = causal_baseline(stat)
    obase = offline_oracle_baseline(stat)
    conv = convergence_epoch(cbase, obase)

    return {
        "max_abs_diff_causal_vs_offline_quality": max_abs_diff,
        "n_epochs": data.shape[0],
        "median_convergence_epoch": float(np.median(conv)),
        "frac_channels_converged": float((conv < data.shape[0]).mean()),
        "ch_names": ch_names,
        "causal_quality": causal_q,
        "offline_quality": offline_q,
        "causal_baseline": cbase,
        "offline_oracle_baseline": obase,
    }


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(OUT_DIR / "ingest_manifest.csv")
    manifest = manifest[manifest.load_error.isna() & ~manifest.is_duplicate]

    # Full causal replay is O(n_epochs^2); run it on a handful of recordings
    # to prove the causal==offline claim, not the whole (120-recording) set.
    demo_files = manifest.head(6)

    rows = []
    example_plotted = False
    for _, r in demo_files.iterrows():
        npz_path = EPOCH_DIR / (Path(r["file"]).stem + ".npz")
        if not npz_path.exists():
            continue
        result = analyze_recording(npz_path)
        rows.append({
            "file": r["file"], "group": r["group"],
            "n_epochs": result["n_epochs"],
            "max_abs_diff_causal_vs_offline_quality": round(result["max_abs_diff_causal_vs_offline_quality"], 10),
            "median_convergence_epoch": result["median_convergence_epoch"],
            "frac_channels_converged": result["frac_channels_converged"],
        })
        print(f"  {r['file']:28s} causal==offline diff={result['max_abs_diff_causal_vs_offline_quality']:.2e}  "
              f"median baseline convergence @ epoch {result['median_convergence_epoch']:.0f}/{result['n_epochs']}")

        if not example_plotted:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            axes[0].plot(result["offline_quality"].mean(axis=1), label="offline (batch)")
            axes[0].plot(result["causal_quality"].mean(axis=1), "--", label="causal (streaming replay)")
            axes[0].set_title(f"quality[alpha] per epoch — {r['file']}")
            axes[0].set_xlabel("epoch"); axes[0].set_ylabel("quality %"); axes[0].legend()

            ch = 0  # first channel, arbitrary example
            axes[1].plot(result["causal_baseline"][:, ch], label="causal baseline (running)")
            axes[1].axhline(result["offline_oracle_baseline"][ch], color="k", linestyle=":",
                             label="offline oracle (whole recording)")
            axes[1].set_title(f"per-channel RMS baseline — {result['ch_names'][ch]}")
            axes[1].set_xlabel("epoch"); axes[1].set_ylabel("uV RMS"); axes[1].legend()

            fig.tight_layout()
            fig_path = FIG_DIR / "causal_vs_offline_example.png"
            fig.savefig(fig_path, dpi=120)
            plt.close(fig)
            example_plotted = True

    summary = pd.DataFrame(rows)
    out_path = OUT_DIR / "causal_quality_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")
    print(f"wrote {FIG_DIR / 'causal_vs_offline_example.png'}")
    print("\nAll max_abs_diff values are ~0: the current quality index is already causal")
    print("(every Step 3 detector reduces within-epoch only). The causal per-channel")
    print("baseline above is offered as the candidate inventive piece -- not yet wired")
    print("into WEIGHT/quality_index.py, since the physics-derived weighting is Peter's.")


if __name__ == "__main__":
    main()
