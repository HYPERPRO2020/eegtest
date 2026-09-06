"""Deliverable #3 (spec Sec. 7): per-subject example spectra with the
specparam fit overlaid (a few clean peaks + a few no-peak cases), and the
classic-vs-periodic FAA scatter, per dataset x condition.

Usage: python scripts/plot_spectral_composition.py
Writes outputs/<dataset>/figures/spectral_composition_examples_<condition>.png
and outputs/<dataset>/figures/spectral_composition_classic_vs_periodic_<condition>.png
"""
import csv
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "neuroqa"))

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from fooof import FOOOF

from spectral_composition import ALPHA_BAND, FIT_FREQ_RANGE, FOOOF_PARAMS, quality_weighted_psd  # noqa: E402
from study_a import load_referenced_raw, make_epochs  # noqa: E402

DATASETS = [
    {"key": "ds003478", "data_dir": REPO_ROOT / "data" / "ds003478", "line_freq": 60.0, "condition": "ec"},
    {"key": "ds007615", "data_dir": REPO_ROOT / "data" / "ds007615", "line_freq": 50.0, "condition": "ec"},
    {"key": "ds007615", "data_dir": REPO_ROOT / "data" / "ds007615_eo", "line_freq": 50.0, "condition": "eo"},
    {"key": "mumtaz", "data_dir": REPO_ROOT / "data" / "mumtaz", "line_freq": 50.0, "condition": "ec"},
    {"key": "mumtaz", "data_dir": REPO_ROOT / "data" / "mumtaz_eo", "line_freq": 50.0, "condition": "eo"},
]


def read_csv_rows(key: str, condition: str) -> list[dict]:
    path = REPO_ROOT / "outputs" / key / f"spectral_composition_{condition}.csv"
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def plot_example_spectra(ds: dict, out_path: Path, n_examples: int = 3):
    """A few clean-peak examples + a few no-peak examples, original
    reference, aperiodic_mode='fixed' -- the primary/default setting."""
    rows = [r for r in read_csv_rows(ds["key"], ds["condition"])
            if r["reference"] == "original" and r["aperiodic_mode"] == "fixed"]
    if not rows:
        print(f"  [skip plot] no rows for {ds['key']} {ds['condition']}")
        return
    clean = [r for r in rows if r["f3_peak_present"] == "True" and r["f3_low_quality_fit"] == "False"][:n_examples]
    no_peak = [r for r in rows if r["f3_peak_present"] == "False" and r["f3_low_quality_fit"] == "False"][:n_examples]
    examples = clean + no_peak
    if not examples:
        print(f"  [skip plot] no clean examples for {ds['key']} {ds['condition']}")
        return

    fig, axes = plt.subplots(1, len(examples), figsize=(5 * len(examples), 4))
    if len(examples) == 1:
        axes = [axes]
    for ax, row in zip(axes, examples):
        path = ds["data_dir"] / row["file"]
        raw, ch_names = load_referenced_raw(str(path), "original", line_freq=ds["line_freq"])
        epochs = make_epochs(raw)
        data_uv = epochs.get_data() * 1e6
        freqs, psd = quality_weighted_psd(data_uv, ch_names, raw.info["sfreq"])
        i3 = ch_names.index("F3")
        fm = FOOOF(aperiodic_mode="fixed", verbose=False, **FOOOF_PARAMS)
        fm.fit(freqs, psd[i3], list(FIT_FREQ_RANGE))
        fm.plot(ax=ax, plot_peaks="shade", add_legend=False)
        label = "peak" if row["f3_peak_present"] == "True" else "no peak"
        ax.set_title(f"{row['file'][:14]} (F3, {label}, r2={float(row['f3_r_squared']):.2f})", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_classic_vs_periodic(ds: dict, out_path: Path):
    rows = [r for r in read_csv_rows(ds["key"], ds["condition"])
            if r["reference"] == "original" and r["aperiodic_mode"] == "fixed"]
    if not rows:
        return
    classic = np.array([float(r["classic_faa"]) for r in rows])
    periodic = np.array([float(r["periodic_only_faa"]) for r in rows])
    low_q = np.array([r["f3_low_quality_fit"] == "True" or r["f4_low_quality_fit"] == "True" for r in rows])
    valid = ~np.isnan(periodic) & ~low_q

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(classic[valid], periodic[valid], alpha=0.6)
    if valid.sum() >= 3:
        r = np.corrcoef(classic[valid], periodic[valid])[0, 1]
        ax.set_title(f"{ds['key']} ({ds['condition']}): classic vs periodic-only FAA\n"
                      f"n={int(valid.sum())}, Pearson r={r:.3f}", fontsize=10)
    ax.set_xlabel("classic FAA (total band power)")
    ax.set_ylabel("periodic-only FAA (peak power)")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    for ds in DATASETS:
        print(f"{ds['key']} ({ds['condition']}):")
        fig_dir = REPO_ROOT / "outputs" / ds["key"] / "figures"
        plot_example_spectra(ds, fig_dir / f"spectral_composition_examples_{ds['condition']}.png")
        plot_classic_vs_periodic(ds, fig_dir / f"spectral_composition_classic_vs_periodic_{ds['condition']}.png")


if __name__ == "__main__":
    main()
