"""Build neuroqa/notebooks/phase1_findings.ipynb from scratch against the
current (upload-driven) architecture and real data, replacing the old
notebook, which called modules/functions (ingest.py, preprocess.main(),
study_b.load_merged(), etc.) that no longer exist post-pivot -- see
ARCHITECTURE.md. Every code cell here calls the same manifest.py/pipeline.py/
study_a.py/study_b.py functions run_local.py and webapp.py's API routes call,
so there is exactly one implementation of "how a recording gets scored" (the
codebase's own stated design principle), not a second copy embedded in the
notebook.

Two datasets, matching the brief's actual ask ("Study result is the
deliverable") plus the secondary-check scope agreed with the user:
  - ds003478 (OpenNeuro, CC0): primary run -- real per-subject BDI severity,
    enables Test B.1.
  - Mumtaz/HUSM (figshare, CC BY 4.0): secondary replication check -- no
    severity in this public deposit, so Test B.1 is skipped for it (by
    design, see study_b.py), Tests A/B.2/B.3 still run.

Run: python scripts/build_notebook.py   (writes the .ipynb, unexecuted)
Then execute with nbclient (see scripts/execute_notebook.py).
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

md(r"""# NeuroQA Phase 1 — an endpoint-aware contamination score, tested against a real depression marker

**One-sentence version:** don't build a classifier, don't build a cleaner — detect artifacts with
MNE, score each one by how much its frequency band overlaps the band we're about to measure, and
use that score to test whether a known depression marker (frontal alpha asymmetry, FAA) is partly
contamination.

This notebook runs the whole Phase 1 pipeline end to end against **two real public datasets**,
calling the exact same `manifest.py` / `pipeline.py` / `study_a.py` / `study_b.py` functions
`run_local.py` (offline CLI) and `webapp.py` (the deployed app's `/api/*` routes) call — one
implementation of "how a recording gets scored," not a copy embedded here. Seeded throughout
(`pipeline.SEED = 0` drives ICA/AutoReject/CV); re-running this notebook against the same data
reproduces the same numbers.

**Primary dataset — [ds003478](https://openneuro.org/datasets/ds003478)** ("EEG: Depression rest",
Cavanagh lab, CC0): 119 resting-EEG recordings matching the dataset's own published high-BDI
(>13) / control (<7) groups, real per-subject BDI severity from `participants.tsv` — this is what
makes Test B.1 (`quality ~ group + severity`) possible at all.

**Secondary replication check — Mumtaz/HUSM** (figshare 4244171, CC BY 4.0): 58 eyes-closed
recordings, H/MDD labels only, **no severity score in this public deposit** — Test B.1 is
skipped for it by design (see `study_b.py`'s `regression_1_skipped_reason`), Test A and Test
B.2/B.3 still run as an independent check on the same questions.

**Read this before the numbers below:** `WEIGHT[artifact.type]` and the artifact/endpoint overlap
logic are domain calls — supplied as placeholders here (`bands.py`, every weight = 1.0), pending
the real physics-derived values from Peter. Nothing in this notebook or the rest of the repo tunes
them against an accuracy number. See Part 8 (Caveats) for what that means for interpreting the
results below.""")

code(r"""import sys
from pathlib import Path

NEUROQA_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(NEUROQA_DIR))
REPO_ROOT = NEUROQA_DIR.parent

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

DATASETS = {
    "ds003478": {"label": "ds003478 (primary, real BDI severity)",
                 "data_dir": REPO_ROOT / "data" / "ds003478",
                 "out_dir": REPO_ROOT / "outputs" / "ds003478"},
    "mumtaz": {"label": "Mumtaz/HUSM (secondary check, no severity)",
               "data_dir": REPO_ROOT / "data" / "mumtaz",
               "out_dir": REPO_ROOT / "outputs" / "mumtaz"},
}
for d in DATASETS.values():
    print(d["label"], "->", d["out_dir"])""")

md(r"""## 1. Data loader

Both datasets go through `manifest.validate_batch` — the same per-file validation (labeled,
F3/F4 present, raw not pre-cleaned, long/fast enough to epoch) and batch-level group check
(`MIN_PER_GROUP` per label) that a real upload through `/study` goes through. `validation.json`
below is `run_local.py`'s persisted output of exactly that call.""")

code(r"""def load_validation(name):
    return json.loads((DATASETS[name]["out_dir"] / "validation.json").read_text())

for name in DATASETS:
    v = load_validation(name)
    n_acc, n_rej = len(v["accepted"]), len(v["rejected"])
    groups = pd.Series([a["diagnosis"] for a in v["accepted"]]).value_counts()
    n_sev = sum(1 for a in v["accepted"] if a.get("severity") is not None)
    print(f"--- {DATASETS[name]['label']} ---")
    print(f"  accepted: {n_acc}   rejected: {n_rej}   group_ok: {v['group_ok']}")
    print(f"  groups: {groups.to_dict()}")
    print(f"  with clinical severity: {n_sev}/{n_acc}")
    if v["rejected"]:
        print(f"  rejection reasons (first 3): {[r['reasons'] for r in v['rejected'][:3]]}")
    print()""")

md(r"""## 2. The endpoint-aware quality index

The core idea: the same detected artifact gets a different penalty depending on which frequency
band we're about to measure (`ENDPOINT_BAND`). A blink is a huge penalty against delta/theta and
near-zero against alpha, because a blink's energy doesn't live in alpha.

```
for artifact in detected_artifacts(recording):
    overlap = spectral_overlap(artifact.band, ENDPOINT_BAND)
    penalty += artifact.severity * WEIGHT[artifact.type] * overlap
quality = f(penalty)
```

**Caveat worth flagging to Peter, not silently fixed here:** `ARTIFACT_BANDS["emg"] = (20, 45)` Hz
has **zero** `spectral_overlap` with alpha (8-13 Hz) under the current placeholder bands — detected
muscle artifact currently contributes *nothing* to the alpha-endpoint quality score below. That's
in tension with the brief's own stated hypothesis ("muscle noise sitting at the same frequency as
alpha"). Artifact-band assignment is a domain call, not this notebook's to widen — but it means
the quality-vs-alpha numbers in Part 5 below likely *underestimate* muscle contamination if real
EMG energy does extend into alpha. Confirm with Peter before treating Test B's alpha-quality
results as a clean test of the EMG-alpha overlap claim.""")

code(r"""from bands import ARTIFACT_BANDS, EEG_BANDS, WEIGHT, spectral_overlap

alpha = EEG_BANDS["alpha"]
print(f"WEIGHT (placeholder, all equal until Peter's physics-derived values land): {WEIGHT}")
print()
for name, band in ARTIFACT_BANDS.items():
    print(f"  spectral_overlap({name:11s} {band}, alpha {alpha}) = {spectral_overlap(band, alpha):.2f}")""")

md(r"""## 3. Quality + FAA, per recording

`quality_alpha_pct` is the endpoint-aware score above with `ENDPOINT_BAND = alpha`; `faa` is
`ln(alpha power at F4) - ln(alpha power at F3)` under the "ours" pipeline (every epoch kept,
weighted by its own alpha-band quality — see `faa.py`/`quality_index.py`), the same computation
`score_and_faa()` returns for every accepted upload.""")

code(r"""summaries = {}
for name in DATASETS:
    df = pd.read_csv(DATASETS[name]["out_dir"] / "quality_faa_summary.csv")
    summaries[name] = df
    print(f"--- {DATASETS[name]['label']}: {len(df)} recordings ---")
    display(df.groupby("group")[["quality_alpha_pct", "faa"]].agg(["mean", "std", "count"]))
    print()""")

code(r"""fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, name in zip(axes, DATASETS):
    df = summaries[name]
    for group, color in [("healthy", "tab:blue"), ("depressed", "tab:orange")]:
        vals = df.loc[df.group == group, "faa"]
        ax.hist(vals, bins=15, alpha=0.6, label=group, color=color)
    ax.set_title(DATASETS[name]["label"], fontsize=9)
    ax.set_xlabel("FAA = ln(alpha@F4) - ln(alpha@F3)")
    ax.legend()
fig.suptitle("FAA distribution by group")
fig.tight_layout()
FIG_DIR = REPO_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_DIR / "faa_by_group.png", dpi=120)
plt.show()""")

md(r"""## 4. Test A — does the cleaning choice decide the FAA-group answer?

FAA for the same recording under 5 preprocessing pipelines (raw / ICA / generic reject /
AutoReject / "ours" — the endpoint-aware quality-weighted average) × 2 reference schemes
(original-as-shipped / average), for a bounded subsample per dataset (`STUDY_A_N_PER_GROUP = 6`
per group — ICA+AutoReject are the slow steps; this question is about spread, not statistical
power, see `pipeline.py`). **Red flag:** a big swing across pipelines means the FAA-group
conclusion is substantially a preprocessing artifact, not a robust finding.""")

code(r"""study_a_wide = {}
for name in DATASETS:
    results = json.loads((DATASETS[name]["out_dir"] / "results.json").read_text())
    wide = pd.DataFrame(results["study_a"]["wide"])
    study_a_wide[name] = wide
    print(f"--- {DATASETS[name]['label']}: {len(wide)} recordings x "
          f"{len(results['study_a']['combo_cols'])} pipeline x reference combos ---")
    print(f"  FAA range per subject: median {wide.faa_range.median():.3f}, "
          f"max {wide.faa_range.max():.3f}, mean {wide.faa_range.mean():.3f}")
display(study_a_wide["ds003478"][["file", "group", "faa_range", "faa_std"]])""")

code(r"""fig, ax = plt.subplots(figsize=(7, 4))
data_by_name = [study_a_wide[n].faa_range.dropna().values for n in DATASETS]
ax.boxplot(data_by_name, tick_labels=[DATASETS[n]["label"] for n in DATASETS], vert=True)
ax.set_ylabel("FAA range across 10 pipeline x reference combos, per subject")
ax.set_title("Test A: pipeline-sensitivity spread")
plt.xticks(rotation=10, ha="right", fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "study_a_faa_spread.png", dpi=120)
plt.show()""")

md(r"""## 5. Test B — is contamination itself tied to diagnosis?

Three analyses (`study_b.run_study_b`):

1. **B.1** `quality ~ group + severity` — clinical severity (BDI here), not an EEG-derived
   quantity (using the latter would make this circular — quality is itself computed from
   artifact severity, see `study_b.py`'s module docstring). Skipped, not faked, for a batch
   with no severity data.
2. **B (confound check)** `FAA ~ group + quality` — does data quality confound the very
   FAA-group relationship FAA gets computed for?
3. **B.2** quality-alone → group classifier (5-fold stratified CV, seeded). Red flag: AUC
   meaningfully above chance means contamination alone leaks diagnosis.""")

code(r"""study_b = {}
for name in DATASETS:
    results = json.loads((DATASETS[name]["out_dir"] / "results.json").read_text())
    b = results["study_b"]
    study_b[name] = b
    print(f"=== {DATASETS[name]['label']} (n={b['n']}, n_with_severity={b['n_with_severity']}) ===")
    r1 = b["regression_1_quality_on_group_severity"]
    if r1 is None:
        print(f"  B.1: SKIPPED -- {b['regression_1_skipped_reason']}")
    else:
        print(f"  B.1 quality~group+severity: group p={r1['pvalues']['group_mdd']:.4f}  "
              f"severity p={r1['pvalues']['clinical_severity']:.4f}  R2={r1['rsquared']:.3f}  n={r1['nobs']}")
    r2 = b["regression_2_faa_on_group_quality"]
    print(f"  FAA~group+quality: group p={r2['pvalues']['group_mdd']:.4f}  "
          f"quality p={r2['pvalues']['quality_alpha_pct']:.4f}  R2={r2['rsquared']:.3f}  n={r2['nobs']}")
    r3 = b["regression_3_quality_classifies_group"]
    print(f"  B.2 quality-only classifier: AUC={r3['auc']:.3f}  "
          f"(baseline {r3['majority_class_baseline']:.3f})  leakage_flag={r3['leakage_flag']}")
    print(f"  FINDING: {b['finding']}")
    print()""")

md(r"""## 6. Results table""")

code(r"""rows = []
for name in DATASETS:
    b = study_b[name]
    wide = study_a_wide[name]
    r1 = b["regression_1_quality_on_group_severity"]
    r2 = b["regression_2_faa_on_group_quality"]
    r3 = b["regression_3_quality_classifies_group"]
    rows.append({"dataset": name, "test": "A -- pipeline sensitivity",
                  "metric": "median FAA range across 10 combos", "value": round(float(wide.faa_range.median()), 3),
                  "red_flag_if": "large swing"})
    rows.append({"dataset": name, "test": "B.1 -- quality~group+severity",
                  "metric": "group coefficient p-value",
                  "value": (round(float(r1["pvalues"]["group_mdd"]), 4) if r1 else None),
                  "red_flag_if": "p<0.05 (group predicts contamination)" if r1 else b["regression_1_skipped_reason"]})
    rows.append({"dataset": name, "test": "B confound -- FAA~group+quality",
                  "metric": "quality coefficient p-value", "value": round(float(r2["pvalues"]["quality_alpha_pct"]), 4),
                  "red_flag_if": "p<0.05 (quality confounds FAA-group)"})
    rows.append({"dataset": name, "test": "B.2 -- quality-alone classifier",
                  "metric": "AUC (5-fold CV)", "value": round(float(r3["auc"]), 3),
                  "red_flag_if": "> ~0.65 (noise alone predicts diagnosis)"})
results_table = pd.DataFrame(rows)
results_table.to_csv(REPO_ROOT / "outputs" / "phase1_results_table.csv", index=False)
display(results_table)""")

md(r"""## 7. Caveats — read before citing any of the above

- **`WEIGHT` is an equal-weighting placeholder**, not Peter's derived physics-based values. Every
  number above is contingent on this; swap `bands.WEIGHT` for the real values and re-run once they
  exist — a one-line change, not a redesign.
- **EMG/alpha overlap is currently zero** (Part 2) — if real muscle artifact does extend into
  alpha, as the brief's own hypothesis claims, the alpha-quality numbers above likely
  *underestimate* contamination there. Needs Peter's sign-off on `ARTIFACT_BANDS["emg"]`.
- **Test A ran on a bounded subsample** (12 of each dataset's recordings) for runtime reasons
  (ICA + AutoReject are slow per recording) — suggestive, not a large-N result. Raise
  `pipeline.STUDY_A_N_PER_GROUP` to widen it.
- **Mumtaz/HUSM has no per-subject severity** in its public deposit — Test B.1 only ran on
  ds003478. Mumtaz's contribution here is Test A and B.2/B confound-check as an independent
  replication check, not a full run of all three Study B analyses.
- **ds003478's own README notes some channels were already interpolated** in a subset of files
  before this public release ("There are no raw data to revert to instead") — `manifest.py`'s
  "still looks raw" check is a heuristic and can't detect this from the file header alone; flagged
  here as a caveat on ds003478's "raw, not pre-cleaned" status, not something this pipeline can
  verify or fix.
- **Statistical power**: FAA effects are known to be small in the literature; a null result on
  B.1/B confound/B.2 is informative but not proof of absence, and is reported here rather than
  hidden either way, per the brief's own instruction.
- **Causal/streaming version** (feeding the quality index a recording sample-by-sample, no
  look-ahead) is explicitly a later, optional stretch per the brief — not required for this
  notebook's finding, and not implemented here.""")

nb["cells"] = cells
nbf.write(nb, str(Path(__file__).resolve().parent.parent / "neuroqa" / "notebooks" / "phase1_findings.ipynb"))
print("wrote neuroqa/notebooks/phase1_findings.ipynb")
