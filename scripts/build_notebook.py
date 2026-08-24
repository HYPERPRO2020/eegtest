"""Build neuroqa/notebooks/phase1_findings.ipynb from scratch against the
current (upload-driven) architecture and real data, replacing the old
notebook, which called modules/functions (ingest.py, preprocess.main(),
study_b.load_merged(), etc.) that no longer exist post-pivot -- see
ARCHITECTURE.md. Every code cell here calls the same manifest.py/pipeline.py/
study_a.py/study_b.py functions run_local.py and webapp.py's API routes call,
so there is exactly one implementation of "how a recording gets scored" (the
codebase's own stated design principle), not a second copy embedded in the
notebook.

Three datasets, matching the brief's actual ask ("Study result is the
deliverable") plus the replication scope agreed with the user:
  - ds003478 (OpenNeuro, CC0): real per-subject BDI severity, enables Test
    B.1. Univ. of Arizona cohort.
  - ds007615 (OpenNeuro, CC0): real per-subject BDI-II severity, also
    enables Test B.1 -- an independent cohort (Univ. of Oslo) from a
    different lab, not overlapping with ds003478's subjects.
  - Mumtaz/HUSM (figshare, CC BY 4.0): replication check -- no severity in
    this public deposit, so Test B.1 is skipped for it (by design, see
    study_b.py), Tests A/B.2/confound-check still run.

Run: python scripts/build_notebook.py   (writes the .ipynb, unexecuted)
Then execute with nbclient (see scripts/execute_notebook.py).
"""
from pathlib import Path

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

This notebook runs the whole Phase 1 pipeline end to end against **three real public datasets**,
calling the exact same `manifest.py` / `pipeline.py` / `study_a.py` / `study_b.py` functions
`run_local.py` (offline CLI) and `webapp.py` (the deployed app's `/api/*` routes) call — one
implementation of "how a recording gets scored," not a copy embedded here. Seeded throughout
(`pipeline.SEED = 0` drives ICA/AutoReject/CV); re-running this notebook against the same data
reproduces the same numbers.

**[ds003478](https://openneuro.org/datasets/ds003478)** ("EEG: Depression rest", Cavanagh lab /
Univ. of Arizona, CC0): 119 resting-EEG recordings matching the dataset's own published high-BDI
(>13) / control (<7) groups, real per-subject BDI severity from `participants.tsv`.

**[ds007615](https://openneuro.org/datasets/ds007615)** ("LDAEP and resting-state EEG in healthy
women", Univ. of Oslo, CC0): 49 recordings, same BDI(>13)/BDI(<7) thresholds applied to a real
per-subject BDI-II score (`phenotype/bdi.tsv`) — an independent cohort from a different lab, not
overlapping with ds003478's subjects.

**Mumtaz/HUSM** (figshare 4244171, CC BY 4.0): 58 eyes-closed recordings, H/MDD labels only,
**no severity score in this public deposit** — Test B.1 is skipped for it by design (see
`study_b.py`'s `regression_1_skipped_reason`), Test A and Test B.2/confound-check still run as an
independent check on the same questions.

Both ds003478 and ds007615 carry real per-subject severity, so Test B.1 runs on two independent
cohorts, not one.

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
    "ds003478": {"label": "ds003478 (real BDI severity)",
                 "data_dir": REPO_ROOT / "data" / "ds003478",
                 "out_dir": REPO_ROOT / "outputs" / "ds003478"},
    "ds007615": {"label": "ds007615 (real BDI-II severity)",
                 "data_dir": REPO_ROOT / "data" / "ds007615",
                 "out_dir": REPO_ROOT / "outputs" / "ds007615"},
    "mumtaz": {"label": "Mumtaz/HUSM (no severity)",
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

code(r"""fig, axes = plt.subplots(1, len(DATASETS), figsize=(5.5 * len(DATASETS), 4))
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
all_results = {}
for name in DATASETS:
    results = json.loads((DATASETS[name]["out_dir"] / "results.json").read_text())
    all_results[name] = results
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

md(r"""## 4b. Five independent FAA-only classifiers, one per pipeline

A request from Peter (domain): not just "how much does FAA move across pipelines" (Test A above),
but 4-5 *independent* classifiers — one per preprocessing pipeline, each one only using that
pipeline's own FAA number to guess depressed/healthy, none sharing data or influencing each
other — with a result reported for each (`faa_classifiers.classify_by_pipeline`, wired into
`pipeline.aggregate()` so it's part of every run's `results.json` going forward, not a one-off
script). Runs on the same bounded Study A subsample (12 recordings) as Test A, so treat n=12
results as indicative, not conclusive — same statistical-power caveat as Test A.""")

code(r"""clf_rows = []
for name in DATASETS:
    clf = all_results[name]["faa_classifiers_by_pipeline"]
    print(f"--- {DATASETS[name]['label']} ---")
    for pipeline_name, r in clf.items():
        if "error" in r:
            print(f"  {pipeline_name:12s} SKIPPED -- {r['error']}")
            continue
        print(f"  {pipeline_name:12s} accuracy={r['accuracy']:.2f}  AUC={r['auc']:.2f} "
              f"[{r['auc_ci_lo']:.2f}, {r['auc_ci_hi']:.2f}]  (baseline {r['baseline']:.2f}, n={r['n']})")
        clf_rows.append({"dataset": name, "pipeline": pipeline_name, **r})
    print()
clf_df = pd.DataFrame(clf_rows)
display(clf_df)""")

code(r"""fig, ax = plt.subplots(figsize=(9, 4))
n_ds = len(DATASETS)
width = 0.8 / n_ds
x = np.arange(len(PIPELINES := ["raw", "ica", "generic", "autoreject", "ours"]))
for i, name in enumerate(DATASETS):
    sub = clf_df[clf_df.dataset == name].set_index("pipeline").reindex(PIPELINES)
    ax.bar(x + i * width, sub["auc"], width, label=DATASETS[name]["label"])
ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance")
ax.set_xticks(x + width * (n_ds - 1) / 2)
ax.set_xticklabels(PIPELINES)
ax.set_ylabel("AUC (5-fold CV)")
ax.set_title("Per-pipeline FAA-only classifier: does any single pipeline's FAA predict group?")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "faa_classifiers_by_pipeline.png", dpi=120)
plt.show()""")

md(r"""## 5. Test B — is contamination itself tied to diagnosis?

Three analyses (`study_b.run_study_b`), each run twice per dataset: on `quality_alpha_frontal_pct`
(F3/F4/Fp1/Fp2 only — **primary**, the direct test of the frontal-muscle-contamination hypothesis)
and on `quality_alpha_pct` (whole-scalp mean — **secondary**, reported alongside so the channel
subset isn't picked after seeing which one looks better).

1. **B.1** `quality ~ group + severity` — clinical severity (BDI here), not an EEG-derived
   quantity (using the latter would make this circular — quality is itself computed from
   artifact severity, see `study_b.py`'s module docstring). Skipped, not faked, for a batch
   with no severity data.
2. **B (confound check)** `FAA ~ group + quality` — does data quality confound the very
   FAA-group relationship FAA gets computed for?
3. **B.2** quality-alone → group classifier (5-fold stratified CV, seeded), with a bootstrap
   95% CI on the AUC. Red flag: AUC meaningfully above chance means contamination alone leaks
   diagnosis.

All p-values feeding the `finding` text are **Holm-Bonferroni corrected** across each variant's
own family (group coefficient, quality coefficient, severity coefficient) before the
significance check — a raw, uncorrected p<0.05 across three-ish tests understates how easy that
threshold is to hit by chance alone.""")

code(r"""study_b = {}
for name in DATASETS:
    results = json.loads((DATASETS[name]["out_dir"] / "results.json").read_text())
    b = results["study_b"]
    study_b[name] = b
    print(f"=== {DATASETS[name]['label']} ===")
    for variant_key, variant_label in [("frontal", "FRONTAL (primary)"), ("whole_scalp", "whole-scalp (secondary)")]:
        v = b[variant_key]
        print(f"  --- {variant_label}: n={v['n']}, n_with_severity={v['n_with_severity']} ---")
        qcol = results["study_b"]["quality_variant_primary"] if variant_key == "frontal" else results["study_b"]["quality_variant_secondary"]
        r1 = v["regression_1_quality_on_group_severity"]
        if r1 is None:
            print(f"    B.1: SKIPPED -- {v['regression_1_skipped_reason']}")
        else:
            print(f"    B.1 quality~group+severity: group p={r1['pvalues']['group_mdd']:.4f}  "
                  f"severity p={r1['pvalues']['clinical_severity']:.4f}  R2={r1['rsquared']:.3f}  n={r1['nobs']}")
        r2 = v["regression_2_faa_on_group_quality"]
        print(f"    FAA~group+quality: group p={r2['pvalues']['group_mdd']:.4f}  "
              f"quality p={r2['pvalues'][qcol]:.4f}  R2={r2['rsquared']:.3f}  n={r2['nobs']}")
        print(f"    Holm-corrected: {v['holm_corrected_pvalues']}")
        r3 = v["regression_3_quality_classifies_group"]
        print(f"    B.2 quality-only classifier: AUC={r3['auc']:.3f} "
              f"(95% CI [{r3['auc_ci_lo']:.2f}, {r3['auc_ci_hi']:.2f}], baseline {r3['majority_class_baseline']:.3f})  "
              f"leakage_flag={r3['leakage_flag']}")
    print(f"  FINDING (frontal-based): {b['finding']}")
    print()""")

md(r"""## 6. Results table""")

code(r"""rows = []
for name in DATASETS:
    b = study_b[name]
    frontal, whole_scalp = b["frontal"], b["whole_scalp"]
    wide = study_a_wide[name]
    rows.append({"dataset": name, "test": "A -- pipeline sensitivity",
                  "metric": f"median FAA range, {len(wide)} recordings", "value": round(float(wide.faa_range.median()), 3),
                  "red_flag_if": "large swing"})
    for variant_name, v in [("frontal", frontal), ("whole-scalp", whole_scalp)]:
        r1 = v["regression_1_quality_on_group_severity"]
        r2 = v["regression_2_faa_on_group_quality"]
        r3 = v["regression_3_quality_classifies_group"]
        rows.append({"dataset": name, "test": f"B.1 ({variant_name}) -- quality~group+severity",
                      "metric": "group coefficient p-value (Holm-corrected)",
                      "value": (round(v["holm_corrected_pvalues"].get("group_mdd", float("nan")), 4) if r1 else None),
                      "red_flag_if": "p<0.05 (group predicts contamination)" if r1 else v["regression_1_skipped_reason"]})
        rows.append({"dataset": name, "test": f"B confound ({variant_name}) -- FAA~group+quality",
                      "metric": "group coefficient p-value (Holm-corrected)",
                      "value": round(v["holm_corrected_pvalues"]["group_mdd"], 4),
                      "red_flag_if": "p>=0.05 here alongside a real B.2 leak would undercut the group-FAA link"})
        rows.append({"dataset": name, "test": f"B.2 ({variant_name}) -- quality-alone classifier",
                      "metric": "AUC (5-fold CV, 95% bootstrap CI)",
                      "value": f"{r3['auc']:.3f} [{r3['auc_ci_lo']:.2f}, {r3['auc_ci_hi']:.2f}]",
                      "red_flag_if": "> ~0.65 (noise alone predicts diagnosis)"})
results_table = pd.DataFrame(rows)
results_table.to_csv(REPO_ROOT / "outputs" / "phase1_results_table.csv", index=False)
display(results_table)""")

md(r"""## 7. v1 -> v2: what changed and why

Per the "Neureidos Phase 1 Methodology Upgrade" implementation guide (2026-08-22), applied and
re-run 2026-08-24. v1 (placeholder-weight, pre-fix) results are archived at
`outputs_v1_placeholder_weights/`, not overwritten in place:

- **`ARTIFACT_BANDS["emg"]` widened (20,45) → (8,45) Hz** per Goncharova et al. (2003) --
  Peter-approved. Was zero overlap with alpha, so detected EMG contributed nothing to the
  alpha-endpoint quality score no matter how severe.
- **Per-dataset mains frequency**: ds003478 notch-filtered at 60 Hz (confirmed via its own
  `eeg.json`: `PowerLineFrequency: 60`, Univ. of Arizona), not the 50 Hz every dataset got before
  this fix -- `detect_line_noise()` was checking the wrong frequency entirely for ds003478.
- **`quality_alpha_frontal_pct`** (F3/F4/Fp1/Fp2 only) is now the primary Study B variable --
  the actual hypothesis is about frontal contamination, not a whole-scalp average diluted by
  15+ unrelated channels.
- **Bootstrap 95% CIs** on every AUC (Study B.2 and the 5 per-pipeline classifiers) and
  **Holm-Bonferroni correction** on the p-values feeding `finding`.
- **Study A scaled from 12 to up to 30 recordings** (15/group where available) and
  **parallelized** (joblib) -- was the real bottleneck on going past 12.

**Not done**, on purpose: widening `detect_emg`'s own 20-45 Hz severity-measurement window
(separate Peter conversation), setting real `WEIGHT` values or the quality-decay function (no
actual numbers supplied yet), an EMG spectral-signature specificity check, resolving whether
Mumtaz/HUSM's original paper reports severity not in the public deposit (checked; not found in
searchable sources), and a Stewart et al. (2011) 70-90 Hz proxy-band comparison arm. All flagged,
none silently dropped.""")

md(r"""## 8. Caveats — read before citing any of the above

- **`WEIGHT` is still an equal-weighting placeholder**, not Peter's derived physics-based values
  -- only `ARTIFACT_BANDS["emg"]` (the band definition) has real domain sign-off so far, not the
  per-type weights or the quality-decay function. Every number above is contingent on this.
- **Test A ran on a bounded subsample** (up to 30 of each dataset's recordings, 15/group where
  available) for runtime reasons (ICA + AutoReject are slow per recording, even parallelized) --
  larger than v1's 12, still suggestive rather than a full-N result.
- **Mumtaz/HUSM has no per-subject severity** in its public deposit -- Test B.1 ran on ds003478
  and ds007615 (two independent cohorts), not Mumtaz. Mumtaz's contribution here is Test A and
  B.2/B confound-check as a replication check, not a full run of all three Study B analyses.
- **ds003478 and ds007615 use the same BDI(>13)/BDI(<7) group thresholds** for consistency --
  chosen to match ds003478's own published groups, not independently re-derived per dataset.
- **ds003478's own README notes some channels were already interpolated** in a subset of files
  before this public release ("There are no raw data to revert to instead") -- `manifest.py`'s
  "still looks raw" check is a heuristic and can't detect this from the file header alone; flagged
  here as a caveat on ds003478's "raw, not pre-cleaned" status, not something this pipeline can
  verify or fix.
- **Statistical power**: FAA effects are known to be small in the literature. ds007615's
  Holm-corrected group effect (Part 5) is the one significant result across three datasets and
  two quality variants each -- real and independently corroborated by the whole-scalp variant,
  but not yet replicated by ds003478 or Mumtaz, and a single significant result among several
  tests is not the same thing as a confirmed effect.
- **The 5 per-pipeline classifiers (Part 4b)** stayed at/below chance in every dataset, including
  ds007615 despite its significant group-level regression result -- a single-feature classifier
  needs more than a mean group difference to beat chance at this sample size; the CI on each AUC
  makes that uncertainty explicit rather than implying false precision.
- **Causal/streaming version** (feeding the quality index a recording sample-by-sample, no
  look-ahead) is explicitly a later, optional stretch per the brief -- not required for this
  notebook's finding, and not implemented here.""")

nb["cells"] = cells
nbf.write(nb, str(Path(__file__).resolve().parent.parent / "neuroqa" / "notebooks" / "phase1_findings.ipynb"))
print("wrote neuroqa/notebooks/phase1_findings.ipynb")
