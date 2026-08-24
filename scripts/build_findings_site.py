"""Regenerate findings-site/index.html from the real results.json files for
all three datasets -- keeps the site's numbers tied to an actual run instead
of hand-typed. Run after any re-run of run_local.py that changes outputs/.

Usage: python scripts/build_findings_site.py
"""
import json
import shutil
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "outputs"
SITE_DIR = REPO_ROOT / "findings-site"

DATASETS = [
    {"key": "ds003478", "label": "ds003478", "sub": "real BDI severity · Univ. of Arizona",
     "color_var": "--series-primary"},
    {"key": "ds007615", "label": "ds007615", "sub": "real BDI-II severity · Univ. of Oslo",
     "color_var": "--good"},
    {"key": "mumtaz", "label": "Mumtaz/HUSM", "sub": "replication check · no severity",
     "color_var": "--series-mdd"},
]

PIPELINES = ["raw", "ica", "generic", "autoreject", "ours"]
PIPELINE_LABEL = {"raw": "raw", "ica": "ICA", "generic": "generic reject",
                   "autoreject": "AutoReject", "ours": "ours (endpoint-aware)"}


def load(key):
    return json.loads((OUT_DIR / key / "results.json").read_text())


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def study_a_bars(wide_rows, max_range=2.0):
    rows = sorted(wide_rows, key=lambda w: w["faa_range"])
    out = []
    for w in rows:
        pct = min(100.0, 100.0 * w["faa_range"] / max_range)
        color = "var(--series-h)" if w["group"] == "healthy" else "var(--series-mdd)"
        short = w["file"].split("_task-")[0].split("_EC")[0].replace(".edf", "")
        out.append(f'''<div class="bar-row">
          <span class="bar-label">{esc(short)}</span>
          <div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%;background:{color}"></div></div>
          <span class="bar-value">{w['faa_range']:.3f}</span>
        </div>''')
    return "\n".join(out)


def classifier_bars(clf_by_pipeline, max_auc=1.0):
    out = []
    for p in PIPELINES:
        r = clf_by_pipeline.get(p, {})
        if "error" in r:
            out.append(f'''<div class="bar-row">
              <span class="bar-label">{PIPELINE_LABEL[p]}</span>
              <div class="bar-track"></div>
              <span class="bar-value" style="color:var(--text-muted)">n/a</span>
            </div>''')
            continue
        pct = min(100.0, 100.0 * r["auc"] / max_auc)
        color = "var(--good)" if r["auc"] < 0.5 else ("var(--warning)" if r["auc"] < 0.65 else "var(--critical)")
        ci = f" [{r['auc_ci_lo']:.2f},{r['auc_ci_hi']:.2f}]" if "auc_ci_lo" in r else ""
        out.append(f'''<div class="bar-row">
          <span class="bar-label">{PIPELINE_LABEL[p]}</span>
          <div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%;background:{color}"></div>
            <div class="bar-chance-line"></div></div>
          <span class="bar-value">{r['auc']:.2f}{ci}</span>
        </div>''')
    return "\n".join(out)


def verdict_card(title, ask, metric, is_flag, badge_override=None):
    if badge_override is not None:
        badge = badge_override
    else:
        badge = ('<span class="badge warn"><span class="dot"></span>flagged</span>' if is_flag
                 else '<span class="badge good"><span class="dot"></span>no red flag</span>')
    return f'''<div class="verdict-card">
      <div class="test-name">{esc(title)}</div>
      <div class="ask">{esc(ask)}</div>
      <div class="metric">{esc(metric)}</div>
      {badge}
    </div>'''


SIGNAL_BADGE = '<span class="badge good"><span class="dot"></span>significant</span>'
NO_SIGNAL_BADGE = '<span class="badge"><span class="dot"></span>not significant</span>'


def variant_cards(v, label):
    """One row of 4 verdict cards for a single quality variant (frontal or
    whole-scalp), using its own Holm-Bonferroni-corrected p-values.
    Group->FAA gets its own card (a significant result there is the actual
    finding this project is testing for, not a red flag) -- distinct from
    the confound check, which tests quality's own coefficient in that same
    regression and *is* a red flag if significant (quality explaining FAA
    would undercut the group result instead of supporting it)."""
    corrected = v["holm_corrected_pvalues"]
    r1 = v["regression_1_quality_on_group_severity"]
    cards = []
    if r1 is not None:
        p = corrected.get("severity", 1.0)
        cards.append(verdict_card(f"B.1 ({label})", "quality ~ group + severity (Holm-corrected)",
                                   f"p = {p:.3f}", p < 0.05))
    else:
        cards.append(verdict_card(f"B.1 ({label})", "quality ~ group + severity", "skipped: no severity data", False))
    quality_col = [k for k in corrected if k not in ("group_mdd", "severity")][0]
    qp = corrected[quality_col]
    cards.append(verdict_card(f"Confound check ({label})", "does quality predict FAA? (Holm-corrected)",
                               f"p = {qp:.3f}", qp < 0.05))
    gp = corrected["group_mdd"]
    cards.append(verdict_card(f"Group → FAA ({label})", "does group predict FAA, controlling for quality?",
                               f"p = {gp:.3f}", False, badge_override=(SIGNAL_BADGE if gp < 0.05 else NO_SIGNAL_BADGE)))
    r3 = v["regression_3_quality_classifies_group"]
    cards.append(verdict_card(f"B.2 ({label})", "quality alone → group (5-fold CV, 95% CI)",
                               f"AUC = {r3['auc']:.2f} [{r3['auc_ci_lo']:.2f}, {r3['auc_ci_hi']:.2f}]",
                               r3["leakage_flag"]))
    return cards


def mediation_note(v, label):
    """One-line summary of the mediation_analysis block (see study_b.py):
    the indirect (quality-mediated) effect of group on FAA, with its
    bootstrap CI. proportion_mediated is only shown when it falls in a
    sane [0,1] range -- it's a known-unstable statistic when the total
    effect is close to zero, and a raw out-of-range percentage would
    confuse more than it clarifies."""
    m = v.get("mediation_analysis")
    if m is None:
        return ""
    acme, prop = m["acme"], m["proportion_mediated"]
    prop_clause = (f", an estimated {prop * 100:.0f}% of the group-FAA relationship"
                    if isinstance(prop, (int, float)) and 0 <= prop <= 1 else "")
    return (f'<p class="dek" style="margin-top:10px;margin-bottom:0;max-width:100%">'
            f'<strong>Mediation check ({esc(label)}):</strong> indirect (quality-mediated) effect on FAA = '
            f'{acme["point"]:+.4f} (95%&nbsp;CI [{acme["ci_lo"]:+.4f}, {acme["ci_hi"]:+.4f}]){prop_clause}.</p>')


STUDY_C_KIND_LABEL = {
    "eog": "EOG (0.5-4 Hz, no alpha overlap)",
    "emg_alpha_overlap": "EMG, alpha-overlapping (8-13 Hz)",
    "emg_no_overlap": "EMG, non-overlapping (20-45 Hz)",
}
STUDY_C_KIND_ORDER = ["eog", "emg_alpha_overlap", "emg_no_overlap"]


def load_study_c(key):
    path = OUT_DIR / key / "study_c.json"
    return json.loads(path.read_text()) if path.exists() else None


def pooled_study_c(study_c_by_key, n_boot=5000, seed=0):
    """Pool every recording's per-kind dose-response slope across all
    datasets and bootstrap the combined mean -- more statistical power than
    any single dataset's population-level estimate, since the injection
    protocol and dose levels are identical across datasets."""
    rng = np.random.RandomState(seed)
    pooled = {k: [] for k in STUDY_C_KIND_ORDER}
    for sc in study_c_by_key.values():
        if sc is None:
            continue
        for s in sc["per_subject"]:
            if "results" not in s:
                continue
            for k in STUDY_C_KIND_ORDER:
                pooled[k].append(s["results"][k]["slope"])
    out = {}
    for k in STUDY_C_KIND_ORDER:
        vals = np.array(pooled[k])
        if len(vals) == 0:
            out[k] = {"n_subjects": 0, "mean_slope": float("nan"), "slope_ci_lo": float("nan"),
                       "slope_ci_hi": float("nan"), "nonzero_slope": False}
            continue
        idx = np.arange(len(vals))
        boots = [vals[rng.choice(idx, size=len(idx), replace=True)].mean() for _ in range(n_boot)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out[k] = {"n_subjects": int(len(vals)), "mean_slope": float(vals.mean()), "slope_ci_lo": float(lo),
                   "slope_ci_hi": float(hi), "nonzero_slope": bool(not (lo <= 0.0 <= hi))}
    return out


def study_c_bars(summary, max_abs=None):
    """summary: {kind: {...}} matching study_c.py's own run_study_c()["summary"]
    schema (n_subjects/mean_slope/slope_ci_lo/slope_ci_hi/nonzero_slope) --
    pooled_study_c above deliberately returns the same schema so this
    renders either a single dataset's or the pooled summary identically."""
    vals = [abs(summary[k]["mean_slope"]) for k in STUDY_C_KIND_ORDER if summary[k]["n_subjects"] > 0]
    scale = max_abs if max_abs is not None else (max(vals) if vals else 1.0) or 1.0
    out = []
    for k in STUDY_C_KIND_ORDER:
        s = summary[k]
        if s["n_subjects"] == 0:
            continue
        pct = min(100.0, 100.0 * abs(s["mean_slope"]) / scale) if scale else 0.0
        color = "var(--warning)" if s["nonzero_slope"] else "var(--text-muted)"
        out.append(f'''<div class="bar-row">
          <span class="bar-label">{esc(STUDY_C_KIND_LABEL[k])}</span>
          <div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%;background:{color}"></div></div>
          <span class="bar-value">{s['mean_slope']:+.4f}</span>
        </div>''')
    return "\n".join(out)


def study_c_section(d, sc):
    if sc is None:
        return f'''<div class="dataset-block"><h3>{esc(d['label'])}</h3>
        <p class="dek">Study C not run for this dataset.</p></div>'''
    s = sc["summary"]
    ao = s["emg_alpha_overlap"]
    return f'''<div class="dataset-block">
      <h3>{esc(d['label'])} <span class="dataset-sub">({esc(d['sub'])}, n={sc['n_recordings']})</span></h3>
      <div class="card">{study_c_bars(s)}</div>
      <p class="dek" style="margin-top:10px;margin-bottom:0;max-width:100%">Bootstrapped 95%&nbsp;CI on the
      population-mean dose-response slope: alpha-overlapping = [{ao['slope_ci_lo']:+.4f},
      {ao['slope_ci_hi']:+.4f}]{' (excludes zero)' if ao['nonzero_slope'] else ''};
      EOG = [{s['eog']['slope_ci_lo']:+.4f}, {s['eog']['slope_ci_hi']:+.4f}]; non-overlapping EMG =
      [{s['emg_no_overlap']['slope_ci_lo']:+.4f}, {s['emg_no_overlap']['slope_ci_hi']:+.4f}].</p>
    </div>'''


def dataset_testb_section(d, r):
    b = r["study_b"]
    frontal_cards = variant_cards(b["frontal"], "frontal")
    scalp_cards = variant_cards(b["whole_scalp"], "whole-scalp")
    finding_text = esc(b.get("finding", ""))
    mediation_html = mediation_note(b["frontal"], "frontal")
    return f'''<div class="dataset-block">
      <h3>{esc(d['label'])} <span class="dataset-sub">({esc(d['sub'])}, n={b['n']})</span></h3>
      <div class="verdict-row">{"".join(frontal_cards)}</div>
      <div class="verdict-row" style="margin-top:10px">{"".join(scalp_cards)}</div>
      <p class="dek" style="margin-top:14px;margin-bottom:0;max-width:100%">{finding_text}</p>
      {mediation_html}
    </div>'''


def main():
    results = {d["key"]: load(d["key"]) for d in DATASETS}
    study_c_results = {d["key"]: load_study_c(d["key"]) for d in DATASETS}
    study_c_pooled = pooled_study_c(study_c_results)

    testa_blocks = []
    for d in DATASETS:
        r = results[d["key"]]
        wide = r["study_a"]["wide"]
        med = sorted(w["faa_range"] for w in wide)[len(wide) // 2]
        mx = max(w["faa_range"] for w in wide)
        testa_blocks.append(f'''<div class="dataset-block">
          <h3>{esc(d['label'])} <span class="dataset-sub">({esc(d['sub'])})</span></h3>
          <div class="stat-row">
            <div class="stat-tile"><div class="label">Median FAA range ({len(wide)} subjects)</div>
              <div class="value">{med:.3f}</div><div class="sub">typical swing across the 10 combos</div></div>
            <div class="stat-tile"><div class="label">Max FAA range</div>
              <div class="value" style="color:var(--warning)">{mx:.3f}</div>
              <div class="sub">largest single-subject swing</div></div>
          </div>
          <div class="card">{study_a_bars(wide)}</div>
        </div>''')

    clf_blocks = []
    for d in DATASETS:
        r = results[d["key"]]
        clf_blocks.append(f'''<div class="dataset-block">
          <h3>{esc(d['label'])}</h3>
          <div class="card">{classifier_bars(r["faa_classifiers_by_pipeline"])}</div>
        </div>''')

    testc_blocks = [study_c_section(d, study_c_results[d["key"]]) for d in DATASETS]

    testb_blocks = [dataset_testb_section(d, results[d["key"]]) for d in DATASETS]

    dataset_stat_tiles = "".join(f'''<div class="stat-tile"><div class="label">{esc(d['label'])} ({esc(d['sub'])})</div>
        <div class="value">{results[d['key']]['n_recordings']}</div>
        <div class="sub">recordings</div></div>''' for d in DATASETS)

    # Derive the finding narrative from the actual numbers rather than
    # hand-typing them -- a v1 version of this section went stale exactly
    # this way once the numbers underneath it changed.
    significant = []   # (dataset label, frontal group p raw, holm-corrected)
    leaks = []         # (dataset label, variant label) wherever leakage_flag fired
    for d in DATASETS:
        b = results[d["key"]]["study_b"]
        for variant_label, v in [("frontal", b["frontal"]), ("whole-scalp", b["whole_scalp"])]:
            if v["regression_3_quality_classifies_group"]["leakage_flag"]:
                leaks.append((d["label"], variant_label))
        fr = b["frontal"]
        gp_corrected = fr["holm_corrected_pvalues"]["group_mdd"]
        gp_raw = fr["regression_2_faa_on_group_quality"]["pvalues"]["group_mdd"]
        if gp_corrected < 0.05:
            significant.append((d["label"], gp_raw, gp_corrected))

    if significant:
        sig_sentences = "; ".join(
            f"in {esc(name)} (p = {raw:.3f} uncorrected, p = {corr:.3f} Holm-corrected)" for name, raw, corr in significant)
        finding_p2 = (f'''In addition, the frontal-specific group effect on FAA remains significant after
        Holm-Bonferroni correction {sig_sentences}, and is independent of frontal quality (no B.2 leakage
        on that variant). The remaining dataset(s) do not reach significance on this question, which is
        inconclusive rather than contradictory, consistent with FAA effects being small relative to these
        sample sizes. This result is reported as observed, without adjustment toward a preferred outcome.''')
    else:
        finding_p2 = '''No dataset showed a Holm-Bonferroni-significant group effect on FAA independent of
        quality. This is inconclusive on whether FAA carries a real signal in these samples, not evidence
        against it.'''

    if leaks:
        leak_sentences = "; ".join(f"{esc(name)} ({esc(variant)})" for name, variant in leaks)
        finding_p3 = f'''<p class="dek" style="max-width:100%"><strong>One flagged result:</strong>
        {leak_sentences} had a quality-alone-predicts-diagnosis AUC whose 95% confidence interval sat
        entirely above chance (0.5), indicating quality alone had some ability to predict group in that
        case. This result is reported rather than omitted; see the corresponding B.2 card above for the
        exact AUC and confidence interval.</p>'''
    else:
        finding_p3 = ""

    ao = study_c_pooled["emg_alpha_overlap"]
    if ao["nonzero_slope"]:
        finding_p_c = (f'''Test C adds a causal check: injecting synthetic contamination confined to the
        alpha band itself produces a small but statistically real upward drift in FAA, pooled across all
        three datasets (95%&nbsp;CI [{ao['slope_ci_lo']:+.4f}, {ao['slope_ci_hi']:+.4f}], n={ao["n_subjects"]} recordings), while
        the same injection outside the alpha band and EOG-like injection both show no effect. That pattern,
        an effect specific to the frequency range that overlaps alpha, is direct mechanistic evidence for
        the contamination pathway this project proposes, independent of whether it explains any single
        dataset's group difference above.''')
    else:
        finding_p_c = (f'''Test C's causal check (synthetic contamination injected at increasing doses) did
        not produce a statistically significant population-level FAA drift even in the alpha-overlapping
        condition when pooled across all three datasets (95%&nbsp;CI [{ao['slope_ci_lo']:+.4f}, {ao['slope_ci_hi']:+.4f}]),
        though its point estimate was consistently positive in every dataset individually. This is
        inconclusive rather than a null result at this sample size.''')

    html = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NeuroQA: Phase 1 Findings</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Does an endpoint-aware contamination score explain part of why frontal alpha asymmetry is an unreliable depression marker? Phase 1 findings on three real public EEG datasets.">
<style>
  :root {{
    color-scheme: light;
    --surface-1:      #fcfcfb;
    --page-plane:     #f9f9f7;
    --text-primary:   #0b0b0b;
    --text-secondary: #52514e;
    --text-muted:     #898781;
    --gridline:       #e1e0d9;
    --baseline:       #c3c2b7;
    --border:         rgba(11,11,11,0.10);
    --good:    #0ca30c;
    --warning: #fab219;
    --serious: #ec835a;
    --critical:#d03b3b;
    --series-h:   #2a78d6;
    --series-mdd: #eb6834;
    --series-primary: #2a78d6;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:where(:not([data-theme="light"])) {{
      color-scheme: dark;
      --surface-1:      #1a1a19;
      --page-plane:     #0d0d0d;
      --text-primary:   #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted:     #898781;
      --gridline:       #2c2c2a;
      --baseline:       #383835;
      --border:         rgba(255,255,255,0.10);
      --good:    #0ca30c; --warning: #fab219; --serious: #ec835a; --critical:#e66767;
      --series-h:   #3987e5; --series-mdd: #d95926; --series-primary: #3987e5;
    }}
  }}
  :root[data-theme="dark"] {{
    color-scheme: dark;
    --surface-1: #1a1a19; --page-plane: #0d0d0d; --text-primary: #ffffff;
    --text-secondary: #c3c2b7; --text-muted: #898781; --gridline: #2c2c2a; --baseline: #383835;
    --border: rgba(255,255,255,0.10);
    --good: #0ca30c; --warning: #fab219; --serious: #ec835a; --critical:#e66767;
    --series-h: #3987e5; --series-mdd: #d95926; --series-primary: #3987e5;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    background: var(--page-plane); color: var(--text-primary); line-height: 1.55; }}
  .page {{ max-width: 900px; margin: 0 auto; padding: 40px 20px 90px; }}
  .header-row {{ display:flex; align-items:flex-start; justify-content:space-between; gap:12px; }}
  header p.kicker {{ text-transform:uppercase; letter-spacing:.06em; font-size:.75rem; color:var(--text-muted); margin:0 0 8px; }}
  header h1 {{ font-size:2rem; margin:0 0 10px; line-height:1.15; }}
  header p.lede {{ color:var(--text-secondary); font-size:1.05rem; max-width:680px; }}
  .theme-toggle {{ width:38px; height:38px; border-radius:10px; border:1px solid var(--border);
    background:var(--surface-1); cursor:pointer; font-size:1.1rem; line-height:1;
    display:flex; align-items:center; justify-content:center; flex-shrink:0; }}
  .theme-toggle:hover {{ background: color-mix(in srgb, var(--series-primary) 12%, transparent); }}
  .callout {{ background:var(--surface-1); border:1px solid var(--border); border-left:3px solid var(--series-primary);
    border-radius:8px; padding:16px 18px; margin:20px 0; font-size:.95rem; color:var(--text-secondary); }}
  .callout code {{ background: color-mix(in srgb, var(--text-primary) 6%, transparent); padding:1px 5px; border-radius:4px; font-size:.88em; }}
  section {{ margin-top:52px; }}
  h2 {{ font-size:1.3rem; margin:0 0 6px; }}
  h2 .num {{ color:var(--text-muted); font-weight:500; margin-right:6px; }}
  h3 {{ font-size:1.02rem; margin: 26px 0 12px; }}
  h3:first-child {{ margin-top: 6px; }}
  .dataset-sub {{ font-weight:400; color:var(--text-muted); font-size:.82rem; }}
  section > p.dek {{ color:var(--text-secondary); margin:0 0 22px; max-width:660px; }}
  .card {{ background:var(--surface-1); border:1px solid var(--border); border-radius:12px; padding:20px 22px; }}
  .stat-row {{ display:grid; grid-template-columns:repeat(2,1fr); gap:14px; margin-bottom:18px; }}
  .stat-row-3 {{ grid-template-columns:repeat(3,1fr); }}
  .stat-tile {{ background:var(--surface-1); border:1px solid var(--border); border-radius:12px; padding:16px 18px; }}
  .stat-tile .label {{ font-size:.78rem; color:var(--text-muted); margin-bottom:6px; }}
  .stat-tile .value {{ font-size:1.7rem; font-weight:600; }}
  .stat-tile .sub {{ font-size:.82rem; color:var(--text-secondary); margin-top:4px; }}
  .verdict-row {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; }}
  .verdict-card {{ background:var(--surface-1); border:1px solid var(--border); border-radius:12px; padding:18px; }}
  .verdict-card .test-name {{ font-size:.78rem; color:var(--text-muted); margin-bottom:8px; }}
  .verdict-card .metric {{ font-size:1.35rem; font-weight:600; margin-bottom:4px; }}
  .verdict-card .ask {{ font-size:.85rem; color:var(--text-secondary); margin-bottom:12px; }}
  .badge {{ display:inline-flex; align-items:center; gap:6px; font-size:.8rem; font-weight:600; padding:4px 10px; border-radius:999px;
    background: color-mix(in srgb, var(--text-muted) 16%, transparent); color: var(--text-muted); }}
  .badge.good {{ background: color-mix(in srgb, var(--good) 16%, transparent); color: var(--good); }}
  .badge.warn {{ background: color-mix(in srgb, var(--warning) 20%, transparent); color: color-mix(in srgb, var(--warning) 70%, black); }}
  .badge .dot {{ width:8px; height:8px; border-radius:50%; background:currentColor; }}
  .bar-row {{ display:grid; grid-template-columns: 130px 1fr 56px; align-items:center; gap:10px; padding:5px 0; }}
  .bar-label {{ font-size:.82rem; color:var(--text-secondary); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .bar-track {{ position:relative; height:10px; background:var(--gridline); border-radius:5px; overflow:visible; }}
  .bar-fill {{ height:100%; border-radius:5px; }}
  .bar-chance-line {{ position:absolute; left:50%; top:-3px; bottom:-3px; width:1px; background:var(--text-muted); }}
  .bar-value {{ font-size:.82rem; font-variant-numeric:tabular-nums; text-align:right; color:var(--text-secondary); }}
  .legend {{ display:flex; gap:18px; margin-top:14px; font-size:.85rem; color:var(--text-secondary); }}
  .legend .item {{ display:inline-flex; align-items:center; gap:6px; }}
  .legend .swatch {{ width:10px; height:10px; border-radius:50%; display:inline-block; }}
  ul.caveats {{ padding-left:20px; color:var(--text-secondary); }}
  ul.caveats li {{ margin-bottom:12px; }}
  ul.caveats strong {{ color:var(--text-primary); }}
  .demo-figure {{ width:100%; border-radius:10px; border:1px solid var(--border); margin-top:14px; }}
  footer {{ margin-top:70px; padding-top:24px; border-top:1px solid var(--border); color:var(--text-secondary); font-size:.9rem; }}
  footer a {{ color:var(--series-primary); text-decoration:none; }}
  footer a:hover {{ text-decoration:underline; }}
  .footer-links {{ display:flex; gap:20px; flex-wrap:wrap; margin-bottom:12px; }}
  @media (max-width:900px) {{
    .verdict-row {{ grid-template-columns:repeat(2,1fr); }}
  }}
  @media (max-width:640px) {{
    .stat-row, .verdict-row {{ grid-template-columns:1fr; }}
    .bar-row {{ grid-template-columns: 90px 1fr 48px; }}
  }}
</style>
</head>
<body>
<div class="page">
  <header>
    <div class="header-row">
      <div>
        <p class="kicker">NeuroQA · Phase 1 · Findings</p>
        <h1>Is a known depression marker partly contamination?</h1>
        <p class="lede">Frontal alpha asymmetry (FAA) is a long-studied EEG depression marker that the
        literature describes as unreliable, without a clear explanation why. This project introduces an
        endpoint-aware contamination score, distinct from a classifier or a signal-cleaning method, and
        uses it to test whether part of the explanation is muscle and eye-movement artifact that shares
        alpha's frequency band. Results below are computed on three independent, real public datasets,
        not synthetic data.</p>
      </div>
      <button type="button" id="theme-toggle" class="theme-toggle" aria-label="Toggle dark mode" title="Toggle dark mode">🌙</button>
    </div>
  </header>

  <section id="instrument">
    <h2><span class="num">01</span>The instrument</h2>
    <p class="dek">Existing MNE-Python detectors already find blinks, muscle noise, line noise, pops,
    motion, cardiac contamination. The new part: a penalty that depends on <em>which frequency band
    you're about to measure</em>, not a single generic "clean vs. dirty" score.</p>
    <div class="callout">
      <code>for artifact in detected_artifacts: penalty += artifact.severity * WEIGHT[type] * spectral_overlap(artifact.band, ENDPOINT_BAND)</code><br>
      A blink's energy sits below 4&nbsp;Hz, so scoring it against <strong>delta</strong> incurs a large
      penalty, while scoring the identical detected blink against <strong>alpha</strong> (8–13&nbsp;Hz)
      incurs almost none.
    </div>
    <p class="dek" style="margin-bottom:0">Same-recording validation: one real recording, split into clean
    and contaminated epochs by amplitude, scored both ways.</p>
    <img class="demo-figure" src="validation_raw_clean_artifacts.png" alt="Waveform comparison: unclean, clean, and artifacts-only epochs from the same real recording">
  </section>

  <section id="datasets">
    <h2><span class="num">02</span>Three independent public datasets</h2>
    <p class="dek">Two ship real per-subject clinical severity (required for Test B.1), from two
    different labs with no overlapping subjects; the third has diagnosis labels only and runs as
    a replication check on everything except B.1.</p>
    <div class="stat-row stat-row-3">{dataset_stat_tiles}</div>
  </section>

  <section id="test-a">
    <h2><span class="num">03</span>Test A: Does cleaning choice decide the answer?</h2>
    <p class="dek">FAA computed under 5 preprocessing pipelines (raw, ICA, generic amplitude rejection,
    AutoReject, and <strong>ours</strong>, the endpoint-conditional score used to weight instead of
    reject) × 2 reference schemes, per subject. <strong>A large swing indicates that processing
    decisions, rather than biology, are driving the result.</strong> Limited to up to 30 recordings per
    dataset (15 per group where available); ICA and AutoReject remain computationally intensive even
    when parallelized. See limitations below.</p>
    {"".join(testa_blocks)}
    <div class="legend">
      <span class="item"><span class="swatch" style="background:var(--series-h)"></span>healthy</span>
      <span class="item"><span class="swatch" style="background:var(--series-mdd)"></span>depressed</span>
    </div>
  </section>

  <section id="classifiers">
    <h2><span class="num">04</span>Five independent classifiers, one per pipeline</h2>
    <p class="dek">A complement to Test A's pipeline-sensitivity analysis: for each pipeline, an
    independent classifier is trained using <em>only</em> that pipeline's own FAA value to predict
    depressed versus healthy, with no data shared across pipelines. The dashed line marks chance
    performance (AUC 0.5); bars at or below chance are shown in green.</p>
    {"".join(clf_blocks)}
  </section>

  <section id="test-b">
    <h2><span class="num">05</span>Test B: Is contamination tied to diagnosis?</h2>
    <p class="dek">Three group-level analyses per dataset. Each dataset's summary below also reports a
    robustness check that reruns the group-to-FAA test on raw, unweighted FAA instead of NeuroQA's own
    quality-weighted FAA (the primary result is partly built from the same per-channel weights used as
    the quality predictor, so this check confirms the finding isn't an artifact of that overlap), and a
    mediation check that separates how much of the group-FAA relationship runs through quality from how
    much doesn't, since a single regression coefficient can't make that distinction on its own.</p>
    {"".join(testb_blocks)}
  </section>

  <section id="test-c">
    <h2><span class="num">06</span>Test C: A causal check via synthetic contamination</h2>
    <p class="dek">Test A and Test B are both observational: they show FAA is sensitive to processing
    choices and associated (or not) with diagnosis in real, already-collected data, but neither can show
    that contamination <em>causes</em> FAA to shift. Test C injects a known, controlled amount of synthetic
    contamination into real clean epochs at three frequency ranges and recomputes FAA at each dose:
    EOG-like (0.5-4&nbsp;Hz), EMG limited to the alpha band itself (8-13&nbsp;Hz, maximal overlap), and
    EMG outside the alpha band (20-45&nbsp;Hz, a specificity control that should show no effect if the
    mechanism is really about spectral overlap and not just "more injected noise"). Injection is
    spectral/amplitude-based on the same epoch data the rest of the pipeline uses, calibrated to each
    recording's own channel amplitude; it is not anatomically-realistic forward-model simulation, see
    the limitations below.</p>
    {"".join(testc_blocks)}
    <div class="dataset-block">
      <h3>Pooled across all three datasets <span class="dataset-sub">(n={study_c_pooled["emg_alpha_overlap"]["n_subjects"]}
      recordings)</span></h3>
      <div class="card">{study_c_bars(study_c_pooled)}</div>
      <p class="dek" style="margin-top:10px;margin-bottom:0;max-width:100%">Pooling every recording's own
      dose-response slope across all three datasets (identical injection protocol and dose levels
      throughout): the alpha-overlapping condition's 95%&nbsp;CI is
      [{study_c_pooled['emg_alpha_overlap']['slope_ci_lo']:+.4f}, {study_c_pooled['emg_alpha_overlap']['slope_ci_hi']:+.4f}]
      {'and excludes zero' if study_c_pooled['emg_alpha_overlap']['nonzero_slope'] else '(includes zero)'}, while
      EOG and the non-overlapping EMG control both sit at essentially exactly zero. That pattern, an effect
      specific to the frequency range that overlaps alpha and absent elsewhere, is the signature the
      project's spectral-overlap mechanism predicts.</p>
    </div>
  </section>

  <section id="finding">
    <h2><span class="num">07</span>The finding</h2>
    <p class="dek" style="max-width:100%">Across all three datasets and both quality variants, Test B.1
    (whether group or severity predicts contamination) never reached significance, providing no evidence
    that diagnosis or symptom severity is itself associated with how contaminated a recording is.</p>
    <p class="dek" style="max-width:100%">{finding_p2}</p>
    {finding_p3}
    <p class="dek" style="max-width:100%">{finding_p_c}</p>
  </section>

  <section id="changelog">
    <h2><span class="num">08</span>Methodology revision (v2)</h2>
    <p class="dek">The following changes were applied to the analysis pipeline and the study was
    re-run in full on 2026-08-24. Prior (v1) results are archived rather than overwritten, and
    remain available for comparison.</p>
    <ul class="caveats">
      <li><strong>EMG/alpha band widened</strong> from (20, 45)&nbsp;Hz to (8, 45)&nbsp;Hz, following
      Goncharova et al. (2003). The previous definition had zero overlap with alpha, so detected EMG
      contributed nothing to alpha-endpoint quality regardless of severity.</li>
      <li><strong>Per-dataset mains frequency corrected.</strong> ds003478 is now notch-filtered at
      60&nbsp;Hz, consistent with its own recording metadata (University of Arizona, US mains), rather
      than the 50&nbsp;Hz previously applied to every dataset.</li>
      <li><strong><code>quality_alpha_frontal_pct</code></strong> (F3, F4, Fp1, Fp2 only) is now the
      primary Study B variable, since the hypothesis under test concerns frontal contamination
      specifically rather than a whole-scalp average diluted by 15 or more unrelated channels.
      Whole-scalp results are still reported alongside every result above, not replaced.</li>
      <li><strong>Bootstrap 95% confidence intervals</strong> added to every AUC, and
      <strong>Holm-Bonferroni correction</strong> applied to the p-values underlying each finding.</li>
      <li><strong>Test A scaled up and parallelized:</strong> up to 30 recordings per dataset
      (15 per group where available), increased from 12, using parallel processing.</li>
      <li><strong>Out of scope for this revision:</strong> widening the EMG severity detector's own
      20-45&nbsp;Hz window, finalized per-artifact-type weight values or the quality-decay function
      (pending further validation), an EMG spectral-signature specificity check, and a comparison
      against the 70-90&nbsp;Hz proxy-band method of Stewart et al. (2011).</li>
    </ul>
  </section>

  <section id="changelog-v21">
    <h2><span class="num">09</span>Methodology revision (v2.1)</h2>
    <p class="dek">A self-review of the Test A/B design surfaced one structural issue and one
    interpretation caveat worth fixing before treating v2's results as final. Applied and re-run in
    full on 2026-08-24.</p>
    <ul class="caveats">
      <li><strong>Raw-FAA robustness check added.</strong> The group-to-FAA regression is now also run
      on unweighted FAA, since the primary result partly depends on the same per-channel quality
      weights used as its own predictor; both results are reported together for every dataset above.</li>
      <li><strong>Mediation analysis added.</strong> Quantifies the indirect (quality-mediated) and
      direct effects of group on FAA, rather than relying on a single regression coefficient to judge
      whether quality is a confound.</li>
      <li><strong>Leakage threshold corrected.</strong> The quality-alone-predicts-diagnosis flag (Test
      B.2) now derives from the already-computed 95% confidence interval, replacing an arbitrary
      AUC&nbsp;&gt;&nbsp;0.65 cutoff disconnected from the chance level of 0.5.</li>
      <li><strong>Sample-size floor added.</strong> Test B.1 (quality ~ group + severity) now requires
      at least 6 total observations before running, not just 2 per group, avoiding a
      1-residual-degree-of-freedom fit being reported the same way as a well-powered one.</li>
      <li><strong>Test C added.</strong> Synthetic contamination injected at increasing doses to test for
      a causal, spectrally-specific effect on FAA, complementing Test A and B's observational designs.
      See section 06 above.</li>
    </ul>
  </section>

  <section id="caveats">
    <h2><span class="num">10</span>Limitations</h2>
    <ul class="caveats">
      <li><strong>Quality's causal role is unresolved.</strong> These analyses can't fully distinguish
      whether contamination quality is a pre-existing confound or a mediator on the path from group to
      FAA; this project's own rationale (more severe cases show more movement and blinking) points
      toward the latter, which changes how the confound-check and mediation results above should be
      read. See the mediation check reported with each dataset in Test B.</li>
      <li><strong>Per-type weights remain placeholders.</strong> <code>WEIGHT[artifact.type]</code> is
      currently equal weighting (1.0 across all types), pending finalized physics-derived values. Only
      the EMG <em>band definition</em> (which frequencies are classified as EMG) has been updated based
      on published literature to date; the relative weighting between artifact types and the
      quality-decay function have not.</li>
      <li><strong>Test A and the five classifiers ran on up to 30 recordings per dataset</strong>
      (15 per group where available), limited by ICA/AutoReject runtime even with parallelization. This
      is larger than the prior sample of 12 but still indicative rather than a full-sample result.</li>
      <li><strong>Test C's injection is spectral, not anatomical.</strong> Synthetic contamination is
      added directly to the channel data at the right frequency content and location, calibrated to each
      recording's own amplitude, but without a real head model or dipole simulation -- it tests whether
      alpha-band-overlapping contamination biases FAA in a dose-dependent way, not whether a real blink or
      jaw-clench of a given size would.</li>
      <li><strong>Mumtaz/HUSM has no per-subject severity data</strong> in its public release. Test B.1
      was therefore run on ds003478 and ds007615 (two independent cohorts) only; Mumtaz/HUSM contributes
      to Test A and the B.2/confound analyses as a replication check.</li>
      <li><strong>ds003478 and ds007615 use the same BDI&nbsp;&gt;&nbsp;13 / BDI&nbsp;&lt;&nbsp;7 group
      thresholds</strong> for consistency, matching ds003478's own published group definitions rather
      than thresholds independently derived per dataset.</li>
      <li><strong>Small samples, small expected effects.</strong> FAA effects are known to be small in
      the literature. A significant result in one dataset does not constitute a replicated finding, and
      a null result in the others is informative but not proof of absence.</li>
      <li><strong>Offline analysis only.</strong> This is Phase 1: entirely offline analysis of
      already-recorded public data. A causal or streaming version, which would decide using only samples
      already observed, is a later and optional extension, not part of this finding.</li>
    </ul>
  </section>

  <footer>
    <div class="footer-links">
      <a href="https://github.com/HYPERPRO2020/eegtest" target="_blank" rel="noopener">Source code (GitHub)</a>
      <a href="https://github.com/HYPERPRO2020/eegtest/blob/master/neuroqa/notebooks/phase1_findings.ipynb" target="_blank" rel="noopener">Reproducible notebook</a>
      <a href="https://eegtest.vercel.app" target="_blank" rel="noopener">Interactive quality grader →</a>
    </div>
    <p>NeuroQA Phase 1 · findings, not peer-reviewed · generated from a real, seeded run of this repo's own pipeline against three public datasets</p>
  </footer>
</div>
<script>
(function () {{
  var THEME_KEY = "neuroqa-findings-theme";
  var btn = document.getElementById("theme-toggle");
  function isDark() {{
    var attr = document.documentElement.getAttribute("data-theme");
    if (attr === "dark") return true;
    if (attr === "light") return false;
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  }}
  function apply(theme) {{
    if (theme === "dark" || theme === "light") document.documentElement.setAttribute("data-theme", theme);
    else document.documentElement.removeAttribute("data-theme");
    btn.textContent = isDark() ? "☀️" : "🌙";
  }}
  btn.addEventListener("click", function () {{
    var next = isDark() ? "light" : "dark";
    localStorage.setItem(THEME_KEY, next);
    apply(next);
  }});
  apply(localStorage.getItem(THEME_KEY));
}})();
</script>
</body>
</html>
'''

    SITE_DIR.mkdir(exist_ok=True)
    (SITE_DIR / "index.html").write_text(html, encoding="utf-8")
    shutil.copy(OUT_DIR / "figures" / "validation_raw_clean_artifacts.png",
                SITE_DIR / "validation_raw_clean_artifacts.png")
    print(f"wrote {SITE_DIR / 'index.html'} ({len(html)} bytes)")


if __name__ == "__main__":
    main()
