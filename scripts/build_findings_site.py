"""Regenerate findings-site/index.html from the real results.json files for
both datasets -- keeps the site's numbers tied to an actual run instead of
hand-typed. Run after any re-run of run_local.py that changes outputs/.

Usage: python scripts/build_findings_site.py
"""
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "outputs"
SITE_DIR = REPO_ROOT / "findings-site"

DATASETS = [
    {"key": "ds003478", "label": "ds003478", "sub": "primary · real BDI severity",
     "color_var": "--series-primary"},
    {"key": "mumtaz", "label": "Mumtaz/HUSM", "sub": "secondary check · no severity",
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
        out.append(f'''<div class="bar-row">
          <span class="bar-label">{PIPELINE_LABEL[p]}</span>
          <div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%;background:{color}"></div>
            <div class="bar-chance-line"></div></div>
          <span class="bar-value">{r['auc']:.2f}</span>
        </div>''')
    return "\n".join(out)


def verdict_card(title, ask, metric, is_flag):
    badge = ('<span class="badge warn"><span class="dot"></span>flagged</span>' if is_flag
             else '<span class="badge good"><span class="dot"></span>no red flag</span>')
    return f'''<div class="verdict-card">
      <div class="test-name">{esc(title)}</div>
      <div class="ask">{esc(ask)}</div>
      <div class="metric">{esc(metric)}</div>
      {badge}
    </div>'''


def dataset_testb_section(d, r):
    b = r["study_b"]
    r1 = b["regression_1_quality_on_group_severity"]
    cards = []
    if r1 is not None:
        p = r1["pvalues"]["group_mdd"]
        cards.append(verdict_card("B.1", "quality ~ group + severity", f"p = {p:.3f}", p < 0.05))
    else:
        cards.append(verdict_card("B.1", "quality ~ group + severity", "skipped — no severity data", False))
    r2 = b["regression_2_faa_on_group_quality"]
    qp = r2["pvalues"]["quality_alpha_pct"]
    cards.append(verdict_card("Confound check", "FAA ~ group + quality", f"p = {qp:.3f}", qp < 0.05))
    r3 = b["regression_3_quality_classifies_group"]
    cards.append(verdict_card("B.2 — the sharp one", "quality alone → group (5-fold CV)",
                               f"AUC = {r3['auc']:.3f}", r3["auc"] > 0.65))
    return f'''<div class="dataset-block">
      <h3>{esc(d['label'])} <span class="dataset-sub">({esc(d['sub'])}, n={b['n']})</span></h3>
      <div class="verdict-row">{"".join(cards)}</div>
    </div>'''


def main():
    results = {d["key"]: load(d["key"]) for d in DATASETS}

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

    testb_blocks = [dataset_testb_section(d, results[d["key"]]) for d in DATASETS]

    n_primary = results["ds003478"]["n_recordings"]
    n_secondary = results["mumtaz"]["n_recordings"]

    html = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NeuroQA — Phase 1 Findings</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Does an endpoint-aware contamination score explain part of why frontal alpha asymmetry is an unreliable depression marker? Phase 1 findings on two real public EEG datasets.">
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
  .stat-tile {{ background:var(--surface-1); border:1px solid var(--border); border-radius:12px; padding:16px 18px; }}
  .stat-tile .label {{ font-size:.78rem; color:var(--text-muted); margin-bottom:6px; }}
  .stat-tile .value {{ font-size:1.7rem; font-weight:600; }}
  .stat-tile .sub {{ font-size:.82rem; color:var(--text-secondary); margin-top:4px; }}
  .verdict-row {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; }}
  .verdict-card {{ background:var(--surface-1); border:1px solid var(--border); border-radius:12px; padding:18px; }}
  .verdict-card .test-name {{ font-size:.78rem; color:var(--text-muted); margin-bottom:8px; }}
  .verdict-card .metric {{ font-size:1.35rem; font-weight:600; margin-bottom:4px; }}
  .verdict-card .ask {{ font-size:.85rem; color:var(--text-secondary); margin-bottom:12px; }}
  .badge {{ display:inline-flex; align-items:center; gap:6px; font-size:.8rem; font-weight:600; padding:4px 10px; border-radius:999px; }}
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
        literature calls unreliable without knowing why. We built an endpoint-aware contamination score
        — not a classifier, not a cleaner — and used it to test whether part of the answer is muscle
        and eye-movement artifact that happens to share alpha's frequency band. Run on two independent
        real public datasets, not synthetic data.</p>
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
      A blink's energy sits below 4&nbsp;Hz — score it against <strong>delta</strong> and it costs a lot;
      score the exact same detected blink against <strong>alpha</strong> (8–13&nbsp;Hz) and it costs almost nothing.
    </div>
    <p class="dek" style="margin-bottom:0">Same-recording sanity check — one real recording, split into clean vs.
    contaminated epochs by amplitude, scored both ways:</p>
    <img class="demo-figure" src="validation_raw_clean_artifacts.png" alt="Waveform comparison: unclean, clean, and artifacts-only epochs from the same real recording">
  </section>

  <section id="datasets">
    <h2><span class="num">02</span>Two independent public datasets</h2>
    <p class="dek">One primary, chosen specifically because it ships real per-subject clinical severity
    (required for Test B.1); one secondary, run as an independent replication check on everything except B.1.</p>
    <div class="stat-row">
      <div class="stat-tile"><div class="label">ds003478 (OpenNeuro, CC0) — primary</div>
        <div class="value">{n_primary}</div>
        <div class="sub">recordings, real per-subject BDI severity, matches the dataset's own published groups</div></div>
      <div class="stat-tile"><div class="label">Mumtaz/HUSM (figshare, CC BY 4.0) — secondary check</div>
        <div class="value">{n_secondary}</div>
        <div class="sub">eyes-closed recordings, diagnosis label only, no severity data</div></div>
    </div>
  </section>

  <section id="test-a">
    <h2><span class="num">03</span>Test A — does cleaning choice decide the answer?</h2>
    <p class="dek">FAA computed under 5 preprocessing pipelines (raw, ICA, generic amplitude rejection,
    AutoReject, and <strong>ours</strong> — the endpoint-conditional score used to weight instead of
    reject) × 2 reference schemes, per subject. <strong>Red flag: a big swing means processing
    decisions, not biology, are steering the result.</strong> Bounded to 12 recordings per dataset —
    ICA and AutoReject are slow; see caveats.</p>
    {"".join(testa_blocks)}
    <div class="legend">
      <span class="item"><span class="swatch" style="background:var(--series-h)"></span>healthy</span>
      <span class="item"><span class="swatch" style="background:var(--series-mdd)"></span>depressed</span>
    </div>
  </section>

  <section id="classifiers">
    <h2><span class="num">04</span>Five independent classifiers, one per pipeline</h2>
    <p class="dek">Not just "how much does FAA move" (Test A) — for each pipeline, an independent
    classifier using <em>only</em> that pipeline's own FAA to guess depressed/healthy, none sharing
    data with each other. Dashed line = chance (AUC 0.5); green = at/below chance.</p>
    {"".join(clf_blocks)}
  </section>

  <section id="test-b">
    <h2><span class="num">05</span>Test B — is contamination tied to diagnosis?</h2>
    <p class="dek">Three group-level analyses per dataset.</p>
    {"".join(testb_blocks)}
  </section>

  <section id="finding">
    <h2><span class="num">06</span>The finding</h2>
    <p class="dek" style="max-width:100%">On both datasets, independently, under the current placeholder
    weights, none of the three red flags fired: group/severity doesn't predict contamination (B.1),
    contamination doesn't confound the FAA–group relationship, and contamination alone can't tell
    healthy from depressed above chance (B.2). That's a real result — a null result, reported as one,
    not tuned toward drama. It does <strong>not</strong> mean the contamination hypothesis is false; it
    means these two datasets, at this sample size, with placeholder rather than physics-derived
    weights, don't show it. See caveats before drawing a stronger conclusion either way.</p>
  </section>

  <section id="caveats">
    <h2><span class="num">07</span>Caveats — read before citing any of this</h2>
    <ul class="caveats">
      <li><strong>The weights are placeholders.</strong> <code>WEIGHT[artifact.type]</code> is equal
      weighting (1.0 across the board) pending the real physics-derived values. Every number above is
      contingent on this.</li>
      <li><strong>EMG/alpha overlap is currently zero.</strong> The muscle-artifact band (20–45&nbsp;Hz)
      doesn't overlap alpha (8–13&nbsp;Hz) under the current placeholder band assignment — in tension
      with the hypothesis that muscle noise sits at the same frequency as alpha. If real EMG
      contamination extends into alpha, the alpha-quality numbers above likely <em>underestimate</em>
      contamination there. Needs domain sign-off, not a unilateral widening to make the hypothesis land.</li>
      <li><strong>Test A and the 5 classifiers ran on 12 recordings per dataset</strong>, bounded by
      ICA/AutoReject runtime — suggestive, not a large-N result.</li>
      <li><strong>Mumtaz/HUSM has no per-subject severity</strong> in its public deposit — Test B.1 only
      ran on ds003478; Mumtaz's contribution is Test A and B.2/confound-check as a replication.</li>
      <li><strong>Small dataset, small expected effect.</strong> FAA effects are known to be small in
      the literature — a null result here is informative but not proof of absence.</li>
      <li><strong>Offline only.</strong> This is Phase 1 — entirely offline analysis of already-recorded
      public data. A causal/streaming version (deciding using only samples already seen) is a later,
      optional stretch, not part of this finding.</li>
    </ul>
  </section>

  <footer>
    <div class="footer-links">
      <a href="https://github.com/HYPERPRO2020/eegtest" target="_blank" rel="noopener">Source code (GitHub)</a>
      <a href="https://github.com/HYPERPRO2020/eegtest/blob/master/neuroqa/notebooks/phase1_findings.ipynb" target="_blank" rel="noopener">Reproducible notebook</a>
      <a href="https://eegtest.vercel.app" target="_blank" rel="noopener">Interactive quality grader →</a>
    </div>
    <p>NeuroQA Phase 1 · findings, not peer-reviewed · generated from a real, seeded run of this repo's own pipeline against two public datasets</p>
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
