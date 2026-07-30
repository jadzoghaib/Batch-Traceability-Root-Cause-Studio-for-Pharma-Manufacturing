"""
Root-cause analysis — the analytical centre of Batch Investigation Console.

Evidence is presented in three explicitly separate tiers because they license
different strengths of claim. Merging them into one "AI confidence score" would
look more impressive and mean considerably less.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np
import pandas as pd
import streamlit as st

from batchrca import analytics as A
from batchrca import charts as CH
from batchrca import config as C
from batchrca import data as D
from batchrca import ui

ui.page("Root cause")
ui.sidebar_brand()

batch = D.load_batches()
queue = D.load_queue()

# ---------------------------------------------------------------- scope
mode = st.sidebar.radio("Analyse", ["A specific batch", "A product cohort"])

focus = st.session_state.get("focus_batch")
if mode == "A specific batch":
    ids = batch["batch_id"].tolist()
    idx = ids.index(f"B-{focus:04d}") if focus and f"B-{focus:04d}" in ids else 0
    sel = st.sidebar.selectbox("Batch", ids, index=idx)
    brow = batch[batch["batch_id"] == sel].iloc[0]
    bno = int(brow["batch"])
    st.session_state["focus_batch"] = bno
    code = int(brow["code"])
    peers, cohort_label = A.peer_cohort(batch, bno)
else:
    codes = sorted(batch["code"].value_counts()
                   .loc[lambda s: s >= C.MIN_COHORT_N].index.tolist())
    code = st.sidebar.selectbox(
        "Product cohort", codes,
        format_func=lambda c: f"P-{c:02d} · {int((batch['code'] == c).sum())} batches")
    bno = None
    peers = batch[batch["code"] == code]
    cohort_label = f"product P-{code:02d}"
    brow = None

cqa = st.sidebar.selectbox("Quality attribute to explain", D.cqa_options(batch),
                           format_func=lambda c: C.CQAS[c]["label"])
adv = st.sidebar.checkbox("Include low-coverage attributes", value=False,
                          help="Adds attributes missing for >30% of batches, "
                               "such as API impurity L.")
ui.data_provenance()

meta = C.CQAS[cqa]
subtitle = (f"{brow['batch_id']} · {brow['product_id']} · explaining "
            f"{meta['label'].lower()}") if brow is not None else \
           (f"P-{code:02d} cohort · explaining {meta['label'].lower()}")
ui.masthead("Root-cause analysis", subtitle)
st.markdown("")

with st.spinner("Ranking candidate drivers within the peer cohort…"):
    res = A.run_rca(peers, cqa, cohort_label, focus_batch=bno,
                    include_low_coverage=adv)

if res.drivers.empty:
    st.warning(" ".join(res.warnings) or "Not enough data for this cohort.")
    st.stop()

# ---------------------------------------------------------------- header KPIs
k = st.columns(4)
with k[0]:
    ui.kpi("Peer cohort", f"{res.n_used}", cohort_label)
with k[1]:
    ui.kpi("Candidate drivers", f"{len(res.drivers)}", "screened")
with k[2]:
    r2 = res.model_r2
    ui.kpi("Model R² (5-fold CV)", f"{r2:.2f}" if r2 is not None else "n/a",
           f"±{res.model_r2_std:.2f}" if res.model_r2_std else "not fitted",
           "ok" if (r2 or 0) > 0.3 else "watch")
with k[3]:
    agree = int((res.drivers["tiers_agree"] >= 2).sum())
    ui.kpi("Corroborated drivers", f"{agree}", "≥2 evidence tiers agree",
           "ok" if agree else "watch")

for w in res.warnings:
    ui.caution(w)

if brow is not None:
    v = brow.get(cqa)
    z = A.robust_z(peers[cqa]).loc[brow.name]
    st.markdown("")
    ui.plain(
        f"<b>{brow['batch_id']}</b> recorded <b>{v:.2f} {meta['unit']}</b> for "
        f"{meta['label'].lower()}, which is <b>{z:+.1f} robust SD</b> from its "
        f"cohort median. The ranking below asks: among {len(res.drivers)} recorded "
        f"upstream conditions, which ones plausibly explain that?")

# ---------------------------------------------------------------- ranking
st.markdown("## Prioritised drivers")
st.plotly_chart(CH.driver_ranking(res.drivers), use_container_width=True)
ui.plain(
    "Bars show how strongly each condition tracks the outcome across comparable "
    "batches. <b>Teal</b> means at least two independent lines of evidence agree; "
    "grey means only one did, so treat it as weaker. Direction matters: a bar to "
    "the right means more of that condition goes with a higher value of the "
    "outcome.")

# ---------------------------------------------------------------- tiers
st.markdown("## Evidence detail")
st.markdown(
    "<span class='bl-tier'>TIER 1 · DESCRIPTIVE</span> this batch differs from peers"
    " &nbsp; <span class='bl-tier'>TIER 2 · ASSOCIATION</span> the pattern holds "
    "across the cohort &nbsp; <span class='bl-tier'>TIER 3 · MODEL</span> it carries "
    "weight alongside everything else", unsafe_allow_html=True)
st.markdown("")

d = res.drivers.head(12).copy()
disp = pd.DataFrame({
    "Driver": d["label"],
    "Where": d["domain"],
    "Association (r)": d["corr"],
    "Golden−poor gap": d["good_vs_poor_delta"],
    "Model importance": d["importance"],
    "This batch (SD)": d["batch_z"],
    "Tiers agreeing": d["tiers_agree"],
})
st.dataframe(
    disp, use_container_width=True, hide_index=True, height=430,
    column_config={
        "Association (r)": st.column_config.NumberColumn(
            format="%+.2f", help="Spearman rank correlation within the peer cohort"),
        "Golden−poor gap": st.column_config.NumberColumn(
            format="%+.2f", help="Median difference between best and worst "
                                 "quartile, in cohort SD units"),
        "Model importance": st.column_config.NumberColumn(
            format="%.3f", help="Permutation importance: drop in R² when this "
                                "variable is shuffled"),
        "This batch (SD)": st.column_config.NumberColumn(
            format="%+.1f", help="Robust z of the selected batch"),
        "Tiers agreeing": st.column_config.ProgressColumn(
            format="%d", min_value=0, max_value=3),
    })

# ---------------------------------------------------------------- narrative
st.markdown("## What this means")
top3 = res.drivers.head(3)
for i, (_, r) in enumerate(top3.iterrows(), 1):
    corroborated = r["tiers_agree"] >= 2
    st.markdown(
        f"""<div class='bl-card' style='margin-bottom:.6rem'>
          <div style='display:flex;justify-content:space-between'>
            <div style='font-weight:660'>{i}. {r['label']}</div>
            <div class='bl-meta'>{r['domain']} ·
              {'corroborated' if corroborated else 'single line of evidence'}</div>
          </div>
          <div style='margin-top:.4rem;font-size:.88rem;line-height:1.55'>
            {A.explain_driver(r, cqa)}</div>
          <div class='bl-meta' style='margin-top:.35rem'>{C.feat_plain(r['driver'])}</div>
        </div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------- evidence plot
st.markdown("## Inspect the evidence")
pick = st.selectbox("Driver", res.drivers["driver"].tolist(),
                    format_func=lambda c: C.feat_label(c))
st.plotly_chart(CH.driver_scatter(peers, pick, cqa, focus=bno),
                use_container_width=True)

# ---------------------------------------------------------------- confirmatory
conf = A.confirmatory_signals(peers, cqa, bno)
if len(conf):
    st.markdown("## Confirmatory in-process signals")
    ui.caution(
        "These signals measure <b>the same physical quantity</b> as the outcome, "
        "just earlier and with a different instrument — the press's own "
        "checkweigher rather than the QC lab. They correlate strongly by "
        "construction, so they are deliberately <b>excluded from the driver "
        "ranking above</b>: they confirm what happened, they do not explain why. "
        "Their value is timing — they are visible during the run, hours before a "
        "lab result exists.")
    cd = conf.copy()
    out = pd.DataFrame({"Signal": cd["label"],
                        "Correlation with outcome": cd["corr"]})
    if "batch_z" in cd:
        out["This batch (SD)"] = cd["batch_z"]
    st.dataframe(out, use_container_width=True, hide_index=True,
                 column_config={
                     "Correlation with outcome": st.column_config.NumberColumn(format="%+.2f"),
                     "This batch (SD)": st.column_config.NumberColumn(format="%+.1f")})

# ---------------------------------------------------------------- caveats
st.markdown("---")
st.markdown("### How to read this analysis")
c1, c2 = st.columns(2)
with c1:
    st.markdown(
        f"""<div class='bl-card'><h4>What it does</h4>
        <div style='font-size:.86rem;line-height:1.6'>
        • Compares only within <b>{cohort_label}</b>, never across products.<br>
        • Reports three independent evidence tiers separately.<br>
        • Promotes a driver only when tiers agree.<br>
        • Excludes signals that restate the outcome.<br>
        • Reports cross-validated model skill honestly, including when it is poor.
        </div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(
        f"""<div class='bl-card'><h4>What it cannot do</h4>
        <div style='font-size:.86rem;line-height:1.6'>
        • Prove causation — this is observational production data.<br>
        • See unrecorded factors (operator, humidity, tooling wear).<br>
        • Separate variables that always move together.<br>
        • Replace a formal deviation investigation.<br>
        • Substitute for a designed experiment to confirm a lead.
        </div></div>""", unsafe_allow_html=True)
