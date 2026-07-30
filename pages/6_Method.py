"""
Method & limitations.

This page exists because the product's central claim is methodological, and a
claim like that should be inspectable rather than asserted in marketing copy.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np
import pandas as pd
import streamlit as st

from batchrca import charts as CH
from batchrca import config as C
from batchrca import data as D
from batchrca import ui

ui.page("Method")
ui.sidebar_brand()

batch = D.load_batches()
ui.data_provenance()

ui.masthead("Method & limitations",
            "Why this application compares batches the way it does — and what it "
            "cannot tell you.")
st.markdown("")

# ---------------------------------------------------------------- claim
st.markdown("## The problem with pooled analysis")
ui.plain(
    "Most quality dashboards correlate a process parameter against a quality "
    "result across <b>every batch in the plant</b>. On this dataset that is not "
    "merely noisy — it frequently points the <b>opposite way</b> from the truth. "
    "Products differ systematically in both their settings and their outcomes, so "
    "a pooled correlation mostly measures product mix. This is Simpson's paradox, "
    "and in a quality investigation it sends people to the wrong root cause.")

cqa = st.selectbox("Quality attribute", D.cqa_options(batch),
                   format_func=lambda c: C.CQAS[c]["label"],
                   index=list(D.cqa_options(batch)).index("tbl_rsd_weight")
                   if "tbl_rsd_weight" in D.cqa_options(batch) else 0)

with st.spinner("Recomputing pooled vs within-cohort correlations…"):
    pw = D.pooled_vs_within_cached(cqa)

if pw.empty:
    st.warning("Not enough data for this attribute.")
    st.stop()

flips = int(pw["sign_flip"].sum())
k = st.columns(4)
with k[0]:
    ui.kpi("Drivers examined", f"{len(pw)}", "with enough variation")
with k[1]:
    ui.kpi("Sign reversals", f"{flips}",
           f"{flips / len(pw):.0%} of drivers", "alert" if flips else "ok")
with k[2]:
    ui.kpi("Cohorts used", f"{int(pw['n_cohorts'].max())}",
           "products with ≥20 batches")
with k[3]:
    eta = C.CQAS[cqa].get("eta2_product", np.nan)
    ui.kpi("Explained by product alone", f"{eta:.0%}" if np.isfinite(eta) else "—",
           "variance from product identity")

st.plotly_chart(CH.pooled_vs_within_chart(pw), use_container_width=True)
ui.plain(
    "<b>Grey ×</b> is the naive pooled correlation across all products. "
    "<b>Teal ●</b> is the correlation measured inside each product cohort and "
    "then combined. <b>Red connectors mark drivers whose direction reverses.</b> "
    "For those variables, a pooled dashboard would tell an investigator to look "
    "for exactly the wrong condition.")

st.markdown("#### Reversed drivers")
fl = pw[pw["sign_flip"]][["label", "domain", "pooled_r", "within_r", "n_cohorts"]]
if len(fl):
    st.dataframe(
        fl.rename(columns={"label": "Driver", "domain": "Where",
                           "pooled_r": "Pooled r", "within_r": "Within-cohort r",
                           "n_cohorts": "Cohorts"}),
        use_container_width=True, hide_index=True,
        column_config={"Pooled r": st.column_config.NumberColumn(format="%+.3f"),
                       "Within-cohort r": st.column_config.NumberColumn(format="%+.3f")})
else:
    st.success("No sign reversals for this attribute — pooled analysis happens to "
               "be safe here. That is a property of this attribute, not a general "
               "licence to pool.")

# ---------------------------------------------------------------- variance
st.markdown("## How much is just 'which product is it?'")
rows = []
for c, m in C.CQAS.items():
    if c not in batch.columns:
        continue
    d = batch.dropna(subset=[c])
    if len(d) < 50:
        continue
    grand = d[c].var()
    within = d.groupby("code")[c].transform("mean")
    eta2 = 1 - (d[c] - within).var() / grand if grand else np.nan
    rows.append({"Quality attribute": m["label"],
                 "Variance from product identity": eta2,
                 "Batches": len(d),
                 "Pooled analysis": "unsafe" if eta2 > 0.3 else "lower risk"})
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
             column_config={"Variance from product identity":
                            st.column_config.ProgressColumn(
                                format="%.0f%%", min_value=0, max_value=1)})
ui.plain(
    "Attributes at the top of this table are dominated by product identity: "
    "comparing them across products mostly measures the recipe, not the run. "
    "Tablet weight RSD is the least product-bound outcome here, which makes it "
    "the most legitimate candidate for cross-product analysis.")

# ---------------------------------------------------------------- pipeline
st.markdown("## Analytical pipeline")
st.markdown(
    """<div class='bl-card'><div style='font-size:.88rem;line-height:1.75'>
    <b>1 · Peer cohort</b> — a batch is matched to batches of the same product
    code. Below 20 peers the cohort falls back to the same strength and the
    substitution is stated on screen.<br>
    <b>2 · Exception scan</b> — every quality attribute is scored with a
    median/MAD robust z against its cohort, so outliers cannot inflate the limits
    that judge them. Watch at 2 SD, Investigate at 3 SD, in the direction that is
    bad for that attribute.<br>
    <b>3 · Same-quantity guard</b> — signals that restate the outcome (in-process
    weight RSD against lab weight RSD; waste against yield) are removed from
    root-cause ranking and reported separately as early-detection signals.<br>
    <b>4 · Three evidence tiers</b> — descriptive contrast between best and worst
    quartile; rank correlation across the cohort; permutation importance from a
    random forest with 5-fold cross-validation. Reported separately, never fused
    into one score.<br>
    <b>5 · Corroboration</b> — a driver is promoted only when at least two
    independent tiers rank it highly.
    </div></div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------- limits
st.markdown("## Limitations")
c1, c2 = st.columns(2)
with c1:
    st.markdown(
        """<div class='bl-card'><h4>Dataset</h4>
        <div style='font-size:.86rem;line-height:1.65'>
        • <b>No registered specifications.</b> Every limit shown is derived from
        historical behaviour, not from a filed specification.<br>
        • <b>No deviation or CAPA records.</b> There is no ground truth about what
        investigators actually concluded, so the ranking cannot be scored against
        real root causes.<br>
        • <b>Anonymised lots.</b> Material lots are integers; no supplier names,
        goods-in dates or certificates.<br>
        • <b>Monthly batch dates.</b> `start` resolves to a month, so
        fine-grained sequencing between batches is unavailable.<br>
        • <b>API impurity L missing for ~36% of batches</b> — excluded by default
        rather than imputed.
        </div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(
        """<div class='bl-card'><h4>Inference</h4>
        <div style='font-size:.86rem;line-height:1.65'>
        • <b>Observational, not experimental.</b> Nothing here was randomised;
        associations are leads, not proven causes.<br>
        • <b>Unrecorded factors exist.</b> Operator, ambient conditions, tooling
        age and maintenance history are not in the data and could drive both the
        settings and the outcome.<br>
        • <b>Collinearity.</b> Compression force, fill depth and thickness move
        together; importance can be split arbitrarily between them.<br>
        • <b>Multiplicity.</b> Screening dozens of drivers produces significant
        results by chance; the app flags this when it happens.<br>
        • <b>Modest R².</b> Within a cohort the recorded variables often explain a
        limited share of outcome variance. That is reported, not hidden.
        </div></div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    """<div class='bl-meta'>
    <b>Provenance.</b> Built on the openly published dataset from Žagar &amp;
    Mihelič, “Big data collection in pharmaceutical manufacturing and its use for
    product quality predictions”, <i>Scientific Data</i> 9, 99 (2022), CC-BY 4.0,
    figshare 10.6084/m9.figshare.c.5645578. 1,005 real production batches of a
    tablet compression process.
    </div>""", unsafe_allow_html=True)
