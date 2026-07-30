"""Review-by-exception queue: the work list a quality reviewer opens each morning."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pandas as pd
import streamlit as st

from batchlens import config as C
from batchlens import data as D
from batchlens import ui

ui.page("Review queue")
ui.sidebar_brand()

batch = D.load_batches()
queue = D.load_queue()
exceptions = D.load_exceptions()

ui.masthead("Review by exception",
            "Batches ranked by how far they sit from their own peer cohort. "
            "Clear batches require no manual review.")

# ---------------------------------------------------------------- filters
st.sidebar.markdown("### Filters")
status_sel = st.sidebar.multiselect("Status", ["Investigate", "Watch", "Clear"],
                                    default=["Investigate", "Watch"])
prods = ["All"] + sorted(batch["product_id"].unique().tolist())
prod_sel = st.sidebar.selectbox("Product", prods)
cqa_sel = st.sidebar.multiselect(
    "Flagged on attribute",
    [c for c in C.CQAS if c in batch.columns],
    format_func=lambda c: C.CQAS[c]["label"])
ui.data_provenance()

q = queue[queue["status"].isin(status_sel)] if status_sel else queue
if prod_sel != "All":
    q = q[q["product_id"] == prod_sel]
if cqa_sel:
    hits = exceptions[exceptions["cqa"].isin(cqa_sel)]["batch"].unique()
    q = q[q["batch"].isin(hits)]

# ---------------------------------------------------------------- summary
c = st.columns(4)
with c[0]:
    ui.kpi("In queue", f"{len(q):,}", "matching current filters")
with c[1]:
    ui.kpi("Investigate", f"{int((q['status'] == 'Investigate').sum())}",
           "≥3 robust SD from peers", "alert")
with c[2]:
    ui.kpi("Watch", f"{int((q['status'] == 'Watch').sum())}",
           "2–3 robust SD from peers", "watch")
with c[3]:
    total_clear = int((queue["status"] == "Clear").sum())
    ui.kpi("Cleared automatically", f"{total_clear:,}",
           f"{100 * total_clear / len(queue):.0f}% of all batches", "ok")

st.markdown("")
ui.plain(
    "<b>How a batch is scored.</b> Each quality attribute is compared with the "
    "distribution of the <i>same product code</i> using a median/MAD robust "
    "z-score, so a single extreme batch cannot inflate the limits that judge it. "
    "A batch is flagged <b>Watch</b> beyond 2 robust SD and <b>Investigate</b> "
    "beyond 3, in the direction that is bad for that attribute.")

st.markdown("## Queue")

show = q.copy()
show["Period"] = pd.to_datetime(show["start_date"]).dt.strftime("%b %Y")
show = show.rename(columns={
    "batch_id": "Batch", "product_id": "Product", "strength_label": "Strength",
    "status": "Status", "n_flags": "Flags", "max_severity": "Severity",
    "flagged_cqas": "Flagged attributes"})
cols = ["Batch", "Product", "Strength", "Period", "Status", "Flags",
        "Severity", "Flagged attributes"]

st.dataframe(
    show[cols], use_container_width=True, hide_index=True, height=430,
    column_config={
        "Severity": st.column_config.ProgressColumn(
            "Severity", help="Worst robust z-score across flagged attributes",
            format="%.1f", min_value=0,
            max_value=float(max(6.0, show["Severity"].max() if len(show) else 6))),
        "Flags": st.column_config.NumberColumn("Flags", width="small"),
    })

# ---------------------------------------------------------------- drilldown
st.markdown("## Open a batch")
if len(q):
    pick = st.selectbox(
        "Select a batch to investigate", q["batch_id"].tolist(),
        format_func=lambda b: (
            f"{b} — {q.loc[q['batch_id'] == b, 'status'].iat[0]} — "
            f"{q.loc[q['batch_id'] == b, 'flagged_cqas'].iat[0] or 'no flags'}"))
    bno = int(q.loc[q["batch_id"] == pick, "batch"].iat[0])
    st.session_state["focus_batch"] = bno

    det = exceptions[exceptions["batch"] == bno]
    if len(det):
        st.markdown("#### Why this batch was flagged")
        d = det.copy()
        d["Attribute"] = d["cqa"].map(lambda c: C.CQAS[c]["label"])
        d["Value"] = d.apply(
            lambda r: f"{r['value']:.3g} {C.CQAS[r['cqa']]['unit']}", axis=1)
        d["Cohort median"] = d["cohort_median"].map(lambda v: f"{v:.3g}")
        d["Control limits"] = d.apply(
            lambda r: f"{r['lcl']:.3g} – {r['ucl']:.3g}", axis=1)
        d["Robust z"] = d["z"].map(lambda v: f"{v:+.1f}")
        st.dataframe(d[["Attribute", "Value", "Cohort median", "Control limits",
                        "Robust z", "severity"]]
                     .rename(columns={"severity": "Status"}),
                     use_container_width=True, hide_index=True)

    a, b = st.columns(2)
    with a:
        st.page_link("pages/2_Batch_detail.py", label="→ Batch detail & trajectory")
    with b:
        st.page_link("pages/4_Root_cause.py", label="→ Run root-cause analysis")
else:
    st.info("No batches match the current filters.")
