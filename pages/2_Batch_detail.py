"""Batch detail: full genealogy for one batch — materials, process, quality, trajectory."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np
import pandas as pd
import streamlit as st

from batchlens import analytics as A
from batchlens import charts as CH
from batchlens import config as C
from batchlens import data as D
from batchlens import ui

ui.page("Batch detail")
ui.sidebar_brand()

batch = D.load_batches()
queue = D.load_queue()
exceptions = D.load_exceptions()

# ---------------------------------------------------------------- selection
ids = batch["batch_id"].tolist()
default = st.session_state.get("focus_batch")
idx = ids.index(f"B-{default:04d}") if default and f"B-{default:04d}" in ids else 0
sel = st.sidebar.selectbox("Batch", ids, index=idx)
row = batch[batch["batch_id"] == sel].iloc[0]
bno = int(row["batch"])
st.session_state["focus_batch"] = bno
ui.data_provenance()

peers, cohort_label = A.peer_cohort(batch, bno)
qrow = queue[queue["batch"] == bno]
status = qrow["status"].iat[0] if len(qrow) else "Clear"

ui.masthead(
    f"{row['batch_id']}",
    f"{row['product_id']} · {row['strength_label']} · "
    f"{pd.to_datetime(row['start_date']):%B %Y} · "
    f"{int(row['planned_tablets']):,} tablets planned",
    ui.pill(status))
st.markdown("")

# ---------------------------------------------------------------- quality KPIs
st.markdown("## Quality outcomes")
cqas = [c for c in D.cqa_options(batch)][:5]
cols = st.columns(len(cqas))
for col, cqa in zip(cols, cqas):
    v = row.get(cqa)
    z = A.robust_z(peers[cqa]).loc[row.name] if peers[cqa].notna().sum() > 5 else np.nan
    bad = A._bad_direction_z(pd.Series([z]), C.CQAS[cqa]["direction"]).iat[0]
    tone = ("alert" if bad >= C.Z_INVESTIGATE else
            "watch" if bad >= C.Z_WATCH else "ok")
    with col:
        ui.kpi(C.CQAS[cqa]["label"],
               f"{v:.2f}" if pd.notna(v) else "—",
               f"{z:+.1f} robust SD vs peers" if np.isfinite(z) else
               C.CQAS[cqa]["unit"], tone)

st.markdown("")
ui.plain(f"Compared against <b>{len(peers)} peer batches</b> ({cohort_label}). "
         "A batch is only ever judged against structurally comparable batches — "
         "same product code where possible — because pooling different products "
         "reverses the apparent direction of many drivers.")

# ---------------------------------------------------------------- tabs
t1, t2, t3, t4 = st.tabs(["Position vs peers", "Compression trajectory",
                          "Materials genealogy", "Process record"])

with t1:
    cqa = st.selectbox("Quality attribute", D.cqa_options(batch),
                       format_func=lambda c: C.CQAS[c]["label"], key="bd_cqa")
    a, b = st.columns([1.55, 1])
    with a:
        st.markdown("#### Run chart within product")
        st.plotly_chart(CH.control_chart(peers, cqa, focus=bno),
                        use_container_width=True)
    with b:
        st.markdown("#### Distribution")
        st.plotly_chart(CH.cohort_distribution(peers, cqa, focus=bno),
                        use_container_width=True)
        v = row.get(cqa)
        s = peers[cqa].dropna()
        if pd.notna(v) and len(s):
            pct = (s < v).mean() * 100
            st.markdown(
                f"<div class='bl-meta'>This batch sits at the "
                f"<b>{pct:.0f}th percentile</b> of its cohort "
                f"({len(s)} batches).</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bl-meta'>{C.CQAS[cqa]['plain']}</div>",
                unsafe_allow_html=True)

with t2:
    st.markdown("#### 10-second compression trajectory")
    ts = D.load_timeseries(int(row["code"]), bno)
    if ts.empty:
        st.info("No time-series record available for this batch.")
    else:
        sigs = st.multiselect(
            "Signals", ["main_comp", "tbl_fill", "SREL", "tbl_speed",
                        "ejection", "stiffness", "pre_comp", "fom"],
            default=["main_comp"], format_func=CH._ts_label)
        if sigs:
            st.plotly_chart(CH.trajectory(ts, sigs), use_container_width=True)
        m = st.columns(4)
        with m[0]:
            ui.kpi("Run duration", f"{row.get('ts_run_hours', np.nan):.1f} h",
                   f"{len(ts):,} samples at 10 s")
        with m[1]:
            ui.kpi("Press stops", f"{int(row.get('ts_stop_count', 0))}",
                   f"longest {row.get('ts_longest_stop_s', np.nan) / 60:.0f} min"
                   if pd.notna(row.get("ts_longest_stop_s")) else "")
        with m[2]:
            ui.kpi("Force drift", f"{row.get('ts_main_comp_slope', np.nan):+.3f}",
                   "kN per hour")
        with m[3]:
            tib = row.get("ts_main_comp_time_in_band", np.nan)
            ui.kpi("Force in stable band", f"{tib:.0f}%" if pd.notna(tib) else "—",
                   "±3 robust SD of run centre")
        ui.plain(
            "Grey bands mark periods when the press was stopped. Those samples are "
            "excluded from every derived feature — leaving them in would drag each "
            "average toward zero and invent excursions that never happened.")

with t3:
    st.markdown("#### Incoming material lots used")
    lots = D.load_lots()
    for mat, spec in C.MATERIALS.items():
        lot_col = spec["lot_col"]
        if lot_col not in batch.columns or pd.isna(row.get(lot_col)):
            continue
        lot_no = int(row[lot_col])
        rec = lots[(lots["material"] == mat) & (lots["lot"] == lot_no)]
        attrs = [a for a in spec["attrs"] if a in batch.columns]
        with st.container():
            head = f"**{mat}** · lot {lot_no}"
            if spec["supplier_col"] and pd.notna(row.get(spec["supplier_col"])):
                head += f" · supplier {int(row[spec['supplier_col']])}"
            if len(rec):
                head += f" · used in {int(rec['n_batches'].iat[0])} batches"
            st.markdown(head)
            vals = []
            for a in attrs:
                v = row.get(a)
                z = A.robust_z(peers[a]).loc[row.name] if peers[a].notna().sum() > 5 else np.nan
                vals.append({
                    "Attribute": C.feat_label(a),
                    "Value": f"{v:.4g} {C.feat_unit(a)}" if pd.notna(v) else "not tested",
                    "vs peers": f"{z:+.1f} SD" if np.isfinite(z) else "—",
                })
            st.dataframe(pd.DataFrame(vals), use_container_width=True,
                         hide_index=True)
    if pd.isna(row.get("api_l_impurity")):
        ui.caution(
            "API impurity L was not recorded for this lot. It is missing for about "
            "36% of batches, so it is excluded from modelling by default rather "
            "than imputed — inventing a value would fabricate evidence.")

with t4:
    st.markdown("#### Compression process record")
    recs = []
    for f in C.PROCESS_FEATURES:
        if f not in batch.columns:
            continue
        v = row.get(f)
        z = A.robust_z(peers[f]).loc[row.name] if peers[f].notna().sum() > 5 else np.nan
        recs.append({"Parameter": C.feat_label(f),
                     "Value": f"{v:.4g}" if pd.notna(v) else "—",
                     "Unit": C.feat_unit(f),
                     "vs peers (robust SD)": z})
    pr = pd.DataFrame(recs).sort_values("vs peers (robust SD)",
                                        key=lambda s: s.abs(), ascending=False)
    st.dataframe(
        pr, use_container_width=True, hide_index=True, height=420,
        column_config={"vs peers (robust SD)": st.column_config.NumberColumn(
            format="%+.2f",
            help="Robust z-score against the peer cohort. |z|>3 is unusual.")})
    ui.plain("Sorted by deviation from the peer cohort, so the parameters that "
             "made this run different appear first.")

st.markdown("---")
a, b = st.columns(2)
with a:
    st.page_link("pages/4_Root_cause.py", label="→ Run root-cause analysis on this batch")
with b:
    st.page_link("pages/3_Compare_batches.py", label="→ Compare against golden batches")
