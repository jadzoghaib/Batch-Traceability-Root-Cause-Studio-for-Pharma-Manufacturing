"""
BatchLens — Quality-to-Process Traceability & RCA for Pharma Manufacturing.

Entry point / portfolio overview.
Run:  streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from batchlens import config as C
from batchlens import data as D
from batchlens import ui

ui.page("Overview")
ui.sidebar_brand()

if not (C.PROCESSED / "fct_batch.parquet").exists():
    st.error("Processed data not found. Run:  `python -m batchlens.etl`")
    st.stop()

batch = D.load_batches()
queue = D.load_queue()
products = D.load_products()

# ---------------------------------------------------------------- sidebar
st.sidebar.markdown("### Filters")
strengths = ["All"] + sorted(batch["strength_label"].dropna().unique().tolist())
sel_strength = st.sidebar.selectbox("Strength", strengths)

dmin = pd.to_datetime(batch["start_date"]).min().date()
dmax = pd.to_datetime(batch["start_date"]).max().date()
sel_range = st.sidebar.slider("Production period", dmin, dmax, (dmin, dmax),
                              format="MMM YY")

view = batch.copy()
if sel_strength != "All":
    view = view[view["strength_label"] == sel_strength]
view = view[(pd.to_datetime(view["start_date"]).dt.date >= sel_range[0])
            & (pd.to_datetime(view["start_date"]).dt.date <= sel_range[1])]
q = queue[queue["batch"].isin(view["batch"])]

ui.data_provenance()

# ---------------------------------------------------------------- header
ui.masthead(
    "Manufacturing quality overview",
    f"{len(view):,} batches · {view['code'].nunique()} products · "
    f"{sel_range[0]:%b %Y} – {sel_range[1]:%b %Y}",
    "Tablet compression line",
)
st.markdown("")

# ---------------------------------------------------------------- KPI row
n_inv = int((q["status"] == "Investigate").sum())
n_watch = int((q["status"] == "Watch").sum())
n_clear = int((q["status"] == "Clear").sum())
pct_clear = 100 * n_clear / len(q) if len(q) else 0

k = st.columns(5)
with k[0]:
    ui.kpi("Batches in scope", f"{len(view):,}",
           f"{view['code'].nunique()} product codes")
with k[1]:
    ui.kpi("Needs investigation", f"{n_inv}",
           "beyond 3 robust SD of peers", "alert" if n_inv else "ok")
with k[2]:
    ui.kpi("On watch", f"{n_watch}", "2–3 robust SD of peers",
           "watch" if n_watch else "ok")
with k[3]:
    ui.kpi("Clear by exception", f"{pct_clear:.0f}%",
           f"{n_clear:,} batches need no review", "ok")
with k[4]:
    med = view[C.PRIMARY_CQA].median()
    ui.kpi("Median dissolution", f"{med:.1f}%", "across batches in scope")

st.markdown("")
ui.plain(
    f"<b>Review by exception.</b> Every batch is scored against its own peer "
    f"cohort — batches of the same product code — rather than against the whole "
    f"plant. On the current selection that clears <b>{pct_clear:.0f}%</b> of "
    f"batches from manual review and concentrates attention on "
    f"<b>{n_inv + n_watch}</b> that genuinely deviate.")

# ---------------------------------------------------------------- body
left, right = st.columns([1.65, 1])

with left:
    st.markdown("## Quality trend")
    tab1, tab2 = st.tabs(["Over time", "By product"])

    with tab1:
        cqa = st.selectbox("Quality attribute", D.cqa_options(view),
                           format_func=lambda c: C.CQAS[c]["label"],
                           key="ov_cqa")
        m = (view.dropna(subset=[cqa])
             .groupby(pd.to_datetime(view["start_date"]).dt.to_period("M"))
             .agg(med=(cqa, "median"), lo=(cqa, lambda s: s.quantile(0.1)),
                  hi=(cqa, lambda s: s.quantile(0.9)), n=(cqa, "size"))
             .reset_index())
        m["month"] = m["start_date"].dt.to_timestamp()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=m["month"], y=m["hi"], mode="lines",
                                 line=dict(width=0), showlegend=False,
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=m["month"], y=m["lo"], mode="lines",
                                 line=dict(width=0), fill="tonexty",
                                 fillcolor="rgba(14,110,140,.13)",
                                 name="10th–90th percentile", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=m["month"], y=m["med"], mode="lines+markers",
                                 name="Monthly median",
                                 line=dict(color=ui.ACCENT, width=2.2),
                                 marker=dict(size=6),
                                 customdata=m[["n"]],
                                 hovertemplate="%{x|%b %Y}<br>median %{y:.2f}"
                                               "<br>%{customdata[0]} batches<extra></extra>"))
        fig.update_yaxes(title_text=f"{C.CQAS[cqa]['label']} ({C.CQAS[cqa]['unit']})")
        st.plotly_chart(ui.style_fig(fig, 330), use_container_width=True)
        st.markdown(f"<div class='bl-meta'>{C.CQAS[cqa]['plain']}</div>",
                    unsafe_allow_html=True)

    with tab2:
        agg = (view.groupby(["code", "product_id", "strength_label"])
               .agg(n=("batch", "size"),
                    med=(C.PRIMARY_CQA, "median"))
               .reset_index())
        agg = agg.merge(q.groupby("code")["status"]
                        .apply(lambda s: (s != "Clear").mean() * 100)
                        .rename("pct_flagged").reset_index(), on="code", how="left")
        agg["pct_flagged"] = agg["pct_flagged"].fillna(0)
        agg = agg.sort_values("pct_flagged", ascending=True)
        fig = go.Figure(go.Bar(
            x=agg["pct_flagged"], y=agg["product_id"], orientation="h",
            marker=dict(color=[ui.ALERT if v >= 50 else ui.WATCH if v >= 25
                               else ui.ACCENT for v in agg["pct_flagged"]]),
            customdata=agg[["strength_label", "n", "med"]],
            hovertemplate=("<b>%{y}</b> · %{customdata[0]}<br>"
                           "%{customdata[1]} batches<br>"
                           "%{x:.0f}% flagged<extra></extra>")))
        fig.update_xaxes(title_text="% of batches flagged for review")
        st.plotly_chart(ui.style_fig(fig, max(300, 22 * len(agg)), legend=False),
                        use_container_width=True)

with right:
    st.markdown("## Priority queue")
    top = q[q["status"] != "Clear"].head(9)
    if top.empty:
        st.success("No batches outside control limits in this selection.")
    for _, r in top.iterrows():
        col, bg, icon = ui.STATUS_COLORS[r["status"]]
        st.markdown(
            f"""<div class='bl-card' style='margin-bottom:.5rem;padding:.7rem .9rem'>
              <div style='display:flex;justify-content:space-between;align-items:center'>
                <div>
                  <span style='font-weight:660;font-size:.94rem'>{r['batch_id']}</span>
                  <span class='bl-meta'> · {r['product_id']} · {r['strength_label']}</span>
                </div>
                <span class='bl-pill' style='background:{bg};color:{col}'>{icon} {r['status']}</span>
              </div>
              <div class='bl-meta' style='margin-top:.3rem'>{r['flagged_cqas']}</div>
            </div>""", unsafe_allow_html=True)
    st.page_link("pages/1_Review_queue.py", label="Open full review queue →")

st.markdown("---")
st.markdown(
    f"<div class='bl-meta'>Control limits shown throughout this application are "
    f"<b>statistically derived</b> from historical peer-cohort behaviour "
    f"(median ±3 robust SD). They are not registered specification limits — the "
    f"published dataset does not contain them.</div>", unsafe_allow_html=True)
