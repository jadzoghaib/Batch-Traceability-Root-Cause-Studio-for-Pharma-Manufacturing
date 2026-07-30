"""
Live operations — the plant as it runs, not as it looks in hindsight.

Every batch here is scored against ONLY the batches that preceded it, using the
control limits that existed at the moment of manufacture. Step the clock forward
and batches arrive, cohorts mature, and limits tighten exactly as they would on
a real line.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from batchlens import analytics as A
from batchlens import config as C
from batchlens import data as D
from batchlens import ui

ui.page("Live operations")
ui.sidebar_brand()

batch = D.load_batches()
scan = D.load_prospective_scan()
queue = D.load_prospective_queue()

months = sorted(pd.to_datetime(batch["start_date"]).dropna().unique())
if "clock" not in st.session_state:
    st.session_state["clock"] = len(months) - 1      # start at "today"

# ---------------------------------------------------------------- clock
st.sidebar.markdown("### Production clock")
c1, c2, c3 = st.sidebar.columns(3)
if c1.button("⏮", help="Back to the first campaign", use_container_width=True):
    st.session_state["clock"] = 0
if c2.button("◀", help="Previous month", use_container_width=True):
    st.session_state["clock"] = max(0, st.session_state["clock"] - 1)
if c3.button("▶", help="Next month", use_container_width=True):
    st.session_state["clock"] = min(len(months) - 1, st.session_state["clock"] + 1)

idx = st.sidebar.select_slider(
    "As of", options=list(range(len(months))),
    value=st.session_state["clock"],
    format_func=lambda i: pd.Timestamp(months[i]).strftime("%b %Y"))
if idx != st.session_state["clock"]:
    st.session_state["clock"] = idx

now = pd.Timestamp(months[st.session_state["clock"]])
ui.data_provenance()

# ---------------------------------------------------------------- slices
dates = pd.to_datetime(batch["start_date"])
released = batch[dates <= now]                 # everything made so far
arriving = batch[dates == now]                 # this month's production
q_rel = queue[queue["batch"].isin(released["batch"])]
q_new = queue[queue["batch"].isin(arriving["batch"])]

ui.masthead(
    "Live operations",
    f"Production clock: <b>{now:%B %Y}</b> · {len(released):,} batches released "
    f"to date · {len(arriving)} this month",
    f"Month {st.session_state['clock'] + 1} of {len(months)}")
st.markdown("")

# ---------------------------------------------------------------- live KPIs
open_inv = int((q_rel["status"] == "Investigate").sum())
new_inv = int((q_new["status"] == "Investigate").sum())
new_watch = int((q_new["status"] == "Watch").sum())
no_base = int((q_rel["status"] == "No baseline").sum())
mature = released.groupby("code").size()
n_mature = int((mature >= 12).sum())

k = st.columns(5)
with k[0]:
    ui.kpi("Released to date", f"{len(released):,}",
           f"of {len(batch):,} total")
with k[1]:
    ui.kpi("Arriving this month", f"{len(arriving)}",
           f"{arriving['code'].nunique()} products" if len(arriving) else "line idle")
with k[2]:
    ui.kpi("New investigations", f"{new_inv}", "flagged on arrival",
           "alert" if new_inv else "ok")
with k[3]:
    ui.kpi("New watch items", f"{new_watch}", "within 2–3 robust SD",
           "watch" if new_watch else "ok")
with k[4]:
    ui.kpi("Products characterised", f"{n_mature}",
           f"of {released['code'].nunique()} run so far", "ok")

st.markdown("")
ui.plain(
    "<b>Nothing here uses future data.</b> Each batch is scored against the "
    "control limits its own product cohort had produced <i>at that moment</i>. "
    "A product needs 12 prior batches before it gets limits at all — until then "
    "the honest status is <b>No baseline</b>, not a limit invented from four "
    "points. Step the clock and watch cohorts mature.")

# ---------------------------------------------------------------- arrivals
st.markdown("## Batch arrivals")
if arriving.empty:
    st.info(f"No batches recorded for {now:%B %Y}.")
else:
    order = {"Investigate": 0, "Watch": 1, "No baseline": 2, "Clear": 3}
    show = q_new.copy()
    show["_o"] = show["status"].map(order).fillna(9)
    show = show.sort_values(["_o", "max_severity"], ascending=[True, False])

    for _, r in show.iterrows():
        if r["status"] == "No baseline":
            col, bg, icon = ui.MUTED, "#F1F5F9", "○"
            detail = (f"cohort still building — "
                      f"{int((released['code'] == r['code']).sum())} batches of "
                      f"{r['product_id']} made so far")
        else:
            col, bg, icon = ui.STATUS_COLORS[r["status"]]
            detail = r["flagged_cqas"] or "all attributes within live control limits"
        st.markdown(
            f"""<div class='bl-card' style='margin-bottom:.5rem;padding:.72rem .95rem'>
              <div style='display:flex;justify-content:space-between;align-items:center'>
                <div>
                  <span style='font-weight:660;font-size:.95rem'>{r['batch_id']}</span>
                  <span class='bl-meta'> · {r['product_id']} · {r['strength_label']}</span>
                </div>
                <span class='bl-pill' style='background:{bg};color:{col}'>
                  {icon} {r['status']}</span>
              </div>
              <div class='bl-meta' style='margin-top:.3rem'>{detail}</div>
            </div>""", unsafe_allow_html=True)

    flagged = show[show["status"].isin(["Investigate", "Watch"])]
    if len(flagged):
        pick = st.selectbox(
            "Open an arriving batch", flagged["batch_id"].tolist(),
            format_func=lambda b: f"{b} — "
                                  f"{flagged.loc[flagged['batch_id'] == b, 'status'].iat[0]}")
        bno = int(flagged.loc[flagged["batch_id"] == pick, "batch"].iat[0])
        st.session_state["focus_batch"] = bno

        det = scan[(scan["batch"] == bno) & (scan["status"] != "Clear")
                   & (scan["status"] != "No baseline")]
        if len(det):
            d = det.copy()
            d["Attribute"] = d["cqa"].map(lambda c: C.CQAS[c]["label"])
            d["Result"] = d.apply(
                lambda r: f"{r['value']:.3g} {C.CQAS[r['cqa']]['unit']}", axis=1)
            d["Live limits"] = d.apply(
                lambda r: f"{r['lcl']:.3g} – {r['ucl']:.3g}", axis=1)
            d["Robust z"] = d["z"].map(lambda v: f"{v:+.1f}")
            d["Built from"] = d["n_history"].map(lambda v: f"{int(v)} prior batches")
            st.dataframe(
                d[["Attribute", "Result", "Live limits", "Robust z",
                   "Built from", "status"]].rename(columns={"status": "Status"}),
                use_container_width=True, hide_index=True)
            ui.plain("<b>Live limits</b> are the control limits this product had "
                     "earned by this point in its history — not the limits derived "
                     "from the full dataset. Early in a product's life they are "
                     "wider, and they tighten as evidence accumulates.")

        a, b_ = st.columns(2)
        with a:
            st.page_link("pages/2_Batch_detail.py", label="→ Batch detail")
        with b_:
            st.page_link("pages/4_Root_cause.py", label="→ Root-cause analysis")

# ---------------------------------------------------------------- rolling view
st.markdown("## Flag rate over the campaign")
hist = queue[queue["batch"].isin(released["batch"])].copy()
hist["month"] = pd.to_datetime(hist["start_date"])
m = (hist.groupby("month")["status"]
     .value_counts().unstack(fill_value=0).reindex(columns=
        ["Investigate", "Watch", "Clear", "No baseline"], fill_value=0)
     .reset_index())

fig = go.Figure()
for name, color in [("Investigate", ui.ALERT), ("Watch", ui.WATCH),
                    ("Clear", ui.ACCENT), ("No baseline", ui.MUTED)]:
    fig.add_trace(go.Bar(x=m["month"], y=m[name], name=name,
                         marker=dict(color=color)))
fig.update_layout(barmode="stack")
fig.update_yaxes(title_text="Batches")
st.plotly_chart(ui.style_fig(fig, 300), use_container_width=True)
ui.plain(
    "Grey is the cold-start period: products that had not yet made enough batches "
    "to earn control limits. It shrinks as the plant accumulates history — which "
    "is exactly the behaviour a real deployment shows in its first months, and "
    "the reason a system like this needs a characterisation phase before it can "
    "carry review-by-exception decisions.")

# ---------------------------------------------------------------- honesty
st.markdown("## Live scoring vs hindsight")
retro = D.load_queue()
cmp = (queue[["batch", "status"]].rename(columns={"status": "live"})
       .merge(retro[["batch", "status"]].rename(columns={"status": "retro"}),
              on="batch"))
cmp = cmp[cmp["batch"].isin(released["batch"])]
diff = int((cmp["live"] != cmp["retro"]).sum())

c1, c2 = st.columns([1, 1.4])
with c1:
    ui.kpi("Judged differently", f"{diff:,}",
           f"{diff / max(len(cmp), 1):.0%} of released batches",
           "watch" if diff else "ok")
    st.markdown("")
    missed = int(((cmp["live"] == "No baseline") & (cmp["retro"] == "Investigate")).sum())
    extra = int(((cmp["live"] == "Investigate") & (cmp["retro"] == "Clear")).sum())
    ui.kpi("Missed while cold-starting", f"{missed}",
           "retrospectively an investigation", "alert" if missed else "ok")
with c2:
    ct = pd.crosstab(cmp["live"], cmp["retro"])
    st.markdown("#### Live status (rows) vs retrospective (columns)")
    st.dataframe(ct, use_container_width=True)

ui.caution(
    f"<b>Why this comparison is in the product.</b> Scoring against the full "
    f"dataset — including batches not yet made — is look-ahead bias. It makes any "
    f"such system look better than it can perform live. On this data it changes "
    f"the verdict for <b>{diff}</b> batches: <b>{extra}</b> would be raised as "
    f"investigations that hindsight calls clear, and <b>{missed}</b> genuine "
    f"investigations are missed entirely because their product had not yet earned "
    f"limits. Any claimed deviation-reduction figure that ignores this is not a "
    f"number you could reproduce in production.")
