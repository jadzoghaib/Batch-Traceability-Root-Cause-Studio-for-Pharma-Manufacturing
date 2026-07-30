"""Golden vs poor cohort comparison — what separated the good runs from the bad ones."""
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

ui.page("Compare batches")
ui.sidebar_brand()

batch = D.load_batches()

# ---------------------------------------------------------------- controls
codes = (batch["code"].value_counts()
         .loc[lambda s: s >= C.MIN_COHORT_N].index.tolist())
codes = sorted(codes)
labels = {c: f"P-{c:02d} · {batch.loc[batch['code'] == c, 'strength_label'].iat[0]}"
             f" · {int((batch['code'] == c).sum())} batches" for c in codes}

focus = st.session_state.get("focus_batch")
default_code = int(batch.loc[batch["batch"] == focus, "code"].iat[0]) if focus and \
    focus in set(batch["batch"]) else codes[0]
if default_code not in codes:
    default_code = codes[0]

code = st.sidebar.selectbox("Product cohort", codes,
                            index=codes.index(default_code),
                            format_func=lambda c: labels[c])
cqa = st.sidebar.selectbox("Quality attribute", D.cqa_options(batch),
                           format_func=lambda c: C.CQAS[c]["label"])
show_focus = st.sidebar.checkbox("Overlay selected batch", value=bool(focus))
ui.data_provenance()

peers = batch[batch["code"] == code]
good, poor = A.golden_cohort(peers, cqa)
meta = C.CQAS[cqa]

focus_row = None
if show_focus and focus and focus in set(peers["batch"]):
    focus_row = peers[peers["batch"] == focus].iloc[0]

ui.masthead(
    "Golden vs poor batch comparison",
    f"P-{code:02d} · {len(peers)} batches · comparing best and worst quartile on "
    f"{meta['label'].lower()}")
st.markdown("")

# ---------------------------------------------------------------- KPIs
gm = good[cqa].median() if len(good) else np.nan
pm = poor[cqa].median() if len(poor) else np.nan
k = st.columns(4)
with k[0]:
    ui.kpi("Golden cohort", f"{len(good)}", "best quartile", "ok")
with k[1]:
    ui.kpi("Poor cohort", f"{len(poor)}", "worst quartile", "alert")
with k[2]:
    ui.kpi(f"Golden median", f"{gm:.2f}" if pd.notna(gm) else "—", meta["unit"], "ok")
with k[3]:
    ui.kpi(f"Poor median", f"{pm:.2f}" if pd.notna(pm) else "—", meta["unit"], "alert")

gap = abs(gm - pm) if pd.notna(gm) and pd.notna(pm) else np.nan
st.markdown("")
ui.plain(
    f"<b>What counts as golden.</b> The best-performing quarter of "
    f"P-{code:02d} batches on {meta['label'].lower()}, compared with the worst "
    f"quarter. The gap between them is <b>{gap:.2f} {meta['unit']}</b>. The "
    f"profile below shows which upstream conditions actually differed between "
    f"those two groups.")

if len(good) < 4 or len(poor) < 4:
    st.warning("Cohort too small to form reliable quartiles.")
    st.stop()

# ---------------------------------------------------------------- profile
st.markdown("## Condition profile: what separated them")

res = A.run_rca(peers, cqa, f"product P-{code:02d}", focus_batch=focus)
if res.drivers.empty:
    st.warning("No driver varied enough within this cohort to compare.")
    st.stop()

n_show = st.slider("Conditions to show", 5, 20, 10)
drivers = res.drivers.head(n_show)["driver"].tolist()

st.plotly_chart(CH.good_vs_poor(peers, good, poor, drivers, focus_row),
                use_container_width=True)
ui.plain(
    "Each row is one manufacturing condition, standardised to robust z against "
    "the cohort so that kilonewtons and micrometres can share an axis. "
    "<b>Zero is the cohort median.</b> A wide gap between the teal and red dots "
    "means golden and poor batches genuinely ran under different conditions on "
    "that variable; a narrow gap means it did not distinguish them.")

# ---------------------------------------------------------------- table
st.markdown("## Side-by-side conditions")
rows = []
for d in drivers:
    if d not in peers.columns:
        continue
    g_, p_ = peers.loc[good.index, d], peers.loc[poor.index, d]
    sd = peers[d].std(ddof=0)
    rec = {
        "Condition": C.feat_label(d),
        "Where": C.feat_domain(d),
        "Golden median": g_.median(),
        "Poor median": p_.median(),
        "Gap (robust SD)": (g_.median() - p_.median()) / sd if sd else np.nan,
        "Unit": C.feat_unit(d),
    }
    if focus_row is not None:
        rec["This batch"] = focus_row.get(d)
    rows.append(rec)
tbl = pd.DataFrame(rows)
st.dataframe(
    tbl, use_container_width=True, hide_index=True,
    column_config={
        "Golden median": st.column_config.NumberColumn(format="%.3f"),
        "Poor median": st.column_config.NumberColumn(format="%.3f"),
        "This batch": st.column_config.NumberColumn(format="%.3f"),
        "Gap (robust SD)": st.column_config.NumberColumn(
            format="%+.2f",
            help="Difference between golden and poor medians, in cohort SD units"),
    })

# ---------------------------------------------------------------- evidence
st.markdown("## Evidence for the top separator")
top = res.drivers.iloc[0]
st.plotly_chart(CH.driver_scatter(peers, top["driver"], cqa, focus=focus),
                use_container_width=True)
ui.plain(f"<b>{top['label']}.</b> {C.feat_plain(top['driver'])} "
         f"{A.explain_driver(top, cqa)}")
ui.caution(
    "This is an <b>observational</b> comparison of production data, not a designed "
    "experiment. Golden and poor batches may also differ in ways nobody recorded — "
    "operator, ambient humidity, tooling age. Treat these as prioritised leads for "
    "investigation, not proven causes.")
