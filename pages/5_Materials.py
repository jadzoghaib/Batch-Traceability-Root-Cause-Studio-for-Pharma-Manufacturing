"""Raw-material and supplier drilldown: does an incoming lot shift downstream quality?"""
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

ui.page("Materials")
ui.sidebar_brand()

batch = D.load_batches()
lots = D.load_lots()

mat = st.sidebar.selectbox("Material", list(C.MATERIALS))
spec = C.MATERIALS[mat]
lot_col = spec["lot_col"]
cqa = st.sidebar.selectbox("Quality attribute", D.cqa_options(batch),
                           format_func=lambda c: C.CQAS[c]["label"])
codes = sorted(batch["code"].value_counts()
               .loc[lambda s: s >= C.MIN_COHORT_N].index.tolist())
code = st.sidebar.selectbox(
    "Within product cohort", codes,
    format_func=lambda c: f"P-{c:02d} · {int((batch['code'] == c).sum())} batches")
ui.data_provenance()

ml = lots[lots["material"] == mat]
ui.masthead(f"{mat} lot traceability",
            f"{len(ml)} distinct lots · used across {int(ml['n_batches'].sum())} batches")
st.markdown("")

k = st.columns(4)
with k[0]:
    ui.kpi("Distinct lots", f"{len(ml)}", mat)
with k[1]:
    ui.kpi("Median batches per lot", f"{ml['n_batches'].median():.0f}",
           "how far a lot travels")
with k[2]:
    if spec["supplier_col"]:
        ui.kpi("Suppliers", f"{int(ml['supplier'].nunique())}", "distinct sources")
    else:
        ui.kpi("Products touched", f"{int(ml['n_products'].max())}",
               "widest lot reach")
with k[3]:
    ui.kpi("Attributes tested", f"{len(spec['attrs'])}", "per incoming lot")

ui.plain(
    f"<b>Why cohort matters here too.</b> A material lot is compared only inside "
    f"<b>P-{code:02d}</b>. Lots are not randomly allocated across products, so a "
    f"plant-wide comparison would largely measure which product a lot happened to "
    f"be used in, not the lot itself.")

# ---------------------------------------------------------------- lot effect
st.markdown(f"## Outcome by {mat} lot within P-{code:02d}")
sub = batch[batch["code"] == code]
counts = sub[lot_col].value_counts()
usable = counts[counts >= 3]

if len(usable) < 2:
    st.info(f"Not enough {mat} lots with ≥3 batches inside this cohort to compare.")
else:
    st.plotly_chart(CH.lot_effect(batch, lot_col, cqa, code=code),
                    use_container_width=True)

    groups = [g[cqa].dropna().values for _, g in sub.groupby(lot_col)
              if len(g) >= 3 and g[cqa].notna().sum() >= 3]
    if len(groups) >= 2:
        grand = np.concatenate(groups)
        k_ = len(groups)
        n_ = len(grand)
        ss_b = sum(len(g) * (g.mean() - grand.mean()) ** 2 for g in groups)
        ss_w = sum(((g - g.mean()) ** 2).sum() for g in groups)
        if ss_w > 0 and n_ > k_:
            f = (ss_b / (k_ - 1)) / (ss_w / (n_ - k_))
            eta2 = ss_b / (ss_b + ss_w)
            st.markdown(
                f"<div class='bl-meta'>Across {k_} lots with at least 3 batches: "
                f"lot identity accounts for <b>{eta2:.1%}</b> of the variation in "
                f"{C.CQAS[cqa]['label'].lower()} within this product "
                f"(F≈{f:.2f}).</div>", unsafe_allow_html=True)
            if eta2 > 0.25:
                ui.caution(
                    "A sizeable share of outcome variation lines up with lot "
                    "identity. Worth checking whether those lots also differ on a "
                    "measured attribute below — if they do not, the difference may "
                    "come from something not recorded at goods-in.")

# ---------------------------------------------------------------- attributes
st.markdown("## Incoming attributes vs outcome")
attrs = [a for a in spec["attrs"] if a in batch.columns]
rows = []
for a in attrs:
    s = sub[[a, cqa]].dropna()
    if len(s) < 10 or s[a].nunique() < 3:
        rows.append({"Attribute": C.feat_label(a), "Unit": C.feat_unit(a),
                     "Batches": len(s), "Correlation": np.nan,
                     "Note": "too few distinct values in this cohort"})
        continue
    r = s[a].rank().corr(s[cqa].rank())
    rows.append({"Attribute": C.feat_label(a), "Unit": C.feat_unit(a),
                 "Batches": len(s), "Correlation": r,
                 "Note": C.feat_plain(a)})
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
             column_config={"Correlation": st.column_config.NumberColumn(
                 format="%+.2f",
                 help=f"Spearman correlation with {C.CQAS[cqa]['label']} inside "
                      f"P-{code:02d}")})

pick = st.selectbox("Inspect attribute", attrs,
                    format_func=lambda a: C.feat_label(a))
st.plotly_chart(CH.driver_scatter(sub, pick, cqa,
                                  focus=st.session_state.get("focus_batch")),
                use_container_width=True)

if mat == "API":
    st.markdown("## Supplier view")
    sup = (batch.groupby("api_code")
           .agg(batches=("batch", "size"), lots=("api_batch", "nunique"),
                med=(cqa, "median"), products=("code", "nunique"))
           .reset_index().rename(columns={"api_code": "Supplier"}))
    sup["Supplier"] = sup["Supplier"].map(lambda v: f"Supplier {int(v)}")
    st.dataframe(sup.rename(columns={
        "batches": "Batches", "lots": "Distinct lots",
        "med": f"Median {C.CQAS[cqa]['label']}", "products": "Products supplied"}),
        use_container_width=True, hide_index=True)
    ui.caution(
        "Supplier medians here are <b>not</b> a supplier scorecard. Suppliers "
        "serve different products in different periods, so this table mixes "
        "supplier effects with product mix and time. Use it to generate questions, "
        "not to rank vendors.")

st.markdown("---")
st.markdown(
    "<div class='bl-meta'>Material lot identifiers in the published dataset are "
    "anonymised integers. Supplier identity, price, and goods-in dates are not "
    "included, so genuine supplier scorecarding is out of scope for this demo.</div>",
    unsafe_allow_html=True)
