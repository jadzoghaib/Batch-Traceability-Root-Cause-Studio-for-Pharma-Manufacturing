"""
Chart vocabulary for BatchLens.

Rules applied consistently:
  * The batch under investigation is always violet and always drawn last.
  * Peer context is grey and low-contrast; it is background, not content.
  * Control limits are dashed, never solid, so they are visually distinct from
    data and never read as a registered specification.
  * Every chart returns a plotly Figure already passed through ui.style_fig.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import analytics as A
from . import config as C
from . import ui


def control_chart(peers: pd.DataFrame, cqa: str, focus: int | None = None,
                  x: str = "seq_in_product") -> go.Figure:
    """Batch-by-batch run chart against statistically derived control limits."""
    d = peers.dropna(subset=[cqa]).sort_values(x)
    lo, med, hi = A.control_limits(d[cqa])
    meta = C.CQAS[cqa]

    fig = go.Figure()
    fig.add_hrect(y0=lo, y1=hi, fillcolor=ui.ACCENT, opacity=0.05, line_width=0)
    for y, name, dash in [(hi, "Upper control limit", "dash"),
                          (med, "Cohort median", "dot"),
                          (lo, "Lower control limit", "dash")]:
        fig.add_hline(y=y, line=dict(color=ui.MUTED, width=1, dash=dash),
                      annotation_text=name, annotation_position="right",
                      annotation_font=dict(size=10, color=ui.MUTED))

    inside = d[(d[cqa] >= lo) & (d[cqa] <= hi)]
    outside = d[(d[cqa] < lo) | (d[cqa] > hi)]
    fig.add_trace(go.Scatter(
        x=inside[x], y=inside[cqa], mode="lines+markers", name="In control",
        line=dict(color=ui.SLATE, width=1.2),
        marker=dict(size=5, color=ui.SLATE),
        customdata=inside[["batch_id"]],
        hovertemplate="%{customdata[0]}<br>%{y:.2f} " + meta["unit"] + "<extra></extra>"))
    if len(outside):
        fig.add_trace(go.Scatter(
            x=outside[x], y=outside[cqa], mode="markers", name="Outside limits",
            marker=dict(size=9, color=ui.ALERT, symbol="diamond",
                        line=dict(width=1, color="white")),
            customdata=outside[["batch_id"]],
            hovertemplate="%{customdata[0]}<br>%{y:.2f} " + meta["unit"] + "<extra></extra>"))
    if focus is not None and focus in set(d["batch"]):
        f = d[d["batch"] == focus]
        fig.add_trace(go.Scatter(
            x=f[x], y=f[cqa], mode="markers", name="This batch",
            marker=dict(size=15, color=ui.FOCUS, symbol="circle",
                        line=dict(width=2.5, color="white")),
            customdata=f[["batch_id"]],
            hovertemplate="%{customdata[0]}<br>%{y:.2f} " + meta["unit"] + "<extra></extra>"))

    fig.update_yaxes(title_text=f"{meta['label']} ({meta['unit']})")
    fig.update_xaxes(title_text="Batch sequence within product")
    return ui.style_fig(fig, 340)


def cohort_distribution(peers: pd.DataFrame, cqa: str,
                        focus: int | None = None) -> go.Figure:
    """Where this batch sits in its cohort's distribution."""
    d = peers.dropna(subset=[cqa])
    meta = C.CQAS[cqa]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=d[cqa], nbinsx=28, name="Peer batches",
                               marker=dict(color=ui.ACCENT, opacity=0.55,
                                           line=dict(width=0))))
    if focus is not None and focus in set(d["batch"]):
        v = float(d.loc[d["batch"] == focus, cqa].iloc[0])
        fig.add_vline(x=v, line=dict(color=ui.FOCUS, width=2.5),
                      annotation_text="This batch",
                      annotation_font=dict(size=11, color=ui.FOCUS))
    fig.update_xaxes(title_text=f"{meta['label']} ({meta['unit']})")
    fig.update_yaxes(title_text="Batches")
    return ui.style_fig(fig, 260, legend=False)


def driver_ranking(res: pd.DataFrame, top: int = 12) -> go.Figure:
    """
    Horizontal ranking of candidate drivers by strength of association,
    coloured by how many independent evidence tiers agree.
    """
    d = res.head(top).iloc[::-1]
    colors = [ui.ACCENT if t >= 2 else ui.MUTED for t in d["tiers_agree"]]
    fig = go.Figure(go.Bar(
        x=d["corr"], y=d["label"], orientation="h",
        marker=dict(color=colors),
        customdata=np.stack([d["tiers_agree"], d["domain"],
                             d["corr_p"].fillna(1)], axis=-1),
        hovertemplate=("<b>%{y}</b><br>%{customdata[1]}<br>"
                       "correlation r=%{x:+.2f}<br>"
                       "evidence tiers agreeing: %{customdata[0]}/3"
                       "<extra></extra>")))
    fig.add_vline(x=0, line=dict(color=ui.LINE, width=1))
    fig.update_xaxes(title_text="Association with outcome (Spearman r)")
    return ui.style_fig(fig, max(260, 30 * len(d)), legend=False)


def good_vs_poor(peers: pd.DataFrame, good: pd.DataFrame, poor: pd.DataFrame,
                 drivers: list[str], focus_row: pd.Series | None = None) -> go.Figure:
    """
    Golden vs poor cohort profile on standardised axes.

    Standardising to cohort robust-z is what makes a compression force in kN and
    a particle size in µm legitimately comparable on one axis.
    """
    labels, g_vals, p_vals, f_vals = [], [], [], []
    for c in drivers:
        if c not in peers.columns:
            continue
        z = A.robust_z(peers[c])
        labels.append(C.feat_label(c))
        g_vals.append(z.loc[good.index].median() if len(good) else np.nan)
        p_vals.append(z.loc[poor.index].median() if len(poor) else np.nan)
        if focus_row is not None and focus_row.name in z.index:
            f_vals.append(z.loc[focus_row.name])
        else:
            f_vals.append(np.nan)

    fig = go.Figure()
    fig.add_vline(x=0, line=dict(color=ui.LINE, width=1))
    fig.add_trace(go.Scatter(x=p_vals, y=labels, mode="markers", name="Poor quartile",
                             marker=dict(size=11, color=ui.POOR, symbol="circle")))
    fig.add_trace(go.Scatter(x=g_vals, y=labels, mode="markers", name="Golden quartile",
                             marker=dict(size=11, color=ui.GOLD, symbol="circle")))
    for i, lab in enumerate(labels):     # connector emphasises the gap
        fig.add_shape(type="line", x0=p_vals[i], x1=g_vals[i], y0=lab, y1=lab,
                      line=dict(color=ui.LINE, width=2), layer="below")
    if focus_row is not None and np.isfinite(np.array(f_vals, dtype=float)).any():
        fig.add_trace(go.Scatter(x=f_vals, y=labels, mode="markers", name="This batch",
                                 marker=dict(size=13, color=ui.FOCUS, symbol="diamond",
                                             line=dict(width=1.5, color="white"))))
    fig.update_xaxes(title_text="Robust z vs cohort (0 = cohort median)")
    return ui.style_fig(fig, max(300, 34 * len(labels)))


def driver_scatter(peers: pd.DataFrame, driver: str, cqa: str,
                   focus: int | None = None) -> go.Figure:
    """The evidence behind one ranked driver, with a LOWESS-free trend line."""
    d = peers.dropna(subset=[driver, cqa])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d[driver], y=d[cqa], mode="markers", name="Peer batches",
        marker=dict(size=7, color=ui.SLATE, opacity=0.55),
        customdata=d[["batch_id"]],
        hovertemplate="%{customdata[0]}<br>x=%{x:.3f}<br>y=%{y:.2f}<extra></extra>"))
    if len(d) > 8 and d[driver].nunique() > 3:
        z = np.polyfit(d[driver], d[cqa], 1)
        xs = np.linspace(d[driver].min(), d[driver].max(), 50)
        fig.add_trace(go.Scatter(x=xs, y=np.polyval(z, xs), mode="lines",
                                 name="Linear trend",
                                 line=dict(color=ui.ACCENT, width=2, dash="dot")))
    if focus is not None and focus in set(d["batch"]):
        f = d[d["batch"] == focus]
        fig.add_trace(go.Scatter(
            x=f[driver], y=f[cqa], mode="markers", name="This batch",
            marker=dict(size=15, color=ui.FOCUS,
                        line=dict(width=2.5, color="white"))))
    fig.update_xaxes(title_text=f"{C.feat_label(driver)} ({C.feat_unit(driver)})")
    fig.update_yaxes(title_text=C.CQAS[cqa]["label"])
    return ui.style_fig(fig, 330)


def trajectory(ts: pd.DataFrame, signals: list[str],
               band: bool = True) -> go.Figure:
    """The 10-second compression trajectory for one batch."""
    palette = [ui.ACCENT, ui.WATCH, ui.GOLD, ui.FOCUS, ui.SLATE]
    fig = go.Figure()
    for i, sig in enumerate(signals):
        if sig not in ts.columns:
            continue
        v = ts[sig].astype(float)
        fig.add_trace(go.Scatter(
            x=ts["elapsed_h"], y=v, mode="lines", name=_ts_label(sig),
            line=dict(color=palette[i % len(palette)], width=1.3)))
        if band and len(signals) == 1:
            run = v[ts["tbl_speed"] > 0]
            if len(run) > 20:
                med = run.median()
                mad = (run - med).abs().median() * 1.4826
                if mad and np.isfinite(mad):
                    fig.add_hrect(y0=med - 3 * mad, y1=med + 3 * mad,
                                  fillcolor=ui.ACCENT, opacity=0.07, line_width=0,
                                  annotation_text="stable band (±3 robust SD)",
                                  annotation_font=dict(size=10, color=ui.MUTED))
    # shade press stoppages
    if "tbl_speed" in ts.columns:
        stopped = ts["tbl_speed"] <= 0
        if stopped.any():
            blocks = (stopped != stopped.shift()).cumsum()[stopped]
            for _, idx in ts[stopped].groupby(blocks).groups.items():
                seg = ts.loc[idx]
                if len(seg) >= 6:      # ignore single-sample blips
                    fig.add_vrect(x0=seg["elapsed_h"].iloc[0],
                                  x1=seg["elapsed_h"].iloc[-1],
                                  fillcolor=ui.MUTED, opacity=0.16, line_width=0)
    fig.update_xaxes(title_text="Hours into run")
    return ui.style_fig(fig, 300)


_TS_LABELS = {
    "main_comp": "Main compression force (kN)", "pre_comp": "Pre-compression (kN)",
    "tbl_fill": "Die fill depth (mm)", "SREL": "In-process weight RSD (%)",
    "tbl_speed": "Press speed (tablets/h)", "ejection": "Ejection force (N)",
    "stiffness": "Powder stiffness (N/mm)", "fom": "Force of main (%)",
}


def _ts_label(sig: str) -> str:
    return _TS_LABELS.get(sig, sig)


def pooled_vs_within_chart(pw: pd.DataFrame, top: int = 14) -> go.Figure:
    """
    The methodology chart: naive pooled correlation vs peer-cohort correlation.
    Sign flips are the whole argument for cohort-based analysis.
    """
    d = pw.head(top).iloc[::-1]
    fig = go.Figure()
    for i, row in enumerate(d.itertuples()):
        flip = row.sign_flip
        fig.add_shape(type="line", x0=row.pooled_r, x1=row.within_r,
                      y0=row.label, y1=row.label,
                      line=dict(color=ui.ALERT if flip else ui.LINE,
                                width=2.5 if flip else 2), layer="below")
    fig.add_trace(go.Scatter(
        x=d["pooled_r"], y=d["label"], mode="markers", name="Pooled (all products)",
        marker=dict(size=11, color=ui.MUTED, symbol="x", line=dict(width=2))))
    fig.add_trace(go.Scatter(
        x=d["within_r"], y=d["label"], mode="markers", name="Within peer cohort",
        marker=dict(size=11, color=ui.ACCENT)))
    fig.add_vline(x=0, line=dict(color=ui.SLATE, width=1))
    fig.update_xaxes(title_text="Correlation with outcome (Spearman r)")
    return ui.style_fig(fig, max(340, 32 * len(d)))


def lot_effect(batch: pd.DataFrame, lot_col: str, cqa: str,
               code: int | None = None, min_n: int = 3) -> go.Figure:
    """Per-lot outcome distribution inside one product cohort."""
    d = batch if code is None else batch[batch["code"] == code]
    d = d.dropna(subset=[cqa, lot_col])
    counts = d[lot_col].value_counts()
    keep = counts[counts >= min_n].index
    d = d[d[lot_col].isin(keep)]
    if d.empty:
        return ui.style_fig(go.Figure(), 240, legend=False)
    order = d.groupby(lot_col)[cqa].median().sort_values().index
    fig = go.Figure()
    for lot in order:
        g = d[d[lot_col] == lot]
        fig.add_trace(go.Box(y=g[cqa], name=str(int(lot)), boxpoints="all",
                             jitter=0.4, pointpos=0, marker=dict(size=4),
                             line=dict(color=ui.ACCENT, width=1.4),
                             fillcolor=ui.ACCENT_SOFT))
    fig.update_xaxes(title_text=f"{lot_col.replace('_batch','').upper()} lot")
    fig.update_yaxes(title_text=C.CQAS[cqa]["label"])
    return ui.style_fig(fig, 300, legend=False)
