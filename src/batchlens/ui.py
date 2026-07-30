"""
BatchLens design system.

Register: regulated-industry operations console. Calm, dense, legible under
fluorescent light at 3pm on a shop floor. The visual job is to make deviation
obvious and everything else quiet.

Palette rationale
  Ink/slate neutrals carry all structure. Colour is reserved for *state* only,
  so a coloured pixel always means something. Status colours are the standard
  operational vocabulary (amber = attention, red = act) and are distinguishable
  for the most common colour-vision deficiencies because each also carries a
  text label and an icon - colour is never the sole channel.
"""
from __future__ import annotations

import streamlit as st

# --------------------------------------------------------------------------
INK = "#0F172A"        # primary text
SLATE = "#475569"      # secondary text
MUTED = "#94A3B8"      # tertiary / axis
LINE = "#E2E8F0"       # hairlines
SURFACE = "#FFFFFF"
CANVAS = "#F6F8FA"     # page background
ACCENT = "#0E6E8C"     # brand teal - links, active nav, primary series
ACCENT_SOFT = "#E5F1F5"

OK = "#1B7F5A"
OK_SOFT = "#E4F3EC"
WATCH = "#B4690E"
WATCH_SOFT = "#FDF2E2"
ALERT = "#B42318"
ALERT_SOFT = "#FDECEA"

GOLD = "#0E7490"       # golden cohort series
POOR = "#B42318"       # poor cohort series
FOCUS = "#7C3AED"      # the batch under investigation

STATUS_COLORS = {
    "Investigate": (ALERT, ALERT_SOFT, "▲"),
    "Watch": (WATCH, WATCH_SOFT, "◆"),
    "Clear": (OK, OK_SOFT, "●"),
}

PLOT_FONT = dict(family="Inter, -apple-system, Segoe UI, sans-serif",
                 size=12, color=SLATE)


def page(title: str, icon: str = "◧") -> None:
    st.set_page_config(page_title=f"{title} · BatchLens", page_icon="◧",
                       layout="wide", initial_sidebar_state="expanded")
    inject_css()


def inject_css() -> None:
    st.markdown(f"""<style>
    .stApp {{ background: {CANVAS}; }}
    html, body, [class*="css"] {{
        font-family: Inter, -apple-system, "Segoe UI", Roboto, sans-serif;
        color: {INK};
    }}
    /* tighten Streamlit's very generous default rhythm */
    .block-container {{ padding-top: 2.1rem; padding-bottom: 3rem; max-width: 1500px; }}
    header[data-testid="stHeader"] {{ background: transparent; height: 0; }}
    #MainMenu, footer {{ visibility: hidden; }}

    h1 {{ font-size: 1.6rem !important; font-weight: 650 !important;
         letter-spacing: -.02em; color: {INK}; margin-bottom: .1rem !important; }}
    h2 {{ font-size: 1.12rem !important; font-weight: 640 !important;
         letter-spacing: -.01em; color: {INK};
         margin-top: 1.6rem !important; margin-bottom: .5rem !important; }}
    h3 {{ font-size: .94rem !important; font-weight: 640 !important; color: {SLATE};
         text-transform: uppercase; letter-spacing: .06em; }}

    section[data-testid="stSidebar"] {{
        background: {SURFACE}; border-right: 1px solid {LINE};
    }}
    section[data-testid="stSidebar"] .block-container {{ padding-top: 1.2rem; }}

    /* cards */
    .bl-card {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 10px;
        padding: 1rem 1.15rem; box-shadow: 0 1px 2px rgba(15,23,42,.04); }}
    .bl-card h4 {{ margin: 0 0 .55rem 0; font-size: .74rem; font-weight: 650;
        text-transform: uppercase; letter-spacing: .07em; color: {MUTED}; }}

    /* KPI */
    .bl-kpi {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 10px;
        padding: .85rem 1rem; height: 100%; }}
    .bl-kpi .lab {{ font-size: .72rem; text-transform: uppercase; letter-spacing: .07em;
        color: {MUTED}; font-weight: 600; }}
    .bl-kpi .val {{ font-size: 1.72rem; font-weight: 660; color: {INK};
        letter-spacing: -.025em; line-height: 1.15; font-variant-numeric: tabular-nums; }}
    .bl-kpi .sub {{ font-size: .76rem; color: {SLATE}; }}

    /* status pill */
    .bl-pill {{ display: inline-flex; align-items: center; gap: .34rem;
        padding: .16rem .58rem; border-radius: 999px; font-size: .73rem;
        font-weight: 640; letter-spacing: .01em; }}

    /* evidence tier chip */
    .bl-tier {{ display:inline-block; padding:.09rem .42rem; border-radius:4px;
        font-size:.67rem; font-weight:650; letter-spacing:.04em;
        background:{ACCENT_SOFT}; color:{ACCENT}; margin-right:.25rem; }}

    /* plain-English explainer under an analytic */
    .bl-plain {{ background: {ACCENT_SOFT}; border-left: 3px solid {ACCENT};
        padding: .6rem .8rem; border-radius: 0 6px 6px 0; font-size: .84rem;
        color: {INK}; line-height: 1.5; }}
    .bl-caution {{ background: {WATCH_SOFT}; border-left: 3px solid {WATCH};
        padding: .6rem .8rem; border-radius: 0 6px 6px 0; font-size: .82rem;
        color: #713F12; line-height: 1.5; }}

    .bl-meta {{ font-size: .78rem; color: {MUTED}; }}
    hr {{ border-color: {LINE}; margin: 1.1rem 0; }}

    /* dataframe */
    [data-testid="stDataFrame"] {{ border: 1px solid {LINE}; border-radius: 8px; }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 1.4rem; border-bottom: 1px solid {LINE}; }}
    .stTabs [data-baseweb="tab"] {{ padding: .45rem 0; font-weight: 600;
        font-size: .87rem; color: {MUTED}; }}
    .stTabs [aria-selected="true"] {{ color: {ACCENT} !important; }}

    div[data-testid="stMetricValue"] {{ font-size: 1.5rem; font-weight: 650; }}
    </style>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# components
# --------------------------------------------------------------------------
def masthead(title: str, subtitle: str = "", right: str = "") -> None:
    c1, c2 = st.columns([4, 1.5])
    with c1:
        st.markdown(f"# {title}")
        if subtitle:
            st.markdown(f"<div class='bl-meta'>{subtitle}</div>",
                        unsafe_allow_html=True)
    if right:
        with c2:
            st.markdown(f"<div style='text-align:right;padding-top:.6rem' "
                        f"class='bl-meta'>{right}</div>", unsafe_allow_html=True)


def pill(status: str) -> str:
    col, bg, icon = STATUS_COLORS.get(status, (SLATE, "#F1F5F9", "●"))
    return (f"<span class='bl-pill' style='background:{bg};color:{col}'>"
            f"{icon} {status}</span>")


def kpi(label: str, value: str, sub: str = "", tone: str | None = None) -> None:
    color = {"ok": OK, "watch": WATCH, "alert": ALERT}.get(tone or "", INK)
    st.markdown(
        f"<div class='bl-kpi'><div class='lab'>{label}</div>"
        f"<div class='val' style='color:{color}'>{value}</div>"
        f"<div class='sub'>{sub}</div></div>", unsafe_allow_html=True)


def plain(text: str) -> None:
    st.markdown(f"<div class='bl-plain'>{text}</div>", unsafe_allow_html=True)


def caution(text: str) -> None:
    st.markdown(f"<div class='bl-caution'>{text}</div>", unsafe_allow_html=True)


def card_open(title: str = "") -> None:
    st.markdown(f"<div class='bl-card'>" +
                (f"<h4>{title}</h4>" if title else ""), unsafe_allow_html=True)


def card_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def style_fig(fig, height: int = 320, legend: bool = True):
    """One place that makes every chart look like it belongs to the same product."""
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=28, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=PLOT_FONT,
        hoverlabel=dict(bgcolor="white", font_size=12, bordercolor=LINE),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font=dict(size=11)),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=LINE,
                     ticks="outside", tickcolor=LINE, tickfont=dict(size=11))
    fig.update_yaxes(showgrid=True, gridcolor=LINE, zeroline=False,
                     linecolor="rgba(0,0,0,0)", tickfont=dict(size=11))
    return fig


def sidebar_brand() -> None:
    st.sidebar.markdown(
        f"""<div style='display:flex;align-items:center;gap:.55rem;
             padding:.1rem 0 .9rem 0;border-bottom:1px solid {LINE};margin-bottom:1rem'>
          <div style='width:26px;height:26px;border-radius:6px;background:{ACCENT};
               color:white;display:flex;align-items:center;justify-content:center;
               font-weight:700;font-size:.85rem'>B</div>
          <div>
            <div style='font-weight:680;font-size:.95rem;letter-spacing:-.01em'>BatchLens</div>
            <div style='font-size:.68rem;color:{MUTED};letter-spacing:.04em'>
              QUALITY-TO-PROCESS RCA</div>
          </div>
        </div>""", unsafe_allow_html=True)


def data_provenance() -> None:
    st.sidebar.markdown(
        f"""<div style='margin-top:1.2rem;padding-top:.8rem;border-top:1px solid {LINE};
             font-size:.7rem;color:{MUTED};line-height:1.5'>
          <b style='color:{SLATE}'>Data source</b><br>
          Žagar &amp; Mihelič (2022), <i>Scientific Data</i>.<br>
          1,005 real production batches.<br>
          CC-BY 4.0 · figshare 10.6084/m9.figshare.c.5645578<br><br>
          <span style='color:{WATCH}'>Independent demo. Not affiliated with,
          endorsed by, or derived from any commercial vendor.</span>
        </div>""", unsafe_allow_html=True)
