"""Cached data access layer. All pages read through here."""
from __future__ import annotations

import io
import zipfile

import pandas as pd
import streamlit as st

from . import analytics as A
from . import config as C
from . import etl


@st.cache_data(show_spinner=False, max_entries=8)
def load_batches() -> pd.DataFrame:
    return pd.read_parquet(C.PROCESSED / "fct_batch.parquet")


@st.cache_data(show_spinner=False, max_entries=8)
def load_products() -> pd.DataFrame:
    return pd.read_parquet(C.PROCESSED / "dim_product.parquet")


@st.cache_data(show_spinner=False, max_entries=8)
def load_lots() -> pd.DataFrame:
    return pd.read_parquet(C.PROCESSED / "dim_material_lot.parquet")


def _cached(name: str):
    """
    Read a precomputed scan written by the ETL, or None if it is absent.

    The fallbacks below keep the app working from fct_batch.parquet alone, so a
    stale or missing cache degrades to a one-second recompute rather than an
    error. The cache is an optimisation, never a dependency.
    """
    p = C.PROCESSED / f"fct_{name}.parquet"
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception:
            return None
    return None


@st.cache_data(show_spinner=False, max_entries=8)
def load_exceptions() -> pd.DataFrame:
    c = _cached("exception_scan")
    return c if c is not None else A.exception_scan(load_batches())


@st.cache_data(show_spinner=False, max_entries=8)
def load_queue() -> pd.DataFrame:
    c = _cached("exception_queue")
    return c if c is not None else A.exception_queue(load_batches(), load_exceptions())


@st.cache_data(show_spinner=False, max_entries=8)
def load_prospective_scan() -> pd.DataFrame:
    """Expanding-window scan: every batch judged only on what preceded it."""
    c = _cached("prospective_scan")
    return c if c is not None else A.prospective_exception_scan(load_batches())


@st.cache_data(show_spinner=False, max_entries=8)
def load_prospective_queue() -> pd.DataFrame:
    c = _cached("prospective_queue")
    return (c if c is not None
            else A.prospective_queue(load_batches(), load_prospective_scan()))


@st.cache_data(show_spinner=False, max_entries=4)
def _rca_tables() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    return _cached("rca_drivers"), _cached("rca_meta")


def run_rca_cached(batch_no: int | None, code: int, cqa: str,
                   include_low_coverage: bool = False):
    """
    Serve a precomputed driver ranking, adding only the focus batch's position.

    Falls back to fitting on demand when the cohort is not in the cache (small
    cohorts, or the low-coverage toggle, which changes the driver universe).
    Fitting imports scikit-learn lazily, so the deployed app normally never
    loads it at all.
    """
    b = load_batches()
    peers = b[b["code"] == code]
    label = f"product P-{code:02d}"
    if len(peers) < C.MIN_COHORT_N and batch_no is not None:
        peers, label = A.peer_cohort(b, batch_no)

    if not include_low_coverage:
        drivers, meta = _rca_tables()
        if drivers is not None and meta is not None and not drivers.empty:
            d = drivers[(drivers["code"] == code) & (drivers["cqa"] == cqa)]
            m = meta[(meta["code"] == code) & (meta["cqa"] == cqa)]
            if not d.empty and not m.empty:
                m0 = m.iloc[0]
                d = (d.drop(columns=["code", "cqa"])
                     .reset_index(drop=True))
                d = A.attach_batch_z(d, peers, batch_no)
                warn = [w for w in str(m0["warnings"]).split("||") if w]
                return A.RCAResult(
                    cqa=cqa, cohort_label=str(m0["cohort_label"]),
                    n_peers=int(m0["n_peers"]), drivers=d,
                    model_r2=(None if pd.isna(m0["model_r2"])
                              else float(m0["model_r2"])),
                    model_r2_std=(None if pd.isna(m0["model_r2_std"])
                                  else float(m0["model_r2_std"])),
                    baseline_r2=(None if pd.isna(m0["baseline_r2"])
                                 else float(m0["baseline_r2"])),
                    n_used=int(m0["n_used"]), warnings=warn)

    return A.run_rca(peers, cqa, label, focus_batch=batch_no,
                     include_low_coverage=include_low_coverage)


@st.cache_data(show_spinner=False, max_entries=8)
def pooled_vs_within_cached(cqa: str) -> pd.DataFrame:
    c = _cached("pooled_vs_within")
    if c is not None and "cqa" in c.columns:
        sub = c[c["cqa"] == cqa]
        if not sub.empty:
            return sub.drop(columns=["cqa"]).reset_index(drop=True)
    return A.pooled_vs_within(load_batches(), cqa)


@st.cache_data(show_spinner=True, max_entries=3)
def load_timeseries(code: int, batch_no: int) -> pd.DataFrame:
    """
    Pull one batch's 10-second trajectory straight out of the zip.

    Reading lazily keeps the 346 MB of raw series out of the repo and out of
    memory; only the batch actually being investigated is materialised.
    """
    zpath = C.RAW / "Process.zip"
    if not zpath.exists():
        return pd.DataFrame()
    target = f"Process/{code}.csv"
    with zipfile.ZipFile(zpath) as zf:
        if target not in zf.namelist():
            return pd.DataFrame()
        with zf.open(target) as fh:
            df = pd.read_csv(io.TextIOWrapper(fh, "utf-8"), sep=";")
    df = df[df["batch"] == batch_no].copy()
    if df.empty:
        return df
    df["timestamp"] = etl.parse_ts(df["timestamp"])
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df["elapsed_h"] = (df["timestamp"] - df["timestamp"].iloc[0]
                       ).dt.total_seconds() / 3600
    return df


def cqa_options(batch: pd.DataFrame) -> list[str]:
    return [c for c in C.CQAS if c in batch.columns and batch[c].notna().any()]


def fmt(v, nd: int = 2) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"{v:,.{nd}f}"
