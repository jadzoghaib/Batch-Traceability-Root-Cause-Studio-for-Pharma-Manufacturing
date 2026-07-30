"""Cached data access layer. All pages read through here."""
from __future__ import annotations

import io
import zipfile

import pandas as pd
import streamlit as st

from . import analytics as A
from . import config as C
from . import etl


@st.cache_data(show_spinner=False)
def load_batches() -> pd.DataFrame:
    return pd.read_parquet(C.PROCESSED / "fct_batch.parquet")


@st.cache_data(show_spinner=False)
def load_products() -> pd.DataFrame:
    return pd.read_parquet(C.PROCESSED / "dim_product.parquet")


@st.cache_data(show_spinner=False)
def load_lots() -> pd.DataFrame:
    return pd.read_parquet(C.PROCESSED / "dim_material_lot.parquet")


@st.cache_data(show_spinner=False)
def load_exceptions() -> pd.DataFrame:
    return A.exception_scan(load_batches())


@st.cache_data(show_spinner=False)
def load_queue() -> pd.DataFrame:
    return A.exception_queue(load_batches(), load_exceptions())


@st.cache_data(show_spinner=False)
def load_prospective_scan() -> pd.DataFrame:
    """Expanding-window scan: every batch judged only on what preceded it."""
    return A.prospective_exception_scan(load_batches())


@st.cache_data(show_spinner=False)
def load_prospective_queue() -> pd.DataFrame:
    return A.prospective_queue(load_batches(), load_prospective_scan())


@st.cache_data(show_spinner=False)
def run_rca_cached(batch_no: int | None, code: int, cqa: str,
                   include_low_coverage: bool = False):
    b = load_batches()
    peers = b[b["code"] == code]
    label = f"product P-{code:02d}"
    if len(peers) < C.MIN_COHORT_N and batch_no is not None:
        peers, label = A.peer_cohort(b, batch_no)
    return A.run_rca(peers, cqa, label, focus_batch=batch_no,
                     include_low_coverage=include_low_coverage)


@st.cache_data(show_spinner=False)
def pooled_vs_within_cached(cqa: str) -> pd.DataFrame:
    return A.pooled_vs_within(load_batches(), cqa)


@st.cache_data(show_spinner=True, max_entries=8)
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
