"""
ETL: raw figshare CSVs -> tidy Parquet star schema.

Entities produced (data/processed/):
    dim_product.parquet        one row per product code
    fct_batch.parquet          one row per batch: context + CQAs + RM attrs + process feats
    dim_material_lot.parquet   one row per (material, lot): attributes + usage stats
    fct_timeseries_feat.parquet one row per batch: features derived from the 10 s series

Cleaning decisions (all measured, see notebooks/01_profiling):
  * Missing numerics arrive as whitespace-padded strings -> coerce, do not fillna here.
  * Laboratory is authoritative for quality; Process.csv's duplicated quality columns
    are dropped (they carry 18 extra nulls).
  * `start` uses Slovenian month abbreviations -> real timestamps.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _num(s: pd.Series) -> pd.Series:
    """Coerce a column to float, treating whitespace-only cells as missing."""
    if s.dtype.kind in "if":
        return s.astype(float)
    return pd.to_numeric(s.astype(str).str.strip().replace({"": None}), errors="coerce")


def parse_ts(s: pd.Series) -> pd.Series:
    """
    The time-series files ship TWO timestamp formats and pandas' `format="mixed"`
    silently yields NaT for the second one, which quietly drops four of the
    largest products (589 batches) from the feature table:

        ISO       2019-01-17 04:09:38
        Compact   07052019 20:14        -> %d%m%Y %H:%M  (day-first; 22112018
                                           can only be a day, never a month)

    Parse each shape explicitly and only then combine.
    """
    s = s.astype(str).str.strip()
    out = pd.to_datetime(s, errors="coerce", format="ISO8601")
    todo = out.isna()
    if todo.any():
        out.loc[todo] = pd.to_datetime(s[todo], errors="coerce",
                                       format="%d%m%Y %H:%M")
    todo = out.isna()
    if todo.any():          # last resort for any remaining stragglers
        out.loc[todo] = pd.to_datetime(s[todo], errors="coerce",
                                       dayfirst=True, format="mixed")
    return out


def _parse_start(s: pd.Series) -> pd.Series:
    """'nov.18' -> Timestamp('2018-11-01'). Slovenian month abbreviations."""
    out = []
    for v in s.astype(str):
        v = v.strip().lower().rstrip(".")
        mon, _, yr = v.partition(".")
        m = C.SL_MONTHS.get(mon[:3])
        if m and yr.isdigit():
            out.append(pd.Timestamp(year=2000 + int(yr), month=m, day=1))
        else:
            out.append(pd.NaT)
    return pd.Series(out, index=s.index)


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------
def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lab = pd.read_csv(C.RAW / "Laboratory.csv", sep=";")
    proc = pd.read_csv(C.RAW / "Process.csv", sep=";")
    norm = pd.read_csv(C.RAW / "Normalization.csv", sep=";")
    return lab, proc, norm


def build_batch_table(lab: pd.DataFrame, proc: pd.DataFrame,
                      norm: pd.DataFrame) -> pd.DataFrame:
    # Laboratory wins on quality: drop Process' duplicated (and nullier) copies.
    dupe_quality = ["Drug release average (%)", "Drug release min (%)",
                    "Residual solvent", "Total impurities", "Impurity O", "Impurity L"]
    proc = proc.drop(columns=[c for c in dupe_quality if c in proc.columns])
    proc = proc.drop(columns=[c for c in ["code"] if c in proc.columns])

    df = lab.merge(proc, on="batch", how="inner", validate="one_to_one")

    # numeric coercion for every measured column
    for col in C.RM_ATTRS + C.PROCESS_FEATURES + list(C.CQAS):
        if col in df.columns:
            df[col] = _num(df[col])

    df["start_date"] = _parse_start(df["start"])
    df["weekend_flag"] = df["weekend"].astype(str).str.strip().str.lower().eq("yes")

    norm = norm.rename(columns={
        "Product code": "code", "Batch Size (tablets)": "planned_tablets",
        "Normalisation factor": "norm_factor"})
    df = df.merge(norm, on="code", how="left")

    # readable identifiers for the UI
    df["batch_id"] = "B-" + df["batch"].astype(int).astype(str).str.zfill(4)
    df["product_id"] = "P-" + df["code"].astype(int).astype(str).str.zfill(2)
    df["strength_label"] = df["strength"].replace({"5MG": "5 mg", "10M": "10 mg",
                                                   "20M": "20 mg", "40M": "40 mg"})
    # waste normalised by batch size makes cross-format comparison meaningful
    df["waste_rate_pct"] = 100 * df["total_waste"] / df["planned_tablets"]
    df["startup_waste_rate_pct"] = 100 * df["startup_waste"] / df["planned_tablets"]

    df = df.sort_values("batch").reset_index(drop=True)
    # sequence within product = "how many of this product have we made before"
    df["seq_in_product"] = df.groupby("code").cumcount() + 1
    return df


def build_product_dim(batch: pd.DataFrame) -> pd.DataFrame:
    g = batch.groupby("code")
    out = pd.DataFrame({
        "code": g.size().index,
        "product_id": g["product_id"].first(),
        "strength": g["strength_label"].first(),
        "planned_tablets": g["planned_tablets"].first(),
        "n_batches": g.size().values,
        "first_batch": g["start_date"].min().values,
        "last_batch": g["start_date"].max().values,
    }).reset_index(drop=True)
    for cqa in C.CQAS:
        if cqa in batch.columns:
            out[f"{cqa}_median"] = g[cqa].median().values
    return out


def build_material_lots(batch: pd.DataFrame) -> pd.DataFrame:
    """One row per (material, lot) with its attributes and how it performed."""
    frames = []
    for mat, spec in C.MATERIALS.items():
        lot_col = spec["lot_col"]
        if lot_col not in batch.columns:
            continue
        attrs = [a for a in spec["attrs"] if a in batch.columns]
        g = batch.groupby(lot_col)
        rec = pd.DataFrame({
            "material": mat,
            "lot": g.size().index,
            "n_batches": g.size().values,
            "first_used": g["start_date"].min().values,
            "last_used": g["start_date"].max().values,
            "n_products": g["code"].nunique().values,
        })
        if spec["supplier_col"] and spec["supplier_col"] in batch.columns:
            rec["supplier"] = g[spec["supplier_col"]].agg(
                lambda s: s.mode().iat[0] if len(s.mode()) else np.nan).values
        else:
            rec["supplier"] = np.nan
        # lot attributes are constant per lot by construction; take the median defensively
        for a in attrs:
            rec[a] = g[a].median().values
        frames.append(rec)
    lots = pd.concat(frames, ignore_index=True)
    lots["lot_id"] = lots["material"].str.upper().str[:3] + "-" + \
        lots["lot"].astype(int).astype(str).str.zfill(3)
    return lots


# --------------------------------------------------------------------------
# time-series feature engineering
# --------------------------------------------------------------------------
TS_NUMERIC = ["tbl_speed", "fom", "main_comp", "tbl_fill", "SREL",
              "pre_comp", "cyl_main", "cyl_pre", "stiffness", "ejection"]


def _batch_ts_features(g: pd.DataFrame) -> dict:
    """
    Features that describe the *shape* of a run, not just its average.

    Running state is isolated first: rows where the press is stopped
    (tbl_speed == 0) would otherwise drag every mean toward zero and invent
    excursions that never happened.
    """
    out: dict[str, float] = {}
    g = g.sort_values("timestamp")
    t = g["timestamp"]
    dur_h = (t.iloc[-1] - t.iloc[0]).total_seconds() / 3600 if len(g) > 1 else np.nan
    out["ts_run_hours"] = dur_h
    out["ts_n_samples"] = float(len(g))

    running = g[g["tbl_speed"] > 0]
    out["ts_running_share"] = len(running) / len(g) if len(g) else np.nan

    # press stops: contiguous blocks of zero speed
    stopped = (g["tbl_speed"] <= 0).astype(int)
    starts = ((stopped.diff() == 1)).sum()
    out["ts_stop_count"] = float(starts)

    if len(running) < 10:
        return out

    hrs = (running["timestamp"] - running["timestamp"].iloc[0]).dt.total_seconds() / 3600

    for col, key in [("main_comp", "main_comp"), ("tbl_fill", "fill")]:
        v = running[col].astype(float)
        if v.notna().sum() < 10 or v.std(ddof=0) == 0:
            continue
        # linear drift across the run
        try:
            slope = np.polyfit(hrs, v, 1)[0]
        except Exception:
            slope = np.nan
        out[f"ts_{key}_slope"] = float(slope)
        out[f"ts_{key}_cv"] = float(v.std(ddof=0) / v.mean()) if v.mean() else np.nan
        # excursions: leaving a +/-3 robust-sigma band around the run's own centre
        med = v.median()
        mad = (v - med).abs().median() * 1.4826
        if mad and np.isfinite(mad):
            outside = (v - med).abs() > 3 * mad
            out[f"ts_{key}_excursions"] = float(
                (outside.astype(int).diff() == 1).sum())
            out[f"ts_{key}_time_in_band"] = float(100 * (~outside).mean())

    # in-process weight variability spikes
    srel = running["SREL"].astype(float)
    if srel.notna().sum() > 10:
        out["ts_srel_mean"] = float(srel.mean())
        out["ts_srel_p95"] = float(srel.quantile(0.95))
        thr = srel.median() + 3 * ((srel - srel.median()).abs().median() * 1.4826 or 1)
        out["ts_srel_excursions"] = float(((srel > thr).astype(int).diff() == 1).sum())

    # recovery after the longest stop: how long until force is back in band
    if starts:
        blocks = (stopped.diff() != 0).cumsum()
        stop_blocks = g[stopped == 1].groupby(blocks[stopped == 1]).size()
        if len(stop_blocks):
            out["ts_longest_stop_s"] = float(stop_blocks.max() * 10)
    return out


def build_timeseries_features(codes: list[int] | None = None,
                              progress=lambda *_: None) -> pd.DataFrame:
    """
    Stream the per-product time-series CSVs (346 MB unzipped) and reduce each
    batch to a feature row. Reads from the zip directly so the repo never has to
    hold the expanded files.
    """
    zip_path = C.RAW / "Process.zip"
    rows: list[dict] = []
    if not zip_path.exists():
        return pd.DataFrame()

    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        for i, name in enumerate(sorted(names)):
            code = Path(name).stem
            if not code.isdigit():
                continue
            if codes and int(code) not in codes:
                continue
            progress(i, len(names), code)
            with zf.open(name) as fh:
                df = pd.read_csv(io.TextIOWrapper(fh, "utf-8"), sep=";")
            df["timestamp"] = parse_ts(df["timestamp"])
            df = df.dropna(subset=["timestamp"])
            for col in TS_NUMERIC:
                if col in df.columns:
                    df[col] = _num(df[col])
            for batch_no, g in df.groupby("batch"):
                feats = _batch_ts_features(g)
                feats["batch"] = int(batch_no)
                feats["campaign"] = int(g["campaign"].iloc[0]) if "campaign" in g else np.nan
                rows.append(feats)
    return pd.DataFrame(rows)


def build_analytics_cache(batch: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Precompute the scans the app would otherwise run on first page load.

    These are cheap (~1 s total) but deterministic: they depend only on
    fct_batch, so computing them once at build time makes every page render
    immediately on a cold container and turns the scan into an inspectable
    artefact rather than a hidden runtime step.

    Imported locally to keep the module import graph one-directional
    (analytics never imports etl).
    """
    from . import analytics as A

    out: dict[str, pd.DataFrame] = {}
    out["exception_scan"] = A.exception_scan(batch)
    out["exception_queue"] = A.exception_queue(batch, out["exception_scan"])
    out["prospective_scan"] = A.prospective_exception_scan(batch)
    out["prospective_queue"] = A.prospective_queue(batch, out["prospective_scan"])

    # pooled-vs-within evidence for every attribute the Method page offers
    frames = []
    for cqa in C.CQAS:
        if cqa not in batch.columns:
            continue
        pw = A.pooled_vs_within(batch, cqa)
        if not pw.empty:
            pw = pw.copy()
            pw.insert(0, "cqa", cqa)
            frames.append(pw)
    out["pooled_vs_within"] = (pd.concat(frames, ignore_index=True) if frames
                               else pd.DataFrame())
    return out


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------
def run(with_timeseries: bool = True, progress=lambda *_: None) -> dict[str, pd.DataFrame]:
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    lab, proc, norm = load_raw()
    batch = build_batch_table(lab, proc, norm)
    product = build_product_dim(batch)
    lots = build_material_lots(batch)

    if with_timeseries:
        tsf = build_timeseries_features(progress=progress)
        if not tsf.empty:
            batch = batch.merge(tsf, on="batch", how="left")
            tsf.to_parquet(C.PROCESSED / "fct_timeseries_feat.parquet", index=False)

    batch.to_parquet(C.PROCESSED / "fct_batch.parquet", index=False)
    product.to_parquet(C.PROCESSED / "dim_product.parquet", index=False)
    lots.to_parquet(C.PROCESSED / "dim_material_lot.parquet", index=False)

    cache = build_analytics_cache(batch)
    for name, df in cache.items():
        if not df.empty:
            df.to_parquet(C.PROCESSED / f"fct_{name}.parquet", index=False)

    return {"batch": batch, "product": product, "lots": lots, **cache}


if __name__ == "__main__":
    def _p(i, n, code):
        print(f"  [{i + 1}/{n}] time series product {code}", flush=True)

    print("Building Batch Investigation Console star schema...")
    out = run(with_timeseries=True, progress=_p)
    for k, v in out.items():
        print(f"  {k:8s} {v.shape}")
    print(f"Written to {C.PROCESSED}")
