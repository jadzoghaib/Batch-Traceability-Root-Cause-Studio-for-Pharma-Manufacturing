"""
The analytical core of Batch Investigation Console.

Everything here obeys one rule, which is the product's central design claim:

    A batch is only ever compared against its peer cohort (same product code).

Justification is empirical, not stylistic. Measured on this dataset:
  * product identity alone explains 85% of hardness variance and 82% of impurities;
  * 22 of 44 candidate drivers REVERSE their correlation sign between pooled and
    within-product analysis.
So a pooled "top correlations" chart - the default in most BI tools - does not
merely add noise, it can point an investigator in the opposite direction.

Evidence is deliberately reported in three separate tiers, never merged into one
score, because they support different strengths of claim:
    Tier 1 DESCRIPTIVE  - this batch differs from its peers on X   (fact)
    Tier 2 ASSOCIATION  - across peers, X tracks the outcome       (correlation)
    Tier 3 MODEL        - X carries predictive weight jointly      (multivariate)
A driver is only promoted to "prioritised" when independent tiers agree.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score

from . import config as C

# --------------------------------------------------------------------------
# robust statistics
# --------------------------------------------------------------------------
def robust_z(s: pd.Series) -> pd.Series:
    """Median/MAD z-score: resistant to the very outliers we are hunting."""
    s = s.astype(float)
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    if not mad or not np.isfinite(mad):
        sd = s.std(ddof=0)
        if not sd or not np.isfinite(sd):
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - med) / sd
    return (s - med) / mad


def control_limits(s: pd.Series, k: float = 3.0) -> tuple[float, float, float]:
    """SPC-style limits derived from history. NOT registered specifications."""
    s = s.dropna().astype(float)
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    if not mad or not np.isfinite(mad):
        mad = s.std(ddof=0)
    return med - k * mad, med, med + k * mad


# --------------------------------------------------------------------------
# cohorts
# --------------------------------------------------------------------------
def peer_cohort(batch: pd.DataFrame, batch_no: int) -> tuple[pd.DataFrame, str]:
    """Structurally comparable batches: same product code, else same strength."""
    row = batch.loc[batch["batch"] == batch_no]
    if row.empty:
        return batch.iloc[0:0], "none"
    row = row.iloc[0]
    peers = batch[batch["code"] == row["code"]]
    if len(peers) >= C.MIN_COHORT_N:
        return peers, f"product {row['product_id']}"
    peers = batch[batch["strength"] == row["strength"]]
    return peers, f"strength {row['strength_label']} (product cohort too small)"


def golden_cohort(peers: pd.DataFrame, cqa: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Best-quartile vs worst-quartile peers on the chosen quality attribute."""
    s = peers[cqa].dropna()
    if len(s) < 8:
        return peers.iloc[0:0], peers.iloc[0:0]
    direction = C.CQAS[cqa]["direction"]
    if direction == "higher_better":
        good = peers[peers[cqa] >= s.quantile(1 - C.GOLDEN_QUANTILE)]
        poor = peers[peers[cqa] <= s.quantile(C.POOR_QUANTILE)]
    elif direction == "lower_better":
        good = peers[peers[cqa] <= s.quantile(C.GOLDEN_QUANTILE)]
        poor = peers[peers[cqa] >= s.quantile(1 - C.POOR_QUANTILE)]
    else:  # target: closeness to the cohort median is what "good" means
        d = (peers[cqa] - s.median()).abs()
        good = peers[d <= d.quantile(C.GOLDEN_QUANTILE)]
        poor = peers[d >= d.quantile(1 - C.POOR_QUANTILE)]
    return good, poor


# --------------------------------------------------------------------------
# review by exception
# --------------------------------------------------------------------------
def _bad_direction_z(z: pd.Series, direction: str) -> pd.Series:
    """Signed z converted to 'how bad', so higher always means worse."""
    if direction == "higher_better":
        return -z
    if direction == "lower_better":
        return z
    return z.abs()


def exception_scan(batch: pd.DataFrame,
                   cqas: list[str] | None = None) -> pd.DataFrame:
    """
    Flag batches that deviate from their own peer cohort on any CQA.

    Returns one row per (batch, cqa) exception plus a per-batch severity roll-up.
    """
    cqas = cqas or [c for c in C.CQAS if c in batch.columns]
    recs = []
    for code, peers in batch.groupby("code"):
        for cqa in cqas:
            if cqa not in peers or peers[cqa].notna().sum() < 8:
                continue
            z = robust_z(peers[cqa])
            badness = _bad_direction_z(z, C.CQAS[cqa]["direction"])
            lo, med, hi = control_limits(peers[cqa])
            for idx, b in badness.items():
                if not np.isfinite(b) or b < C.Z_WATCH:
                    continue
                recs.append({
                    "batch": peers.at[idx, "batch"],
                    "code": code,
                    "cqa": cqa,
                    "value": peers.at[idx, cqa],
                    "cohort_median": med,
                    "lcl": lo, "ucl": hi,
                    "z": z.at[idx],
                    "severity_score": float(b),
                    "severity": "Investigate" if b >= C.Z_INVESTIGATE else "Watch",
                })
    return pd.DataFrame(recs)


def exception_queue(batch: pd.DataFrame, exceptions: pd.DataFrame) -> pd.DataFrame:
    """Per-batch roll-up: the work list a quality reviewer actually opens."""
    base = batch[["batch", "batch_id", "product_id", "strength_label",
                  "start_date", "code"]].copy()
    if exceptions.empty:
        base["n_flags"] = 0
        base["max_severity"] = 0.0
        base["status"] = "Review by exception: clear"
        base["flagged_cqas"] = ""
        return base

    g = exceptions.groupby("batch")
    roll = pd.DataFrame({
        "batch": g.size().index,
        "n_flags": g.size().values,
        "max_severity": g["severity_score"].max().values,
        "flagged_cqas": g["cqa"].apply(
            lambda s: ", ".join(C.CQAS[c]["label"] for c in s)).values,
    })
    out = base.merge(roll, on="batch", how="left")
    out["n_flags"] = out["n_flags"].fillna(0).astype(int)
    out["max_severity"] = out["max_severity"].fillna(0.0)
    out["flagged_cqas"] = out["flagged_cqas"].fillna("")
    out["status"] = np.where(out["max_severity"] >= C.Z_INVESTIGATE, "Investigate",
                     np.where(out["max_severity"] >= C.Z_WATCH, "Watch", "Clear"))
    return out.sort_values(["max_severity", "n_flags"], ascending=False)


# --------------------------------------------------------------------------
# prospective (live) scanning
# --------------------------------------------------------------------------
def prospective_exception_scan(batch: pd.DataFrame,
                               cqas: list[str] | None = None,
                               min_history: int = 12) -> pd.DataFrame:
    """
    Score every batch using ONLY the batches that preceded it.

    This is the difference between a retrospective study and a running system.
    `exception_scan` computes cohort limits from the full history — including
    batches made *after* the one being judged. That is legitimate for archive
    analysis, but it cannot be reproduced live and it flatters the result:
    the limits are fitted with knowledge of the future.

    Here each batch is scored against an expanding window of its own product
    cohort as that cohort existed at the moment of manufacture. A batch is only
    scored once `min_history` prior batches exist; before that the process is
    genuinely not yet characterised, and the honest answer is "no baseline yet"
    rather than a limit invented from four points.

    Returns one row per (batch, CQA) with the limits that were live at the time.
    """
    cqas = cqas or [c for c in C.CQAS if c in batch.columns]
    recs: list[dict] = []

    for code, grp in batch.groupby("code"):
        # manufacturing order: month first, batch number as tie-break
        grp = grp.sort_values(["start_date", "batch"])
        ids = grp["batch"].to_numpy()
        for cqa in cqas:
            if cqa not in grp.columns:
                continue
            vals = grp[cqa].to_numpy(dtype=float)
            direction = C.CQAS[cqa]["direction"]
            for i in range(len(grp)):
                v = vals[i]
                if np.isnan(v):
                    continue
                prior = vals[:i]
                hist = prior[~np.isnan(prior)]
                if len(hist) < min_history:
                    recs.append({
                        "batch": int(ids[i]), "code": code, "cqa": cqa,
                        "value": float(v), "n_history": int(len(hist)),
                        "status": "No baseline", "severity_score": 0.0,
                        "z": np.nan, "cohort_median": np.nan,
                        "lcl": np.nan, "ucl": np.nan})
                    continue
                med = float(np.median(hist))
                mad = float(np.median(np.abs(hist - med)) * 1.4826)
                if not mad or not np.isfinite(mad):
                    mad = float(np.std(hist))
                z = (v - med) / mad if mad and np.isfinite(mad) else np.nan
                if not np.isfinite(z):
                    continue
                bad = (-z if direction == "higher_better"
                       else z if direction == "lower_better" else abs(z))
                recs.append({
                    "batch": int(ids[i]), "code": code, "cqa": cqa,
                    "value": float(v), "n_history": int(len(hist)),
                    "status": ("Investigate" if bad >= C.Z_INVESTIGATE
                               else "Watch" if bad >= C.Z_WATCH else "Clear"),
                    "severity_score": float(max(bad, 0.0)),
                    "z": float(z), "cohort_median": med,
                    "lcl": med - 3 * mad, "ucl": med + 3 * mad})
    if not recs:
        return pd.DataFrame(columns=["batch", "code", "cqa", "value", "n_history",
                                     "status", "severity_score", "z",
                                     "cohort_median", "lcl", "ucl"])
    return pd.DataFrame(recs)


def prospective_queue(batch: pd.DataFrame, scan: pd.DataFrame) -> pd.DataFrame:
    """Per-batch roll-up of the prospective scan, in manufacturing order."""
    base = batch[["batch", "batch_id", "product_id", "strength_label",
                  "start_date", "code"]].copy()
    if scan.empty:
        base["status"] = "No baseline"
        base["max_severity"] = 0.0
        base["n_flags"] = 0
        base["flagged_cqas"] = ""
        base["n_scored"] = 0
        return base

    scored = scan[scan["status"] != "No baseline"]
    rows = []
    for bno, d in scored.groupby("batch"):
        flagged = d.loc[d["status"] != "Clear", "cqa"]
        rows.append({
            "batch": int(bno),
            "max_severity": float(d["severity_score"].max()),
            "n_flags": int(len(flagged)),
            "flagged_cqas": ", ".join(C.CQAS[c]["label"] for c in flagged),
            "n_scored": int(len(d)),
        })
    roll = pd.DataFrame(rows)
    out = base.merge(roll, on="batch", how="left")
    out["max_severity"] = out["max_severity"].fillna(0.0)
    out["n_flags"] = out["n_flags"].fillna(0).astype(int)
    out["flagged_cqas"] = out["flagged_cqas"].fillna("")
    out["n_scored"] = out["n_scored"].fillna(0).astype(int)
    out["status"] = np.where(
        out["n_scored"] == 0, "No baseline",
        np.where(out["max_severity"] >= C.Z_INVESTIGATE, "Investigate",
                 np.where(out["max_severity"] >= C.Z_WATCH, "Watch", "Clear")))
    return out.sort_values(["start_date", "batch"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# driver universe
# --------------------------------------------------------------------------
def driver_columns(batch: pd.DataFrame, include_low_coverage: bool = False,
                   min_coverage: float = 0.7, cqa: str | None = None) -> list[str]:
    """
    Candidate root causes for `cqa`.

    Signals that measure the same physical quantity as the outcome are removed:
    they produce spectacular correlations and zero insight (see config.PROXY_SIGNALS).
    """
    cols = [c for c in C.RM_ATTRS + C.PROCESS_FEATURES if c in batch.columns]
    cols += [c for c in batch.columns
             if c.startswith("ts_") and c not in ("ts_n_samples",)]
    if not include_low_coverage:
        cols = [c for c in cols if c not in C.LOW_COVERAGE_ATTRS]
    if cqa:
        blocked = set(C.proxy_signals(cqa)) | {cqa}
        cols = [c for c in cols if c not in blocked]
    keep = []
    for c in cols:
        s = batch[c]
        if s.notna().mean() < min_coverage:
            continue
        if s.nunique(dropna=True) < 5:          # constant within cohort -> useless
            continue
        keep.append(c)
    return keep


# --------------------------------------------------------------------------
# RCA
# --------------------------------------------------------------------------
@dataclass
class RCAResult:
    cqa: str
    cohort_label: str
    n_peers: int
    drivers: pd.DataFrame            # ranked, with the three evidence tiers
    model_r2: float | None
    model_r2_std: float | None
    baseline_r2: float | None
    n_used: int
    warnings: list[str] = field(default_factory=list)

    @property
    def model_is_trustworthy(self) -> bool:
        return (self.model_r2 is not None and self.model_r2 > 0.10
                and self.n_used >= 40)


def run_rca(peers: pd.DataFrame, cqa: str, cohort_label: str,
            focus_batch: int | None = None,
            include_low_coverage: bool = False) -> RCAResult:
    """
    Rank candidate drivers of a quality outcome inside one peer cohort.

    Deliberately NOT a single black-box score. Each driver carries:
      corr / corr_p        Tier 2 association across the cohort
      good_vs_poor_delta   Tier 1 descriptive contrast (golden vs worst quartile)
      importance           Tier 3 multivariate permutation importance
      batch_z              where the batch under investigation sits
      tiers_agree          how many independent tiers point the same way
    """
    warns: list[str] = []
    drivers = driver_columns(peers, include_low_coverage, cqa=cqa)
    d = peers.dropna(subset=[cqa])
    n = len(d)
    if n < 20 or not drivers:
        return RCAResult(cqa, cohort_label, len(peers), pd.DataFrame(),
                         None, None, None, n,
                         ["Cohort too small for reliable analysis."])
    if n < 40:
        warns.append(f"Only {n} peer batches — treat model evidence as indicative.")

    y = d[cqa].astype(float)
    X = d[drivers].astype(float)
    X = X.fillna(X.median())

    good, poor = golden_cohort(d, cqa)

    # ---- Tier 2: association -------------------------------------------------
    rows = []
    for c in drivers:
        x = X[c]
        if x.nunique() < 5:
            continue
        r = x.rank().corr(y.rank())                       # Spearman, no scipy dep
        # two-sided p via t approximation
        p = np.nan
        if np.isfinite(r) and n > 3 and abs(r) < 1:
            t = r * np.sqrt((n - 2) / (1 - r ** 2))
            from math import erf, sqrt
            p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
        # ---- Tier 1: descriptive contrast -----------------------------------
        delta = np.nan
        if len(good) >= 4 and len(poor) >= 4:
            gm, pm = d.loc[good.index, c].median(), d.loc[poor.index, c].median()
            pooled_sd = d[c].std(ddof=0)
            delta = (gm - pm) / pooled_sd if pooled_sd else np.nan
        rows.append({"driver": c, "corr": r, "corr_p": p, "good_vs_poor_delta": delta})
    res = pd.DataFrame(rows)
    if res.empty:
        return RCAResult(cqa, cohort_label, len(peers), res, None, None, None, n,
                         warns + ["No driver varied enough inside this cohort."])

    # ---- Tier 3: multivariate model -----------------------------------------
    model_r2 = model_std = base_r2 = None
    use = [c for c in res["driver"] if c in X.columns]
    Xm = X[use]
    if n >= 40:
        rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=3,
                                   random_state=0, n_jobs=-1)
        cv = KFold(5, shuffle=True, random_state=0)
        sc = cross_val_score(rf, Xm, y, cv=cv, scoring="r2")
        model_r2, model_std = float(sc.mean()), float(sc.std())
        base_r2 = 0.0
        rf.fit(Xm, y)
        pi = permutation_importance(rf, Xm, y, n_repeats=15,
                                    random_state=0, n_jobs=-1, scoring="r2")
        res = res.merge(pd.DataFrame({"driver": use,
                                      "importance": pi.importances_mean,
                                      "importance_sd": pi.importances_std}),
                        on="driver", how="left")
        if model_r2 is not None and model_r2 <= 0.10:
            warns.append(
                f"Cross-validated model R² is {model_r2:.2f} — within this cohort the "
                "recorded variables explain little of the outcome. Model ranking is "
                "shown for completeness but should not drive the investigation.")
    else:
        res["importance"] = np.nan
        res["importance_sd"] = np.nan
        warns.append("Cohort below 40 batches — multivariate model not fitted.")

    # ---- where does the batch under investigation sit? ----------------------
    res["batch_z"] = np.nan
    if focus_batch is not None and focus_batch in set(d["batch"]):
        frow = d.loc[d["batch"] == focus_batch].iloc[0]
        for i, c in enumerate(res["driver"]):
            z = robust_z(d[c])
            idx = d.index[d["batch"] == focus_batch][0]
            res.iat[i, res.columns.get_loc("batch_z")] = float(z.loc[idx])

    # ---- tier agreement ------------------------------------------------------
    res["corr_rank"] = res["corr"].abs().rank(ascending=False)
    res["imp_rank"] = res["importance"].rank(ascending=False)
    res["delta_rank"] = res["good_vs_poor_delta"].abs().rank(ascending=False)
    k = max(5, len(res) // 5)
    tiers = ((res["corr_rank"] <= k).astype(int)
             + (res["imp_rank"] <= k).fillna(0).astype(int)
             + (res["delta_rank"] <= k).astype(int))
    res["tiers_agree"] = tiers
    res["significant"] = res["corr_p"] < 0.05

    # composite ordering: agreement first, then strength of association
    res["rank_score"] = (res["tiers_agree"] * 10
                         + res["corr"].abs().fillna(0) * 5
                         + res["importance"].fillna(0).clip(lower=0) * 3)
    res = res.sort_values("rank_score", ascending=False).reset_index(drop=True)

    res["label"] = res["driver"].map(C.feat_label)
    res["unit"] = res["driver"].map(C.feat_unit)
    res["plain"] = res["driver"].map(C.feat_plain)
    res["domain"] = res["driver"].map(C.feat_domain)

    # multiple-comparison honesty
    n_tests = len(res)
    n_sig = int(res["significant"].sum())
    expected_false = 0.05 * n_tests
    if n_sig and n_sig < 2 * expected_false:
        warns.append(
            f"{n_sig} of {n_tests} drivers reached p<0.05; about {expected_false:.0f} "
            "would be expected by chance alone. Treat single significant results as "
            "hypotheses, not findings.")
    return RCAResult(cqa, cohort_label, len(peers), res,
                     model_r2, model_std, base_r2, n, warns)


def confirmatory_signals(peers: pd.DataFrame, cqa: str,
                         focus_batch: int | None = None) -> pd.DataFrame:
    """
    In-process signals that measure the same quantity as the lab outcome.

    Excluded from root-cause ranking on purpose, but reported separately because
    they are the earliest available evidence: the press sees them during the run,
    hours before the lab result exists. That is a detection story, not a cause.
    """
    sigs = [s for s in C.proxy_signals(cqa) if s in peers.columns]
    d = peers.dropna(subset=[cqa])
    rows = []
    for s in sigs:
        sub = d[[s, cqa]].dropna()
        if len(sub) < 10:
            continue
        r = sub[s].rank().corr(sub[cqa].rank())
        rec = {"signal": s, "label": C.feat_label(s), "corr": r,
               "plain": C.feat_plain(s)}
        if focus_batch is not None and focus_batch in set(d["batch"]):
            z = robust_z(d[s])
            idx = d.index[d["batch"] == focus_batch][0]
            rec["batch_z"] = float(z.loc[idx])
            rec["batch_value"] = float(d.at[idx, s])
        rows.append(rec)
    if not rows:      # most CQAs have no same-quantity proxy at all
        return pd.DataFrame(columns=["signal", "label", "corr", "plain"])
    return pd.DataFrame(rows).sort_values("corr", key=abs, ascending=False)


def explain_driver(row: pd.Series, cqa: str) -> str:
    """One sentence a non-statistician can act on."""
    lab = row.get("label", row["driver"])
    cqa_lab = C.CQAS[cqa]["label"]
    r = row.get("corr", np.nan)
    if not np.isfinite(r):
        return f"{lab}: insufficient data in this cohort."
    direction = "higher" if r > 0 else "lower"
    strength = ("strong" if abs(r) > 0.5 else
                "moderate" if abs(r) > 0.3 else "weak")
    txt = (f"Across comparable batches, {direction} {lab.lower()} goes with higher "
           f"{cqa_lab.lower()} ({strength} association, r={r:+.2f}).")
    bz = row.get("batch_z", np.nan)
    if np.isfinite(bz) and abs(bz) >= 1.5:
        side = "above" if bz > 0 else "below"
        txt += (f" This batch sat {abs(bz):.1f} robust SD {side} its peers on "
                f"{lab.lower()}.")
    return txt


# --------------------------------------------------------------------------
# cross-cohort driver scan (for the "which lever matters most" view)
# --------------------------------------------------------------------------
def pooled_vs_within(batch: pd.DataFrame, cqa: str,
                     min_n: int = 20) -> pd.DataFrame:
    """
    The evidence behind the product's central design claim. For each driver,
    compare the naive pooled correlation against the cohort-weighted
    within-product correlation, and flag sign reversals.
    """
    drivers = driver_columns(batch)
    rows = []
    d = batch.dropna(subset=[cqa])
    for c in drivers:
        s = d[[c, cqa, "code"]].dropna()
        if len(s) < 100 or s[c].nunique() < 5:
            continue
        pooled = s[c].rank().corr(s[cqa].rank())
        ws, wn = [], []
        for _, g in s.groupby("code"):
            if len(g) >= min_n and g[c].nunique() >= 5:
                r = g[c].rank().corr(g[cqa].rank())
                if np.isfinite(r):
                    ws.append(r)
                    wn.append(len(g))
        if not ws:
            continue
        within = float(np.average(ws, weights=wn))
        rows.append({
            "driver": c, "label": C.feat_label(c), "domain": C.feat_domain(c),
            "pooled_r": pooled, "within_r": within, "n_cohorts": len(ws),
            "sign_flip": bool(np.sign(pooled) != np.sign(within)),
            "magnitude_drop": abs(pooled) - abs(within),
        })
    if not rows:
        return pd.DataFrame(columns=["driver", "label", "domain", "pooled_r",
                                     "within_r", "n_cohorts", "sign_flip",
                                     "magnitude_drop"])
    return pd.DataFrame(rows).sort_values("within_r", key=abs, ascending=False)
