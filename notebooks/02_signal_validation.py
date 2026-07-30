# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "scipy", "scikit-learn"]
# ///
"""
Insight validation: is there REAL signal linking materials+process -> quality,
or is everything just product-code / strength confounding?

This decides whether the RCA page is honest or theatre.
"""
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.dummy import DummyRegressor

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

RAW = "data/raw"
lab = pd.read_csv(f"{RAW}/Laboratory.csv", sep=";")
proc = pd.read_csv(f"{RAW}/Process.csv", sep=";")

# Laboratory is authoritative for quality; drop Process' duplicated quality cols
dup_q = ["Drug release average (%)", "Drug release min (%)", "Residual solvent",
         "Total impurities", "Impurity O", "Impurity L"]
proc_f = proc.drop(columns=[c for c in dup_q if c in proc.columns])
df = lab.merge(proc_f.drop(columns=["code"]), on="batch", how="inner")
print(f"merged: {df.shape}")

TARGET = "dissolution_av"

RM = ["api_water", "api_total_impurities", "api_l_impurity", "api_content",
      "api_ps01", "api_ps05", "api_ps09", "lactose_water", "lactose_sieve0045",
      "lactose_sieve015", "lactose_sieve025", "smcc_water", "smcc_td", "smcc_bd",
      "smcc_ps01", "smcc_ps05", "smcc_ps09", "starch_ph", "starch_water"]
PROCF = ["tbl_speed_mean", "tbl_speed_change", "tbl_speed_0_duration", "fom_mean",
         "fom_change", "SREL_startup_mean", "SREL_production_mean", "SREL_production_max",
         "main_CompForce mean", "main_CompForce_sd", "main_CompForce_median",
         "pre_CompForce_mean", "tbl_fill_mean", "tbl_fill_sd", "cyl_height_mean",
         "stiffness_mean", "stiffness_max", "stiffness_min", "ejection_mean",
         "ejection_max", "ejection_min", "Startup_tbl_fill_maxDifference",
         "Startup_main_CompForce_mean", "Startup_tbl_fill_mean", "total_waste", "startup_waste"]
FEATS = [c for c in RM + PROCF if c in df.columns]

print("\n" + "=" * 78)
print("1. HOW MUCH OF QUALITY VARIANCE IS JUST 'WHICH PRODUCT IS IT?'")
print("=" * 78)
for tgt in ["dissolution_av", "fct_av_hardness", "tbl_rsd_weight", "impurities_total"]:
    grand = df[tgt].var()
    within = df.groupby("code")[tgt].transform("mean")
    eta2 = 1 - (df[tgt] - within).var() / grand
    f, p = stats.f_oneway(*[g[tgt].dropna().values for _, g in df.groupby("code") if len(g) > 2])
    print(f"{tgt:20s} eta^2(product_code) = {eta2:5.1%}   ANOVA F={f:7.1f} p={p:.2e}")
print("\n-> If eta^2 is large, POOLED correlations are confounded by product mix.")

print("\n" + "=" * 78)
print("2. POOLED vs WITHIN-PRODUCT CORRELATION (Simpson's paradox check)")
print("=" * 78)
rows = []
for c in FEATS:
    s = df[[c, TARGET, "code"]].dropna()
    if len(s) < 100 or s[c].nunique() < 5:
        continue
    pooled = s[c].rank().corr(s[TARGET].rank())
    # within-product: average correlation weighted by cohort size, only cohorts with variation
    ws, wn = [], []
    for code, g in s.groupby("code"):
        if len(g) >= 20 and g[c].nunique() >= 5:
            r = g[c].rank().corr(g[TARGET].rank())
            if pd.notna(r):
                ws.append(r); wn.append(len(g))
    if not ws:
        continue
    within = float(np.average(ws, weights=wn))
    rows.append({"feature": c, "pooled_r": pooled, "within_r": within,
                 "flip": np.sign(pooled) != np.sign(within), "n_cohorts": len(ws)})
res = pd.DataFrame(rows).sort_values("within_r", key=abs, ascending=False)
print(res.head(18).round(3).to_string(index=False))
n_flip = int(res["flip"].sum())
print(f"\n-> {n_flip}/{len(res)} features FLIP SIGN between pooled and within-product.")

print("\n" + "=" * 78)
print("3. PREDICTIVE SIGNAL: can we beat baseline, grouped CV by product code?")
print("=" * 78)
print("   (GroupKFold on product code = predict for products never seen in training;")
print("    this is the honest test that we're not just memorising product identity.)")
d = df[FEATS + [TARGET, "code"]].dropna(subset=[TARGET])
X = d[FEATS].fillna(d[FEATS].median())
y = d[TARGET]
g = d["code"]

rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=0, n_jobs=-1)
for name, model, groups in [
    ("Dummy (mean)      ", DummyRegressor(), g),
    ("RF  GroupKFold    ", rf, g),
]:
    cv = GroupKFold(n_splits=5)
    sc = cross_val_score(model, X, y, cv=cv, groups=groups, scoring="r2")
    print(f"{name} R2 = {sc.mean():6.3f}  (folds: {np.round(sc,2).tolist()})")

from sklearn.model_selection import KFold
sc = cross_val_score(rf, X, y, cv=KFold(5, shuffle=True, random_state=0), scoring="r2")
print(f"RF  random KFold   R2 = {sc.mean():6.3f}   <-- inflated: product identity leaks in")

print("\n" + "=" * 78)
print("4. WITHIN-COHORT MODEL: largest product code only (apples-to-apples)")
print("=" * 78)
top = df["code"].value_counts().head(3)
for code, n in top.items():
    sub = df[df["code"] == code]
    sx = sub[FEATS].fillna(sub[FEATS].median())
    sy = sub[TARGET]
    keep = sx.nunique() > 3
    sx = sx.loc[:, keep]
    if len(sub) < 40:
        continue
    sc = cross_val_score(RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                               random_state=0, n_jobs=-1),
                         sx, sy, cv=KFold(5, shuffle=True, random_state=0), scoring="r2")
    base = cross_val_score(DummyRegressor(), sx, sy, cv=KFold(5, shuffle=True, random_state=0), scoring="r2")
    print(f"code {code:3d} (n={n:3d}, {sx.shape[1]:2d} varying feats): RF R2={sc.mean():6.3f}  dummy={base.mean():6.3f}")

print("\n" + "=" * 78)
print("5. MATERIAL LOT EFFECT: does API lot shift quality within a product?")
print("=" * 78)
for code in top.index[:3]:
    sub = df[df["code"] == code]
    grps = [gg[TARGET].dropna().values for _, gg in sub.groupby("api_batch") if len(gg) >= 3]
    if len(grps) >= 3:
        f, p = stats.f_oneway(*grps)
        print(f"code {code:3d}: api_batch groups={len(grps):3d}  ANOVA F={f:6.2f} p={p:.3e}"
              f"  {'<-- significant lot effect' if p < 0.05 else ''}")
