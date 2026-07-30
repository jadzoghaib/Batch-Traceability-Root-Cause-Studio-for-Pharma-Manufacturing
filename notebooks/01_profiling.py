# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy"]
# ///
"""Profile the Zagar & Mihelic pharma manufacturing dataset to ground the data model."""
import pandas as pd
import numpy as np

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 100)

RAW = "data/raw"

lab = pd.read_csv(f"{RAW}/Laboratory.csv", sep=";")
proc = pd.read_csv(f"{RAW}/Process.csv", sep=";")
norm = pd.read_csv(f"{RAW}/Normalization.csv", sep=";")

print("=" * 70)
print("SHAPES")
print("=" * 70)
print(f"Laboratory   : {lab.shape}")
print(f"Process      : {proc.shape}")
print(f"Normalization: {norm.shape}")

print("\n" + "=" * 70)
print("JOIN INTEGRITY (batch is the key)")
print("=" * 70)
print(f"lab.batch unique     : {lab['batch'].nunique()} / {len(lab)} rows")
print(f"proc.batch unique    : {proc['batch'].nunique()} / {len(proc)} rows")
lab_b, proc_b = set(lab["batch"]), set(proc["batch"])
print(f"in lab not proc      : {len(lab_b - proc_b)}")
print(f"in proc not lab      : {len(proc_b - lab_b)}")
print(f"intersection         : {len(lab_b & proc_b)}")
print(f"lab codes            : {sorted(lab['code'].unique())}")
print(f"norm codes           : {sorted(norm['Product code'].unique())}")
print(f"strengths            : {lab['strength'].value_counts().to_dict()}")
print(f"batch sizes          : {sorted(lab['size'].unique())}")

print("\n" + "=" * 70)
print("MISSINGNESS (% null, only cols with >0)")
print("=" * 70)
for name, df in [("LAB", lab), ("PROC", proc)]:
    miss = (df.isna().mean() * 100).round(1)
    miss = miss[miss > 0].sort_values(ascending=False)
    print(f"\n--- {name} ---")
    print(miss.to_string() if len(miss) else "  (no missing)")

print("\n" + "=" * 70)
print("CANDIDATE QUALITY TARGETS (dependent vars)")
print("=" * 70)
targets = [
    "dissolution_av", "dissolution_min", "impurities_total", "impurity_o",
    "impurity_l", "resodual_solvent", "batch_yield", "tbl_yield",
    "fct_av_hardness", "tbl_av_hardness", "tbl_rsd_weight", "fct_rsd_weight",
    "tbl_tensile", "fct_tensile",
]
print(lab[[c for c in targets if c in lab.columns]].describe().T.round(3).to_string())

print("\n" + "=" * 70)
print("RAW MATERIAL ATTRIBUTES (incoming lots)")
print("=" * 70)
rm = [c for c in lab.columns if c.startswith(("api_", "lactose_", "smcc_", "starch_"))]
rm_num = [c for c in rm if pd.api.types.is_numeric_dtype(lab[c])]
print(lab[rm_num].describe().T.round(3).to_string())

print("\n" + "=" * 70)
print("MATERIAL LOT IDs (supplier/lot drilldown feasibility)")
print("=" * 70)
for c in ["api_code", "api_batch", "smcc_batch", "lactose_batch", "starch_batch"]:
    if c in lab.columns:
        print(f"{c:16s}: {lab[c].nunique():3d} distinct  {sorted(lab[c].dropna().unique())[:12]}")

print("\n" + "=" * 70)
print("PROCESS FEATURES")
print("=" * 70)
pnum = proc.select_dtypes(include=[np.number]).columns.tolist()
print(proc[pnum].describe().T.round(3).to_string())
print(f"\nweekend flag: {proc['weekend'].value_counts().to_dict()}")

print("\n" + "=" * 70)
print("TIME COVERAGE (start column)")
print("=" * 70)
print(f"distinct start values: {lab['start'].nunique()}")
print(f"sample: {lab['start'].dropna().unique()[:18].tolist()}")

print("\n" + "=" * 70)
print("SIGNAL CHECK: corr of drivers vs dissolution_av (pooled, all products)")
print("=" * 70)
drivers = rm_num + [c for c in pnum if c not in ("batch", "code")]
merged = lab.merge(proc, on="batch", suffixes=("", "_p"))
if "dissolution_av" in merged:
    cors = {}
    for c in drivers:
        if c in merged.columns and merged[c].notna().sum() > 50:
            v = merged[c].corr(merged["dissolution_av"], method="spearman")
            if pd.notna(v):
                cors[c] = v
    s = pd.Series(cors).sort_values(key=abs, ascending=False)
    print(s.head(20).round(3).to_string())
