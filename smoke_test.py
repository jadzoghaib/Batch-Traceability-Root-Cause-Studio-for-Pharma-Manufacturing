"""
Exercise every analytical path the UI can reach, including the awkward corners:
tiny cohorts, all-missing attributes, cohorts with no proxy signals, products
with too few batches to model. Cheaper and far more thorough than clicking.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd

from batchlens import analytics as A
from batchlens import charts as CH
from batchlens import config as C

batch = pd.read_parquet(C.PROCESSED / "fct_batch.parquet")
cqas = [c for c in C.CQAS if c in batch.columns]
fails: list[str] = []


def check(name: str, fn):
    try:
        fn()
    except Exception as e:
        fails.append(f"{name}: {type(e).__name__}: {e}")
        traceback.print_exc()


print(f"batches={len(batch)}  cqas={len(cqas)}  products={batch['code'].nunique()}")

# ---- exception engine -----------------------------------------------------
ex = A.exception_scan(batch)
q = A.exception_queue(batch, ex)
print(f"exceptions={len(ex)}  queue={len(q)}  {q['status'].value_counts().to_dict()}")

# empty-input robustness
check("exception_queue(empty)",
      lambda: A.exception_queue(batch, ex.iloc[0:0]))

# ---- every CQA x every viable cohort --------------------------------------
codes = sorted(batch["code"].unique())
big = [c for c in codes if (batch["code"] == c).sum() >= C.MIN_COHORT_N]
small = [c for c in codes if c not in big]
print(f"cohorts: {len(big)} large, {len(small)} small")

for cqa in cqas:
    for code in big:
        peers = batch[batch["code"] == code]
        focus = int(peers["batch"].iloc[0])
        check(f"rca[{cqa}|P-{code}]",
              lambda p=peers, c=cqa, f=focus: A.run_rca(p, c, "t", focus_batch=f))
        check(f"confirm[{cqa}|P-{code}]",
              lambda p=peers, c=cqa, f=focus: A.confirmatory_signals(p, c, f))
        check(f"golden[{cqa}|P-{code}]",
              lambda p=peers, c=cqa: A.golden_cohort(p, c))

# small cohorts must degrade gracefully, not explode
for code in small:
    peers = batch[batch["code"] == code]
    focus = int(peers["batch"].iloc[0])
    check(f"rca_small[P-{code}] n={len(peers)}",
          lambda p=peers, f=focus: A.run_rca(p, C.PRIMARY_CQA, "t", focus_batch=f))
    check(f"peer_cohort_fallback[P-{code}]",
          lambda f=focus: A.peer_cohort(batch, f))

# ---- charts across all CQAs ----------------------------------------------
for cqa in cqas:
    peers = batch[batch["code"] == big[0]]
    focus = int(peers["batch"].iloc[0])
    res = A.run_rca(peers, cqa, "t", focus_batch=focus)
    good, poor = A.golden_cohort(peers, cqa)
    check(f"control_chart[{cqa}]", lambda p=peers, c=cqa, f=focus:
          CH.control_chart(p, c, f))
    check(f"distribution[{cqa}]", lambda p=peers, c=cqa, f=focus:
          CH.cohort_distribution(p, c, f))
    if not res.drivers.empty:
        check(f"ranking[{cqa}]", lambda r=res: CH.driver_ranking(r.drivers))
        drv = res.drivers.head(6)["driver"].tolist()
        check(f"good_vs_poor[{cqa}]", lambda p=peers, g=good, po=poor, d=drv:
              CH.good_vs_poor(p, g, po, d))
        check(f"scatter[{cqa}]", lambda p=peers, d=drv[0], c=cqa, f=focus:
              CH.driver_scatter(p, d, c, f))
    check(f"lot_effect[{cqa}]", lambda c=cqa: CH.lot_effect(batch, "api_batch", c,
                                                            code=big[0]))
    check(f"pooled_within[{cqa}]", lambda c=cqa: CH.pooled_vs_within_chart(
        A.pooled_vs_within(batch, c)))

# ---- materials paths ------------------------------------------------------
for mat, spec in C.MATERIALS.items():
    lot_col = spec["lot_col"]
    for cqa in cqas[:3]:
        check(f"lot_effect[{mat}|{cqa}]",
              lambda l=lot_col, c=cqa: CH.lot_effect(batch, l, c, code=big[0]))

# ---- low-coverage toggle --------------------------------------------------
check("rca_low_coverage",
      lambda: A.run_rca(batch[batch["code"] == big[0]], C.PRIMARY_CQA, "t",
                        include_low_coverage=True))

# ---- proxy guard actually holds ------------------------------------------
for cqa, proxies in C.PROXY_SIGNALS.items():
    if cqa not in batch.columns:
        continue
    cols = A.driver_columns(batch, cqa=cqa)
    leaked = [p for p in proxies if p in cols]
    if leaked:
        fails.append(f"PROXY LEAK [{cqa}]: {leaked} present in driver ranking")
    if cqa in cols:
        fails.append(f"SELF LEAK [{cqa}]: outcome present among its own drivers")

print("\n" + "=" * 60)
if fails:
    print(f"FAILURES: {len(fails)}")
    for f in fails[:25]:
        print("  ✗", f)
    sys.exit(1)
print("ALL PATHS PASS")
