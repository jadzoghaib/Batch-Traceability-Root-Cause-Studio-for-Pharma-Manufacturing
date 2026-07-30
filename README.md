# Batch Investigation Console — Quality-to-Process Traceability & RCA for Pharma Manufacturing

Trace final product quality back to incoming raw-material attributes and in-process
compression conditions, compare good against bad batches, and rank likely root
causes — on **1,005 real production batches** of a tablet compression process.

---

## The finding that shaped the product

Most quality dashboards correlate process parameters against quality results across
every batch in the plant. On this dataset, that approach is not merely noisy — it
frequently points the **opposite way** from the truth.

| | Pooled across all products | Within peer cohort |
|---|---|---|
| Main compression force → dissolution | **−0.42** | **+0.10** |
| Die fill depth → dissolution | **−0.43** | **+0.07** |

**22 of 44 candidate drivers reverse their correlation sign** between pooled and
within-product analysis. Product identity alone explains 85% of hardness variance
and 82% of impurities variance. A pooled "top correlations" chart would send a
quality investigator hunting for exactly the wrong condition.

Every comparison in this tool is therefore made **only against structurally
comparable batches** — same product code, falling back to same strength with the
substitution stated on screen. The evidence for this is inspectable in-app on the
**Method** page.

---

## What it does

| Page | Purpose |
|---|---|
| **Live operations** | The plant as it runs: step a production clock month by month, watch batches arrive and cohorts mature |
| **Overview** | Portfolio health, quality trend, products ranked by flag rate |
| **Review queue** | Review-by-exception work list; clears 66% of batches from manual review |
| **Batch detail** | One batch end to end: quality position, 10-second compression trajectory, material genealogy, full process record |
| **Compare batches** | Golden (best quartile) vs poor (worst quartile) condition profile |
| **Root cause** | Ranked drivers with three separated evidence tiers |
| **Materials** | Lot and supplier drilldown within a product cohort |
| **Method** | The pooled-vs-cohort evidence, the pipeline, and an honest limitations register |

### Three evidence tiers, never fused

A single "AI confidence score" would look more impressive and mean considerably
less. Each candidate driver carries three independent lines of evidence:

1. **Descriptive** — this batch differs from its peers on X *(fact)*
2. **Association** — across peers, X tracks the outcome *(correlation)*
3. **Model** — X carries weight jointly, via permutation importance on a
   cross-validated random forest *(multivariate)*

A driver is promoted to "prioritised" only when at least two tiers agree.

### Live scoring, not hindsight

The **Live operations** page runs the plant forward on a production clock. Every
batch is scored against an expanding window of its own product cohort — the control
limits that existed *at the moment of manufacture*. A product needs 12 prior batches
before it earns limits at all; until then the status is `No baseline`, not a limit
invented from four points.

This matters more than it sounds. Scoring against the full dataset is look-ahead
bias, and on this data it **changes the verdict for 394 of 1,005 batches (39%)**:

| | |
|---|---|
| Batches with no baseline yet when made | **223** |
| Raised live, but clear in hindsight | **26** |
| Genuine investigations missed during cold start | **24** |
| Products characterised, Nov 2018 → Apr 2021 | **2 of 10 → 15 of 25** |

Any deviation-reduction claim that ignores this is not a number you could reproduce
in production. The app shows the live-vs-retrospective confusion matrix rather than
quoting the flattering figure.

### The same-quantity guard

In-process weight RSD (`SREL`, from the press checkweigher) correlates **0.80**
with lab-measured tablet weight RSD. That is not a root cause — it is the same
physical quantity measured by a second instrument. Left in, it dominates the
ranking and explains nothing.

Such signals are moved into a separate **confirmatory in-process signals**
lane. They keep their real value — visibility *during* the run, hours before a lab
result exists — without masquerading as causes. Removing them honestly dropped
model R² from 0.175 to 0.070, which the app reports rather than hides.

---

## Quick start

```bash
uv venv .venv && uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

```bash
python scripts/download_data.py
```

```bash
python -m batchrca.etl
```

```bash
streamlit run app.py
```

`download_data.py` pulls ~30 MB from figshare. The ETL streams the 346 MB of
10-second time series straight out of the zip and reduces it to a 420 KB Parquet
star schema — raw data never enters the repository.

---

## Data

Žagar, J. & Mihelič, J. *Big data collection in pharmaceutical manufacturing and its
use for product quality predictions.* **Scientific Data 9, 99 (2022).**
DOI [10.1038/s41597-022-01203-x](https://doi.org/10.1038/s41597-022-01203-x) ·
figshare `10.6084/m9.figshare.c.5645578` · CC-BY 4.0

- 1,005 production batches · 25 product codes · 4 strengths · Nov 2018 – Apr 2021
- Incoming lot testing for API, SMCC, lactose, starch (238 / 18 / 22 / 17 lots)
- Compression time series at 10-second resolution (14 parameters)
- Final quality results: dissolution, hardness, tensile strength, weight RSD, yield, impurities

### Data quality issues found and handled

Real published data is messier than its documentation suggests:

- **Two timestamp formats.** ISO `2019-01-17 04:09:38` alongside compact
  `07052019 20:14` (day-first). Pandas' `format="mixed"` silently returns `NaT`
  for the second, which quietly dropped four of the largest products — 589
  batches — from the feature table. Each shape is now parsed explicitly.
- **Whitespace-padded nulls.** Six API columns arrive as strings because missing
  values are runs of spaces, invisible to a normal null check.
- **`api_l_impurity` missing for ~36% of batches** — excluded from modelling by
  default rather than imputed. Inventing values would fabricate evidence.
- **Duplicated quality columns.** `Process.csv` repeats the lab results with 18
  extra nulls; `Laboratory.csv` is treated as authoritative.

---

## Architecture

```
pharma-batch-rca/
├── app.py                    Overview (entry point)
├── pages/                    Review queue · Batch detail · Compare · RCA · Materials · Method
├── src/batchrca/
│   ├── config.py             Schema, CQA vocabulary, proxy-signal guard, analytical policy
│   ├── etl.py                Raw CSV/zip → Parquet star schema + time-series features
│   ├── analytics.py          Cohorts, robust SPC, exception scan, three-tier RCA
│   ├── charts.py             Plotly vocabulary
│   ├── ui.py                 Design system
│   └── data.py               Cached access layer
├── scripts/download_data.py
└── smoke_test.py             Exercises every CQA × cohort × chart path
```

**Streamlit + Parquet + Plotly.** For a single-developer demo, Streamlit keeps
analytics and UI in one process with no API layer to maintain; the effort saved
went into a custom design system instead. Parquet keeps the whole star schema at
532 KB, and the raw trajectory for a single batch is read lazily from the zip only
when that batch is opened.

No query engine is used, deliberately: cohorts are at most a few hundred rows and
pandas handles them in milliseconds. The Parquet layout is DuckDB-ready for the
point where trajectories need querying *across* batches rather than one at a time,
but adding it now would be a dependency carrying no weight.

### Data model

```
dim_product (25)          ─┐
dim_material_lot (295)    ─┼─→  fct_batch (1,005 × 109)
fct_timeseries_feat       ─┘         join key: batch (1:1, zero orphans)
```

### Precomputed scans

The ETL also writes the exception scans and the pooled-vs-cohort evidence to
Parquet, so every page renders immediately on a cold container instead of
recomputing on first load:

| Artefact | Rows | Read | Recompute |
|---|---|---|---|
| `fct_exception_scan` / `_queue` | 445 / 1,005 | | 0.17 s |
| `fct_prospective_scan` / `_queue` | 8,039 / 1,005 | | 0.75 s |
| `fct_pooled_vs_within` (8 attributes) | 464 | | 0.41 s each |
| **all three combined** | | **0.010 s** | 1.16 s |

These are deterministic functions of `fct_batch`, and the loaders fall back to
computing them if the files are missing or unreadable — the cache is an
optimisation, never a dependency.

---

## Limitations

The dataset contains **no registered specifications**, so every limit shown is a
statistically derived control limit (median ±3 robust SD of the peer cohort),
labelled as such throughout. It also contains **no deviation or CAPA records**, so
the driver ranking cannot be scored against investigator-confirmed root causes.

This is observational production data. Nothing was randomised, unrecorded factors
(operator, humidity, tooling age) are absent, and collinear parameters can split
importance arbitrarily. Outputs are **prioritised leads for investigation, not
proven causes** — the app states this everywhere it ranks anything.
