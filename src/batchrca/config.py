"""
Central schema, vocabulary and analytical policy for Batch Investigation Console.

Design note on limits
---------------------
The published dataset does NOT include registered specification limits. We therefore
never display a "spec limit". Instead we derive *statistical control limits* from the
historical peer-cohort distribution (SPC-style) and label them as such everywhere in
the UI. Fabricating regulatory specs for a demo would be dishonest and, in a GxP
context, actively misleading.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TS_DIR = RAW / "Process"          # extracted per-product time-series CSVs

# figshare file ids (collection 10.6084/m9.figshare.c.5645578)
FIGSHARE_FILES = {
    "Laboratory.csv": "30966250",
    "Process.csv": "30874192",
    "Normalization.csv": "30874189",
    "Process.zip": "30874219",     # ~30 MB zipped, ~346 MB unzipped
}

# --------------------------------------------------------------------------
# Slovenian month abbreviations used in Laboratory.start ("nov.18")
# --------------------------------------------------------------------------
SL_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "maj": 5, "jun": 6,
    "jul": 7, "avg": 8, "sep": 9, "okt": 10, "nov": 11, "dec": 12,
}

# --------------------------------------------------------------------------
# Quality outcomes (CQAs). `direction` = which tail is bad.
#   higher_better -> low values are the failure mode
#   lower_better  -> high values are the failure mode
#   target        -> deviation in either direction is bad
# eta2_product records how much variance is explained by product identity alone
# (measured, see notebooks/01_profiling). It drives whether pooled analysis is safe.
# --------------------------------------------------------------------------
CQAS: dict[str, dict] = {
    "dissolution_av": {
        "label": "Dissolution (avg)", "unit": "%", "direction": "higher_better",
        "eta2_product": 0.397, "family": "Performance",
        "plain": "How fast the tablet releases its active ingredient. Low values risk "
                 "the product not working as intended.",
    },
    "dissolution_min": {
        "label": "Dissolution (min unit)", "unit": "%", "direction": "higher_better",
        "eta2_product": 0.36, "family": "Performance",
        "plain": "The worst individual tablet in the sample. Protects against a single "
                 "bad unit hiding behind a good average.",
    },
    "tbl_rsd_weight": {
        "label": "Tablet weight RSD", "unit": "%", "direction": "lower_better",
        "eta2_product": 0.073, "family": "Uniformity",
        "plain": "How much tablet weight varies within the batch. High values mean the "
                 "press was not filling dies consistently.",
    },
    "fct_rsd_weight": {
        "label": "Coated weight RSD", "unit": "%", "direction": "lower_better",
        "eta2_product": 0.09, "family": "Uniformity",
        "plain": "Weight variation after coating.",
    },
    "batch_yield": {
        "label": "Batch yield", "unit": "%", "direction": "higher_better",
        "eta2_product": 0.30, "family": "Efficiency",
        "plain": "Share of planned tablets actually released. Low yield means material "
                 "was lost to waste, rejects or setup.",
    },
    "fct_av_hardness": {
        "label": "Hardness (coated, avg)", "unit": "N", "direction": "target",
        "eta2_product": 0.854, "family": "Mechanical",
        "plain": "Tablet crushing strength. Too soft chips in the blister; too hard can "
                 "slow dissolution.",
    },
    "tbl_tensile": {
        "label": "Tensile strength (core)", "unit": "MPa", "direction": "target",
        "eta2_product": 0.70, "family": "Mechanical",
        "plain": "Hardness normalised for tablet size — comparable across formats.",
    },
    "impurities_total": {
        "label": "Total impurities", "unit": "%", "direction": "lower_better",
        "eta2_product": 0.815, "family": "Purity",
        "plain": "Chemical degradation products. Must stay low for shelf-life.",
    },
}

PRIMARY_CQA = "dissolution_av"

# --------------------------------------------------------------------------
# Raw-material attributes (incoming lot testing), grouped by material.
# --------------------------------------------------------------------------
MATERIALS = {
    "API": {
        "lot_col": "api_batch", "supplier_col": "api_code",
        "attrs": ["api_water", "api_content", "api_total_impurities",
                  "api_l_impurity", "api_ps01", "api_ps05", "api_ps09"],
    },
    "SMCC": {
        "lot_col": "smcc_batch", "supplier_col": None,
        "attrs": ["smcc_water", "smcc_td", "smcc_bd",
                  "smcc_ps01", "smcc_ps05", "smcc_ps09"],
    },
    "Lactose": {
        "lot_col": "lactose_batch", "supplier_col": None,
        "attrs": ["lactose_water", "lactose_sieve0045",
                  "lactose_sieve015", "lactose_sieve025"],
    },
    "Starch": {
        "lot_col": "starch_batch", "supplier_col": None,
        "attrs": ["starch_ph", "starch_water"],
    },
}

RM_ATTRS = [a for m in MATERIALS.values() for a in m["attrs"]]
LOT_COLS = [m["lot_col"] for m in MATERIALS.values()]

# Attributes with heavy missingness — excluded from modelling by default and
# flagged in the UI rather than silently imputed.
LOW_COVERAGE_ATTRS = {"api_l_impurity"}          # ~36% missing (measured)

# --------------------------------------------------------------------------
# Process features supplied in Process.csv (already derived from the 10 s series)
# --------------------------------------------------------------------------
PROCESS_FEATURES = [
    "tbl_speed_mean", "tbl_speed_change", "tbl_speed_0_duration",
    "fom_mean", "fom_change",
    "SREL_startup_mean", "SREL_production_mean", "SREL_production_max",
    "main_CompForce mean", "main_CompForce_sd", "main_CompForce_median",
    "pre_CompForce_mean",
    "tbl_fill_mean", "tbl_fill_sd", "cyl_height_mean",
    "stiffness_mean", "stiffness_max", "stiffness_min",
    "ejection_mean", "ejection_max", "ejection_min",
    "Startup_tbl_fill_maxDifference", "Startup_main_CompForce_mean",
    "Startup_tbl_fill_mean",
    "total_waste", "startup_waste",
]

# Human labels + plain-English meaning for every driver we may surface.
FEATURE_META: dict[str, dict] = {
    # --- process ---
    "tbl_speed_mean": ("Press speed (mean)", "tablets/h", "How fast the press ran."),
    "tbl_speed_change": ("Press speed changes", "count", "How often the operator changed speed — a proxy for an unsettled run."),
    "tbl_speed_0_duration": ("Press stopped time", "s", "Total time the press was halted mid-run."),
    "fom_mean": ("Force-of-main (mean)", "%", "Average main compression loading."),
    "fom_change": ("Force-of-main changes", "count", "Adjustments to compression setpoint."),
    "SREL_startup_mean": ("Weight RSD at startup", "%", "Mass variability while the press was stabilising."),
    "SREL_production_mean": ("Weight RSD in production", "%", "In-process tablet mass variability during steady state."),
    "SREL_production_max": ("Weight RSD peak", "%", "Worst in-process mass variability seen."),
    "main_CompForce mean": ("Main compression force", "kN", "Force used to form the tablet."),
    "main_CompForce_sd": ("Compression force variability", "kN", "Instability in compression force — often a die-fill symptom."),
    "main_CompForce_median": ("Main compression force (median)", "kN", "Robust centre of compression force."),
    "pre_CompForce_mean": ("Pre-compression force", "kN", "Gentle pre-squeeze that de-aerates the powder."),
    "tbl_fill_mean": ("Die fill depth", "mm", "How deep the die was filled with powder."),
    "tbl_fill_sd": ("Die fill variability", "mm", "Inconsistency in die filling — a direct driver of weight variation."),
    "cyl_height_mean": ("Tablet thickness setting", "mm", "Punch gap controlling tablet thickness."),
    "stiffness_mean": ("Powder stiffness (mean)", "N/mm", "Resistance of the powder bed to compaction."),
    "stiffness_max": ("Powder stiffness (max)", "N/mm", "Peak compaction resistance."),
    "stiffness_min": ("Powder stiffness (min)", "N/mm", "Lowest compaction resistance."),
    "ejection_mean": ("Ejection force (mean)", "N", "Force needed to push the tablet out of the die — rises with sticking/lubrication issues."),
    "ejection_max": ("Ejection force (max)", "N", "Peak ejection force."),
    "ejection_min": ("Ejection force (min)", "N", "Lowest ejection force."),
    "Startup_tbl_fill_maxDifference": ("Startup fill swing", "mm", "Largest die-fill excursion during startup."),
    "Startup_main_CompForce_mean": ("Startup compression force", "kN", "Compression force while stabilising."),
    "Startup_tbl_fill_mean": ("Startup die fill", "mm", "Die fill during startup."),
    "total_waste": ("Total waste", "tablets", "Tablets rejected across the run."),
    "startup_waste": ("Startup waste", "tablets", "Tablets rejected before steady state."),
    # --- raw material ---
    "api_water": ("API moisture", "%", "Water content of the active ingredient lot."),
    "api_content": ("API assay", "%", "Measured potency of the API lot."),
    "api_total_impurities": ("API total impurities", "%", "Impurity burden of the incoming API lot."),
    "api_l_impurity": ("API impurity L", "%", "Specific impurity in the API lot."),
    "api_ps01": ("API particle size D10", "µm", "Size below which 10% of API particles fall — fines fraction."),
    "api_ps05": ("API particle size D50", "µm", "Median API particle size."),
    "api_ps09": ("API particle size D90", "µm", "Coarse end of the API distribution."),
    "smcc_water": ("SMCC moisture", "%", "Water content of the cellulose excipient."),
    "smcc_td": ("SMCC tapped density", "g/mL", "Density after settling — affects die filling."),
    "smcc_bd": ("SMCC bulk density", "g/mL", "Loose density — affects how the powder flows into dies."),
    "smcc_ps01": ("SMCC particle size D10", "µm", "Fines fraction of the excipient."),
    "smcc_ps05": ("SMCC particle size D50", "µm", "Median excipient particle size."),
    "smcc_ps09": ("SMCC particle size D90", "µm", "Coarse fraction of the excipient."),
    "lactose_water": ("Lactose moisture", "%", "Water content of the lactose lot."),
    "lactose_sieve0045": ("Lactose sieve 45 µm", "%", "Share passing the finest sieve."),
    "lactose_sieve015": ("Lactose sieve 150 µm", "%", "Mid-fraction of the lactose distribution."),
    "lactose_sieve025": ("Lactose sieve 250 µm", "%", "Coarse fraction of the lactose distribution."),
    "starch_ph": ("Starch pH", "pH", "Acidity of the starch lot."),
    "starch_water": ("Starch moisture", "%", "Water content of the starch lot."),
    # --- derived from raw time series ---
    "ts_main_comp_slope": ("Compression force drift", "kN/h", "Trend in compression force across the run — a drifting press."),
    "ts_main_comp_excursions": ("Force excursions", "count", "Times compression force left its stable band."),
    "ts_time_in_band": ("Time in stable band", "%", "Share of the run with compression force under control."),
    "ts_fill_slope": ("Die fill drift", "mm/h", "Trend in die fill depth across the run."),
    "ts_srel_excursions": ("Weight RSD excursions", "count", "Times in-process weight variability spiked."),
    "ts_srel_mean": ("In-process weight RSD (mean)", "%", "Average tablet mass variability seen by the press itself."),
    "ts_srel_p95": ("In-process weight RSD (p95)", "%", "Near-worst tablet mass variability during the run."),
    "ts_main_comp_cv": ("Compression force CV", "", "Relative variability of compression force across the run."),
    "ts_main_comp_time_in_band": ("Force time in band", "%", "Share of the run with compression force stable."),
    "ts_fill_cv": ("Die fill CV", "", "Relative variability of die fill depth."),
    "ts_fill_excursions": ("Die fill excursions", "count", "Times die fill left its stable band."),
    "ts_fill_time_in_band": ("Die fill time in band", "%", "Share of the run with die fill stable."),
    "ts_running_share": ("Running share", "%", "Fraction of recorded time the press was actually producing."),
    "ts_longest_stop_s": ("Longest stop", "s", "Duration of the single longest press stoppage."),
    "ts_n_samples": ("Samples recorded", "count", "Number of 10-second records for the run."),
    "waste_rate_pct": ("Waste rate", "%", "Waste normalised by planned batch size."),
    "startup_waste_rate_pct": ("Startup waste rate", "%", "Startup waste normalised by planned batch size."),
    "ts_run_hours": ("Run duration", "h", "Length of the compression run."),
    "ts_stop_count": ("Press stops", "count", "Number of distinct press stoppages."),
    "ts_restart_recovery": ("Restart recovery", "s", "Time to regain stable force after the longest stop."),
}


def feat_label(col: str) -> str:
    return FEATURE_META.get(col, (col, "", ""))[0]


def feat_unit(col: str) -> str:
    return FEATURE_META.get(col, (col, "", ""))[1]


def feat_plain(col: str) -> str:
    return FEATURE_META.get(col, (col, "", ""))[2]


def feat_domain(col: str) -> str:
    """Which layer of the manufacturing chain a driver belongs to."""
    if col in RM_ATTRS:
        for name, m in MATERIALS.items():
            if col in m["attrs"]:
                return f"Raw material · {name}"
    if col.startswith("ts_"):
        return "Process · time series"
    return "Process · compression"


# --------------------------------------------------------------------------
# Same-quantity guard  (prevents tautological "root causes")
# --------------------------------------------------------------------------
# Some recorded signals measure the SAME physical quantity as a quality outcome,
# just with a different instrument or at a different moment. SREL is tablet mass
# variability measured in-process by the press; tbl_rsd_weight is tablet mass
# variability measured in the lab. Correlating them yields r≈0.80 and means
# nothing causally — the press is not the reason the tablets varied, it is simply
# a second witness to the same event.
#
# These signals are genuinely valuable, but as EARLY DETECTION (visible hours
# before lab release), not as root causes. The app therefore separates them into
# a "confirmatory in-process signal" lane instead of ranking them as drivers.
PROXY_SIGNALS: dict[str, list[str]] = {
    "tbl_rsd_weight": ["SREL_production_mean", "SREL_production_max",
                       "SREL_startup_mean", "ts_srel_mean", "ts_srel_p95",
                       "ts_srel_excursions"],
    "fct_rsd_weight": ["SREL_production_mean", "SREL_production_max",
                       "SREL_startup_mean", "ts_srel_mean", "ts_srel_p95",
                       "ts_srel_excursions", "tbl_rsd_weight"],
    # Hardness and tensile strength are two expressions of the same compaction
    # result; tensile is hardness normalised for tablet geometry.
    "fct_av_hardness": ["tbl_tensile", "fct_tensile", "tbl_av_hardness"],
    "tbl_tensile": ["fct_av_hardness", "tbl_av_hardness", "fct_tensile"],
    # Yield is largely waste by construction.
    "batch_yield": ["total_waste", "startup_waste", "tbl_yield",
                    "waste_rate_pct", "startup_waste_rate_pct"],
}


def proxy_signals(cqa: str) -> list[str]:
    return PROXY_SIGNALS.get(cqa, [])


# --------------------------------------------------------------------------
# Analytical policy
# --------------------------------------------------------------------------
# Peer cohort: batches are only ever compared to structurally similar batches.
# Measured justification: 22 of 44 candidate drivers reverse their correlation
# sign between pooled and within-product analysis (Simpson's paradox), so pooled
# ranking is not merely noisy — it can be directionally wrong.
COHORT_KEYS = ["code"]              # product code == strength + batch size + format
COHORT_FALLBACK = ["strength"]
MIN_COHORT_N = 20                   # below this, fall back and warn

# Review-by-exception thresholds (robust z on peer cohort)
Z_WATCH = 2.0
Z_INVESTIGATE = 3.0

# Golden cohort definition
GOLDEN_QUANTILE = 0.25              # best quartile on the selected CQA
POOR_QUANTILE = 0.25                # worst quartile
