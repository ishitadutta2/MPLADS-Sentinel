"""
MPLADS Sentinel — central configuration.

Everything that a deployer might want to tune lives here so the rest of
the codebase never hard-codes a magic number.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PHOTO_DIR = BASE_DIR / "sample_photos"
# Real, user-uploaded evidence photos live in a SEPARATE directory from the
# synthetic demo corpus (PHOTO_DIR) — kept apart on purpose so it's always
# visually/structurally obvious which photos are this demo's fake data and
# which are something a real person actually uploaded through the app.
UPLOADED_PHOTO_DIR = BASE_DIR / "uploaded_evidence"

# --- Database -----------------------------------------------------------
# SQLite by default (zero-config, ships with the repo). Point DATABASE_URL
# at a real Postgres/PostGIS instance in production — the ORM models are
# written to be dialect-agnostic, so no code changes are required, only
# this env var.
DATABASE_URL = os.environ.get(
    "SENTINEL_DATABASE_URL", f"sqlite:///{DATA_DIR / 'sentinel.db'}"
)

# --- Risk engine weights (must sum to 1.0) -------------------------------
RISK_WEIGHTS = {
    "financial": 0.30,   # cost/tender/vendor anomalies
    "cost_benchmark": 0.15,  # unit-cost deviation vs district benchmark
    "geo_photo": 0.20,   # geo-tagging & visual-consistency failures
    "contractor_network": 0.20,  # shell/cartel network signals
    "document": 0.15,   # invoice/UC/measurement-book mismatches
}
assert abs(sum(RISK_WEIGHTS.values()) - 1.0) < 1e-6

RISK_BANDS = [
    (0, 20, "LOW", "#2e7d32"),
    (20, 32, "MEDIUM", "#f9a825"),
    (32, 45, "HIGH", "#ef6c00"),
    (45, 101, "CRITICAL", "#c62828"),
]

def risk_band(score: float):
    for lo, hi, label, color in RISK_BANDS:
        if lo <= score < hi:
            return label, color
    return "CRITICAL", "#c62828"

# --- Geo tolerance --------------------------------------------------------
GEO_TOLERANCE_METERS = 500  # photo GPS vs. declared project site
CONSTITUENCY_RADIUS_KM = 60  # sanity radius for "is this even in-district"

# --- Real-world reference date (for guideline-based overdue checks) -----
# Per the eSAKSHI process: "sanctioned works are generally required to be
# completed within one year of sanction." REPORT_AS_OF_DATE is the date
# this dashboard is being "read" as of, used to check that guideline.
REPORT_AS_OF_DATE = "2026-08-29"
SANCTION_TO_COMPLETION_GUIDELINE_DAYS = 365

# --- Synthetic data generation -------------------------------------------
RANDOM_SEED = 42
N_MPS = 25
N_PROJECTS_PER_MP = 40
N_CONTRACTORS = 120
ANOMALY_RATE = 0.12  # fraction of projects deliberately seeded as irregular

# --- Audit log --------------------------------------------------------
AUDIT_LOG_PATH = DATA_DIR / "audit_chain.jsonl"
