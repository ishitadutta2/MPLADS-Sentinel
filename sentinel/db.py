"""
Database layer for MPLADS Sentinel.

Uses plain stdlib `sqlite3` + `pandas` rather than an ORM — this keeps the
dependency footprint tiny (no SQLAlchemy/driver install needed) while the
schema itself is written in portable ANSI SQL so migrating the same DDL to
PostgreSQL/PostGIS for a production deployment is a copy-paste job (swap
`AUTOINCREMENT` -> `SERIAL`, add a `geography(Point)` column for true
spatial indexing/queries).
"""
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from sentinel.config import DATABASE_URL, DATA_DIR

DB_PATH = DATABASE_URL.replace("sqlite:///", "")

SCHEMA = """
CREATE TABLE IF NOT EXISTS mps (
    mp_id TEXT PRIMARY KEY,
    mp_name TEXT, constituency TEXT, state TEXT, party TEXT,
    term_start TEXT, fund_allocated_total REAL
);

CREATE TABLE IF NOT EXISTS contractors (
    contractor_id TEXT PRIMARY KEY,
    contractor_name TEXT, registration_address TEXT, bank_account TEXT,
    director_name TEXT, registration_year INTEGER,
    is_cartel_seed INTEGER, cartel_cluster TEXT
);

CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY,
    mp_id TEXT REFERENCES mps(mp_id),
    constituency TEXT, state TEXT, district TEXT, category TEXT,
    description TEXT, sanctioned_amount REAL, expenditure REAL,
    unit_cost_basis REAL, status TEXT,
    recommendation_date TEXT, sanction_date TEXT,
    implementing_agency TEXT, marked_complete_by_ia INTEGER,
    contractor_id TEXT REFERENCES contractors(contractor_id),
    site_latitude REAL, site_longitude REAL,
    photo_latitude REAL, photo_longitude REAL,
    is_seeded_anomalous INTEGER, seeded_anomaly_types TEXT
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id TEXT PRIMARY KEY,
    project_id TEXT REFERENCES projects(project_id),
    contractor_id TEXT, amount REAL, txn_date TEXT, installment_no INTEGER
);

CREATE TABLE IF NOT EXISTS photos (
    photo_id TEXT PRIMARY KEY,
    project_id TEXT REFERENCES projects(project_id),
    category TEXT, latitude REAL, longitude REAL, capture_date TEXT,
    is_seeded_duplicate INTEGER, filename TEXT
);

CREATE TABLE IF NOT EXISTS citizen_reports (
    report_id TEXT PRIMARY KEY,
    project_id TEXT REFERENCES projects(project_id),
    complaint_type TEXT, report_date TEXT, citizen_name_masked TEXT,
    status TEXT
);

CREATE TABLE IF NOT EXISTS risk_scores (
    project_id TEXT PRIMARY KEY REFERENCES projects(project_id),
    financial_score REAL, cost_benchmark_score REAL, geo_photo_score REAL,
    contractor_network_score REAL, document_score REAL,
    composite_score REAL, risk_band TEXT, explanation TEXT,
    scored_at TEXT, ml_risk_probability REAL, ml_top_signal TEXT
);

CREATE TABLE IF NOT EXISTS officer_feedback (
    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT REFERENCES projects(project_id),
    officer_name TEXT, verdict TEXT, notes TEXT, submitted_at TEXT
);

CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('mp', 'district_officer', 'central_admin')),
    -- scope ties an account to what it's allowed to see: an MP account is
    -- scoped to one mp_id, a district officer to one district, central
    -- admin is unscoped (NULL = sees everything). Enforced in
    -- sentinel/auth.py, not just decorative.
    scope_mp_id TEXT,
    scope_district TEXT,
    password_hash TEXT NOT NULL,
    password_salt TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS login_audit (
    login_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    event TEXT CHECK(event IN ('LOGIN_SUCCESS', 'LOGIN_FAILURE')),
    at TEXT NOT NULL
);

-- Real evidence photos a logged-in person actually uploaded through the
-- app (as opposed to `photos`, which is this demo's synthetic corpus and
-- gets dropped-and-reloaded wholesale from CSV every time
-- load_csvs_into_db() runs — see the note there). Kept as its own table
-- so it survives that reload and behaves like real persisted data.
CREATE TABLE IF NOT EXISTS uploaded_evidence (
    upload_id TEXT PRIMARY KEY,
    -- Deliberately NOT "REFERENCES projects(project_id)": that table gets
    -- dropped and recreated by pandas.to_sql(if_exists="replace") on every
    -- load_csvs_into_db() call, which doesn't preserve the PRIMARY KEY
    -- constraint from SCHEMA above — so a real FK constraint here would
    -- fail with "foreign key mismatch" (SQLite requires the referenced
    -- column to currently be indexed unique, which projects.project_id
    -- stops being the moment it's reloaded). Every other table in this
    -- schema that "references" projects has the exact same gap; they just
    -- never hit the error because they're ALSO recreated by pandas on
    -- every load, which drops their own FK clause along with everything
    -- else. This table isn't, so it can't rely on the same accident.
    project_id TEXT,
    declared_category TEXT,
    filename TEXT NOT NULL,
    uploaded_by TEXT,
    uploaded_at TEXT NOT NULL,
    visual_consistency REAL,
    consistency_fail INTEGER,
    is_duplicate INTEGER,
    duplicate_of_project TEXT,
    duplicate_of_photo TEXT,
    geo_source TEXT,
    geo_lat REAL,
    geo_lon REAL,
    geo_distance_m REAL,
    geo_fail INTEGER
);

-- Indices on the columns every dashboard page actually filters/joins on.
-- On SQLite these matter most once the officer queue and per-MP views
-- run against a much larger project count than this demo's 1,000 rows;
-- they cost nothing at this scale but are the difference between an
-- index scan and a full table scan once this is real data.
-- Kept as a SEPARATE script (not inline in SCHEMA) because
-- load_csvs_into_db() reloads several of these tables via
-- pandas.to_sql(if_exists="replace"), which drops and recreates the
-- table — wiping any index defined inline in CREATE TABLE. Re-running
-- INDEX_SCHEMA after that reload is what actually keeps the indices in
-- place; see load_csvs_into_db() below.
"""

INDEX_SCHEMA = """
CREATE INDEX IF NOT EXISTS idx_projects_mp_id ON projects(mp_id);
CREATE INDEX IF NOT EXISTS idx_projects_contractor_id ON projects(contractor_id);
CREATE INDEX IF NOT EXISTS idx_projects_district ON projects(district);
CREATE INDEX IF NOT EXISTS idx_transactions_project_id ON transactions(project_id);
CREATE INDEX IF NOT EXISTS idx_photos_project_id ON photos(project_id);
CREATE INDEX IF NOT EXISTS idx_citizen_reports_project_id ON citizen_reports(project_id);
CREATE INDEX IF NOT EXISTS idx_risk_scores_risk_band ON risk_scores(risk_band);
CREATE INDEX IF NOT EXISTS idx_risk_scores_composite ON risk_scores(composite_score);
CREATE INDEX IF NOT EXISTS idx_uploaded_evidence_project_id ON uploaded_evidence(project_id);
"""


@contextmanager
def get_conn():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # WAL mode lets readers (dashboard queries) proceed concurrently with
    # a writer (a scoring run or an officer submitting feedback) instead
    # of blocking on SQLite's default single-writer lock — the realistic
    # concurrency pattern for this app even before any Postgres migration.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_schema():
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        conn.executescript(INDEX_SCHEMA)


def load_csvs_into_db():
    """Idempotently (re)loads the CSVs in data/ into the SQLite tables.

    citizen_reports is handled separately from the rest of table_files:
    unlike those tables (pure reference/demo data — always safe to fully
    replace from the CSV), citizen_reports ALSO receives live INSERTs from
    the Citizen Reporting page's public submission form. This function
    used to replace it wholesale on every single call, same as the others
    — which meant the next time *anything* triggered a cache clear and a
    fresh load (e.g. right after someone submitted a report, since that
    submission itself calls st.cache_data.clear()), every citizen report
    a real person had ever submitted through the app was silently
    discarded and replaced with just the static demo baseline again. So
    it's seeded from the CSV once — only if the table doesn't exist yet
    or is genuinely empty — and left alone after that.
    """
    init_schema()
    table_files = {
        "mps": "mps.csv",
        "contractors": "contractors.csv",
        "projects": "projects.csv",
        "transactions": "transactions.csv",
        "photos": "photos.csv",
    }
    with get_conn() as conn:
        for table, fname in table_files.items():
            fpath = DATA_DIR / fname
            if not fpath.exists():
                continue
            df = pd.read_csv(fpath)
            df.to_sql(table, conn, if_exists="replace", index=False)

        existing_reports = conn.execute("SELECT COUNT(*) FROM citizen_reports").fetchone()[0]
        if existing_reports == 0:
            fpath = DATA_DIR / "citizen_reports.csv"
            if fpath.exists():
                df = pd.read_csv(fpath)
                df.to_sql("citizen_reports", conn, if_exists="replace", index=False)

        # to_sql(if_exists="replace") drops and recreates each table above,
        # which drops any index on it — re-apply so the indices actually
        # persist past a reload.
        conn.executescript(INDEX_SCHEMA)


def read_table(table_name: str) -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql(f"SELECT * FROM {table_name}", conn)


def upsert_risk_scores(df: pd.DataFrame):
    with get_conn() as conn:
        df.to_sql("risk_scores", conn, if_exists="replace", index=False)


def insert_feedback(project_id: str, officer_name: str, verdict: str, notes: str):
    import datetime
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO officer_feedback (project_id, officer_name, verdict, notes, submitted_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (project_id, officer_name, verdict, notes, datetime.datetime.now(datetime.timezone.utc).isoformat()),
        )


def read_feedback() -> pd.DataFrame:
    with get_conn() as conn:
        try:
            return pd.read_sql("SELECT * FROM officer_feedback", conn)
        except Exception:
            return pd.DataFrame()


def insert_uploaded_evidence(record: dict):
    """Persists one real, user-uploaded evidence photo's metadata. The
    actual image bytes are saved to UPLOADED_PHOTO_DIR by the caller
    (app/pages/4_Photo_Verification.py) — this just records the row so it
    survives reruns/restarts and can be listed and deleted later."""
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO uploaded_evidence ("
            "upload_id, project_id, declared_category, filename, uploaded_by, uploaded_at, "
            "visual_consistency, consistency_fail, is_duplicate, duplicate_of_project, duplicate_of_photo, "
            "geo_source, geo_lat, geo_lon, geo_distance_m, geo_fail"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record["upload_id"], record["project_id"], record["declared_category"], record["filename"],
                record["uploaded_by"], record["uploaded_at"],
                record.get("visual_consistency"), record.get("consistency_fail"),
                record.get("is_duplicate"), record.get("duplicate_of_project"), record.get("duplicate_of_photo"),
                record.get("geo_source"), record.get("geo_lat"), record.get("geo_lon"),
                record.get("geo_distance_m"), record.get("geo_fail"),
            ),
        )


def read_uploaded_evidence(project_id: str = None) -> pd.DataFrame:
    with get_conn() as conn:
        if project_id:
            return pd.read_sql(
                "SELECT * FROM uploaded_evidence WHERE project_id = ? ORDER BY uploaded_at DESC",
                conn, params=(project_id,),
            )
        return pd.read_sql("SELECT * FROM uploaded_evidence ORDER BY uploaded_at DESC", conn)


def delete_uploaded_evidence(upload_id: str):
    """Deletes the DB row only — the caller is responsible for removing
    the underlying file from UPLOADED_PHOTO_DIR, since this module
    deliberately doesn't do filesystem I/O."""
    with get_conn() as conn:
        conn.execute("DELETE FROM uploaded_evidence WHERE upload_id = ?", (upload_id,))
