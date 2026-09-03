"""
Scoring pipeline — orchestrates all five engines end-to-end and persists
the composite risk scores to the database. This is what the Streamlit app
calls on first load (and what a nightly cron job would call in
production).
"""
import time

import pandas as pd

from sentinel.db import load_csvs_into_db, read_table, upsert_risk_scores
from sentinel.engines.financial import run_financial_engine
from sentinel.engines.cost import run_cost_benchmark_engine
from sentinel.engines.vision_geo import run_vision_geo_engine
from sentinel.engines.contractor_graph import run_contractor_network_engine
from sentinel.engines.document_intel import run_document_engine
from sentinel.engines.ml_risk_classifier import train_and_score
from sentinel.engines.risk_engine import compute_composite_risk
from sentinel.audit.hash_chain import append_event


def run_full_pipeline(verbose: bool = True) -> pd.DataFrame:
    t0 = time.time()
    load_csvs_into_db()

    projects = read_table("projects")
    transactions = read_table("transactions")
    photos = read_table("photos")
    contractors = read_table("contractors")

    if verbose:
        print(f"Loaded {len(projects)} projects, {len(transactions)} transactions, "
              f"{len(photos)} photos, {len(contractors)} contractors")

    financial_df = run_financial_engine(projects, transactions)
    if verbose:
        print(f"[1/6] Financial engine done ({time.time()-t0:.1f}s)")

    cost_df = run_cost_benchmark_engine(projects)
    if verbose:
        print(f"[2/6] Cost benchmark engine done ({time.time()-t0:.1f}s)")

    geo_df = run_vision_geo_engine(projects, photos)
    if verbose:
        print(f"[3/6] Vision/geo engine done ({time.time()-t0:.1f}s)")

    network_df, _graph = run_contractor_network_engine(projects, contractors)
    if verbose:
        print(f"[4/6] Contractor network engine done ({time.time()-t0:.1f}s)")

    document_df = run_document_engine(projects, transactions)
    if verbose:
        print(f"[5/6] Document engine done ({time.time()-t0:.1f}s)")

    composite = compute_composite_risk(financial_df, cost_df, geo_df, network_df, document_df)

    # --- supervised ML layer: cross-validated risk probability ---------
    # Trained (for this demo) against the seeded ground truth; in
    # production this would train against accumulated officer verdicts —
    # see sentinel/engines/ml_risk_classifier.py for the full rationale
    # and the production training path.
    ml_df = train_and_score(
        {"financial": financial_df, "cost": cost_df, "geo": geo_df,
         "network": network_df, "document": document_df},
        projects,
    )
    composite = composite.set_index("project_id").join(ml_df[["ml_risk_probability", "ml_top_signal"]]).reset_index()
    if verbose:
        print(f"[6/6] ML risk classifier done ({time.time()-t0:.1f}s)")

    # persist a DB-friendly (stringified list columns) copy
    db_copy = composite.copy()
    for col in ["financial_flags", "geo_vision_flags", "network_flags", "document_flags"]:
        db_copy[col] = db_copy[col].apply(lambda l: ",".join(l) if isinstance(l, list) else "")
    db_ready = db_copy[[
        "project_id", "financial_score", "cost_benchmark_score", "geo_photo_score",
        "contractor_network_score", "document_score", "composite_score",
        "risk_band", "explanation", "scored_at", "ml_risk_probability", "ml_top_signal",
    ]]
    upsert_risk_scores(db_ready)

    append_event(
        event_type="PIPELINE_RUN",
        project_id="ALL",
        actor="system",
        details={
            "n_projects_scored": len(composite),
            "n_high_or_critical": int((composite["risk_band"].isin(["HIGH", "CRITICAL"])).sum()),
            "duration_seconds": round(time.time() - t0, 2),
        },
    )

    if verbose:
        print(f"Done in {time.time()-t0:.1f}s. Risk band distribution:")
        print(composite["risk_band"].value_counts())

    return composite


if __name__ == "__main__":
    run_full_pipeline()
