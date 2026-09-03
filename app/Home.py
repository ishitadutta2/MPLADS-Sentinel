"""
MPLADS Sentinel — Home / National Oversight Dashboard.

Entry point (`streamlit run app/Home.py`). Gives a national-level view
across all MPs/constituencies: total funds tracked, risk distribution,
top flagged projects, and state-wise risk concentration — the view a
central monitoring cell (e.g. Ministry of Statistics & Programme
Implementation) would want first.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import plotly.express as px  # noqa: F401 (imported lazily below if available)
import streamlit as st

from common import (
    get_scored_dataset, inject_base_style, risk_pill, RISK_COLORS, require_login,
    show_user_badge, scoped_query, apply_plot_theme, render_html_table,
)

st.set_page_config(
    page_title="MPLADS Sentinel — National Dashboard",
    page_icon="🛡️",
    layout="wide",
)
inject_base_style()
user = require_login()
show_user_badge()

st.title("🛡️ MPLADS Sentinel")
st.caption(
    "AI-assisted transparency & fraud-detection layer for the Members of Parliament "
    "Local Area Development Scheme, built around the real eSAKSHI process (recommendation → "
    "sanction → Implementing Agency execution → payment → mark-complete). "
    "**This dashboard runs on a synthetic demo dataset** so detection logic can be shown "
    "without attaching fabricated risk scores to real, named MPs — see **National Allocation "
    "Overview** in the sidebar for the real, published eSAKSHI entitlement data."
)

with st.spinner("Scoring all projects across the five risk engines..."):
    df = get_scored_dataset()

if user["role"] != "central_admin":
    df = scoped_query(df, user)
    st.info(
        f"Showing your scoped view ({len(df)} projects) — your account's role restricts what this "
        f"dashboard can show you, the same way it would for a real MP office or district officer login."
    )

# ---------------------------------------------------------------------------
# Top-line KPIs
# ---------------------------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Projects Tracked", f"{len(df):,}")
col2.metric("Total Sanctioned", f"₹{df['sanctioned_amount'].sum()/1e7:,.1f} Cr")
col3.metric("High + Critical Risk", f"{(df.risk_band.isin(['HIGH','CRITICAL'])).sum():,}")
col4.metric("Amount at Risk", f"₹{df.loc[df.risk_band.isin(['HIGH','CRITICAL']), 'sanctioned_amount'].sum()/1e7:,.1f} Cr")
col5.metric("MPs Covered", f"{df['mp_id'].nunique()}")

st.divider()

left, right = st.columns([1, 1.4])

with left:
    st.subheader("Risk Band Distribution")
    band_counts = df["risk_band"].value_counts().reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"]).fillna(0)
    fig = px.bar(
        x=band_counts.index, y=band_counts.values,
        color=band_counts.index,
        color_discrete_map=RISK_COLORS,
        labels={"x": "Risk Band", "y": "Number of Projects"},
    )
    fig.update_layout(showlegend=False, height=340)
    fig = apply_plot_theme(fig)
    st.plotly_chart(fig, width='stretch')

with right:
    st.subheader("Risk Concentration by State")
    state_risk = df.groupby("state").agg(
        n_projects=("project_id", "count"),
        n_high_critical=("risk_band", lambda s: s.isin(["HIGH", "CRITICAL"]).sum()),
        avg_score=("composite_score", "mean"),
    ).reset_index()
    state_risk["pct_flagged"] = (state_risk["n_high_critical"] / state_risk["n_projects"] * 100).round(1)
    fig2 = px.bar(
        state_risk.sort_values("pct_flagged", ascending=True),
        x="pct_flagged", y="state", orientation="h",
        color="pct_flagged", color_continuous_scale="OrRd",
        labels={"pct_flagged": "% Projects Flagged HIGH/CRITICAL", "state": ""},
    )
    fig2.update_layout(height=340, coloraxis_showscale=False)
    fig2 = apply_plot_theme(fig2)
    st.plotly_chart(fig2, width='stretch')

st.divider()

# ---------------------------------------------------------------------------
# Engine-level breakdown — shows this is 5 real signals, not one black box
# ---------------------------------------------------------------------------
st.subheader("Signal Breakdown Across the Five Detection Engines")
engine_cols = ["financial_score", "cost_benchmark_score", "geo_photo_score",
               "contractor_network_score", "document_score"]
engine_labels = {
    "financial_score": "Financial Anomaly (IsolationForest + rules)",
    "cost_benchmark_score": "Cost vs. Peer Benchmark",
    "geo_photo_score": "Geo & Visual Evidence Consistency",
    "contractor_network_score": "Contractor Network / Cartel Detection",
    "document_score": "Document & Transaction Consistency",
}
avg_by_engine = df[engine_cols].mean().rename(index=engine_labels).reset_index()
avg_by_engine.columns = ["Engine", "Average Score (0-100)"]
fig3 = px.bar(avg_by_engine, x="Average Score (0-100)", y="Engine", orientation="h", height=280)
fig3.update_traces(marker_color="#1565c0")
fig3 = apply_plot_theme(fig3)
st.plotly_chart(fig3, width='stretch')

st.divider()

# ---------------------------------------------------------------------------
# Top flagged projects table
# ---------------------------------------------------------------------------
st.subheader("Highest-Risk Projects Nationally")
top = df.sort_values("composite_score", ascending=False).head(25)[
    ["project_id", "mp_name", "state", "district", "category", "contractor_name",
     "sanctioned_amount", "composite_score", "risk_band"]
].copy()
top["sanctioned_amount"] = top["sanctioned_amount"].apply(lambda v: f"₹{v:,.0f}")
top["risk_band"] = top["risk_band"].apply(risk_pill)
render_html_table(top.to_html(escape=False, index=False))

st.caption(
    "Click into **MP Dashboard**, **District Officer Queue**, or search a project ID directly "
    "in **Project Detail** from the sidebar to drill down."
)
