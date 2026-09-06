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
    show_user_badge, scoped_query, apply_plot_theme, render_html_table, page_header,
)
from i18n import t

st.set_page_config(
    page_title="MPLADS Sentinel — National Dashboard",
    page_icon="🛡️",
    layout="wide",
)
inject_base_style()
user = require_login()
show_user_badge()

page_header("🛡️", t("home_title"), t("home_desc"))

with st.spinner(t("scoring_spinner")):
    df = get_scored_dataset()

if user["role"] != "central_admin":
    df = scoped_query(df, user)
    st.info(t("scoped_view_info", n=len(df)))

# ---------------------------------------------------------------------------
# Top-line KPIs
# ---------------------------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric(t("kpi_projects_tracked"), f"{len(df):,}")
col2.metric(t("kpi_total_sanctioned"), f"₹{df['sanctioned_amount'].sum()/1e7:,.1f} Cr")
col3.metric(t("kpi_high_critical"), f"{(df.risk_band.isin(['HIGH','CRITICAL'])).sum():,}")
col4.metric(t("kpi_amount_at_risk"), f"₹{df.loc[df.risk_band.isin(['HIGH','CRITICAL']), 'sanctioned_amount'].sum()/1e7:,.1f} Cr")
col5.metric(t("kpi_mps_covered"), f"{df['mp_id'].nunique()}")

st.divider()

left, right = st.columns([1, 1.4])

with left:
    st.subheader(t("risk_band_distribution"))
    band_counts = df["risk_band"].value_counts().reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"]).fillna(0)
    fig = px.bar(
        x=band_counts.index, y=band_counts.values,
        color=band_counts.index,
        color_discrete_map=RISK_COLORS,
        labels={"x": t("axis_risk_band"), "y": t("axis_num_projects")},
    )
    fig.update_layout(showlegend=False, height=340)
    fig = apply_plot_theme(fig)
    st.plotly_chart(fig, width='stretch')

with right:
    st.subheader(t("risk_concentration_state"))
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
        labels={"pct_flagged": t("axis_pct_flagged"), "state": ""},
    )
    fig2.update_layout(height=340, coloraxis_showscale=False)
    fig2 = apply_plot_theme(fig2)
    st.plotly_chart(fig2, width='stretch')

st.divider()

# ---------------------------------------------------------------------------
# Engine-level breakdown — shows this is 5 real signals, not one black box
# ---------------------------------------------------------------------------
st.subheader(t("signal_breakdown"))
engine_cols = ["financial_score", "cost_benchmark_score", "geo_photo_score",
               "contractor_network_score", "document_score"]
engine_labels = {
    "financial_score": t("engine_financial"),
    "cost_benchmark_score": t("engine_cost_benchmark"),
    "geo_photo_score": t("engine_geo_photo"),
    "contractor_network_score": t("engine_contractor_network"),
    "document_score": t("engine_document"),
}
avg_by_engine = df[engine_cols].mean().rename(index=engine_labels).reset_index()
avg_by_engine.columns = [t("axis_engine"), t("axis_avg_score")]
fig3 = px.bar(avg_by_engine, x=t("axis_avg_score"), y=t("axis_engine"), orientation="h", height=280)
fig3.update_traces(marker_color="#1565c0")
fig3 = apply_plot_theme(fig3)
st.plotly_chart(fig3, width='stretch')

st.divider()

# ---------------------------------------------------------------------------
# Top flagged projects table
# ---------------------------------------------------------------------------
st.subheader(t("highest_risk_national"))
top = df.sort_values("composite_score", ascending=False).head(25)[
    ["project_id", "mp_name", "state", "district", "category", "contractor_name",
     "sanctioned_amount", "composite_score", "risk_band"]
].copy()
top["sanctioned_amount"] = top["sanctioned_amount"].apply(lambda v: f"₹{v:,.0f}")
top["risk_band"] = top["risk_band"].apply(risk_pill)
top.columns = [t("col_project_id"), t("col_mp_name"), t("col_state"), t("district"), t("category"),
               t("col_contractor"), t("col_sanctioned_amount"), t("col_score"), t("risk")]
render_html_table(top.to_html(escape=False, index=False))

st.caption(t("drill_down_caption"))
