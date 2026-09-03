"""
Citizen Reporting — a lightweight public-facing intake form (citizens
report a problem with a project they can see in real life) plus a viewer
showing how citizen reports correlate with Sentinel's own risk scoring —
demonstrating the "citizen verification loop" the brief calls for.
"""
import sys
from pathlib import Path
import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from common import (
    get_scored_dataset, get_raw_tables, inject_base_style, risk_pill, require_login,
    show_user_badge, render_html_table, apply_plot_theme, RISK_COLORS,
)
from sentinel.db import get_conn
from sentinel.audit.hash_chain import append_event

st.set_page_config(page_title="Citizen Reporting — MPLADS Sentinel", page_icon="🗣️", layout="wide")
inject_base_style()
user = require_login()
show_user_badge()
st.title("🗣️ Citizen Reporting")
st.caption(
    "Any citizen can report a discrepancy between what was funded and what actually exists on the ground. "
    "Reports feed directly into officer review queues and correlate strongly with Sentinel's own risk scores."
)

df = get_scored_dataset()
raw = get_raw_tables()

tab1, tab2 = st.tabs(["📝 Submit a Report", "📊 Reports Overview"])

with tab1:
    if st.session_state.get("_last_citizen_report_id"):
        st.success(
            f"Report submitted and logged (Report ID: {st.session_state['_last_citizen_report_id']}). "
            "Thank you for helping keep public funds accountable. It now appears in the 'Reports Overview' tab."
        )
        del st.session_state["_last_citizen_report_id"]

    with st.form("citizen_report_form"):
        project_id = st.text_input("Project ID (find this on the MPLADS project signboard, e.g. PRJ00363)")
        complaint_type = st.selectbox("What's the issue?", [
            "Work not started despite funds released",
            "Poor quality of construction material",
            "Work incomplete but marked complete",
            "No such facility exists at stated location",
            "Overpricing compared to similar local works",
            "Other",
        ])
        details = st.text_area("Additional details (optional)")
        citizen_name = st.text_input("Your name (optional — will be masked in public records)")
        submitted = st.form_submit_button("Submit Report")

        if submitted:
            pid = project_id.strip().upper()
            if not pid:
                st.error("Please enter a Project ID.")
            else:
                with get_conn() as conn:
                    masked = f"Citizen_{abs(hash(citizen_name or 'anon')) % 9000 + 1000}"
                    report_id = f"CIT_NEW_{int(datetime.datetime.now(datetime.timezone.utc).timestamp())}"
                    conn.execute(
                        "INSERT INTO citizen_reports (report_id, project_id, complaint_type, report_date, citizen_name_masked, status) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (report_id, pid, complaint_type, datetime.date.today().isoformat(), masked, "Open"),
                    )
                append_event(
                    event_type="CITIZEN_REPORT",
                    project_id=pid,
                    actor=masked,
                    details={"complaint_type": complaint_type, "details": details},
                )
                st.cache_data.clear()
                # A previous version showed st.success() here directly, but
                # Streamlit's tabs don't re-run the script on their own —
                # clicking over to "Reports Overview" right after submitting
                # would still be looking at the pre-submission dataframe
                # that was fetched at the top of THIS run, before the
                # INSERT happened, making the new report look like it never
                # saved even though it genuinely had. Stash the id and
                # force a real rerun so the rest of the page re-fetches
                # fresh data (see the success message above, which reads
                # this back) with the new report already in it.
                st.session_state["_last_citizen_report_id"] = report_id
                st.rerun()

with tab2:
    citizen = raw["citizen_reports"]
    st.metric("Total Citizen Reports on File", len(citizen))

    merged = citizen.merge(
        df[["project_id", "composite_score", "risk_band", "category", "district", "mp_name"]],
        on="project_id", how="left",
    )

    st.subheader("Correlation With Sentinel Risk Scoring")
    st.caption("Projects with citizen complaints tend to already carry a higher Sentinel risk score — validating that both signals point at the same underlying problems.")
    corr_view = merged.groupby("risk_band")["report_id"].count().reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"]).fillna(0)
    fig = px.bar(
        x=corr_view.index, y=corr_view.values, color=corr_view.index,
        color_discrete_map=RISK_COLORS, labels={"x": "Risk Band", "y": "Citizen Reports"},
    )
    fig.update_layout(showlegend=False, height=320)
    fig = apply_plot_theme(fig)
    st.plotly_chart(fig, width='stretch')

    st.subheader("All Reports")
    view = merged.sort_values("composite_score", ascending=False)[
        ["report_id", "project_id", "complaint_type", "district", "mp_name", "status", "composite_score", "risk_band"]
    ].copy()
    view["risk_band"] = view["risk_band"].fillna("LOW").apply(risk_pill)
    render_html_table(view.to_html(escape=False, index=False))
