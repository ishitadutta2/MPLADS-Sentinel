"""
Citizen Reporting — a lightweight public-facing intake form (citizens
report a problem with a project they can see in real life) plus a viewer
showing how citizen reports correlate with Sentinel's own risk scoring —
demonstrating the "citizen verification loop" the brief calls for.
"""
import sys
from pathlib import Path
import datetime
import html

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from common import (
    get_scored_dataset, get_raw_tables, inject_base_style, risk_pill, require_login,
    show_user_badge, render_html_table, apply_plot_theme, RISK_COLORS, page_header, svg_icon,
)
from sentinel.db import get_conn
from sentinel.audit.hash_chain import append_event
from i18n import t

st.set_page_config(page_title="Citizen Reporting — MPLADS Sentinel", page_icon=":material/forum:", layout="wide")
inject_base_style()
user = require_login()
show_user_badge()
page_header("chat", t("citizen_title"), t("citizen_sub"))

df = get_scored_dataset()
raw = get_raw_tables()

tab1, tab2 = st.tabs([t("tab_submit_report"), t("tab_reports_overview")])

with tab1:
    if st.session_state.get("_last_citizen_report_id"):
        st.success(t("report_submitted_success", id=st.session_state['_last_citizen_report_id']))
        del st.session_state["_last_citizen_report_id"]

    with st.form("citizen_report_form"):
        project_id = st.text_input(t("project_id_signboard_lbl"))
        complaint_options = [
            t("complaint_not_started"), t("complaint_poor_quality"), t("complaint_incomplete_marked"),
            t("complaint_no_facility"), t("complaint_overpricing"), t("complaint_other"),
        ]
        complaint_type = st.selectbox(t("whats_issue_lbl"), complaint_options)
        details = st.text_area(t("additional_details_lbl"))
        citizen_name = st.text_input(t("your_name_lbl"))
        submitted = st.form_submit_button(t("submit_report_btn"))

        if submitted:
            pid = project_id.strip().upper()
            if not pid:
                st.error(t("please_enter_project_id"))
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
    st.metric(t("kpi_total_citizen_reports"), len(citizen))

    merged = citizen.merge(
        df[["project_id", "composite_score", "risk_band", "category", "district", "mp_name"]],
        on="project_id", how="left",
    )

    st.subheader(t("correlation_header"))
    st.caption(t("correlation_caption"))
    corr_view = merged.groupby("risk_band")["report_id"].count().reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"]).fillna(0)
    fig = px.bar(
        x=corr_view.index, y=corr_view.values, color=corr_view.index,
        color_discrete_map=RISK_COLORS, labels={"x": t("axis_risk_band"), "y": t("axis_citizen_reports")},
    )
    fig.update_layout(showlegend=False, height=320)
    fig = apply_plot_theme(fig)
    st.plotly_chart(fig, width='stretch')

    st.subheader(t("all_reports_header"))
    # channel / details / is_unlisted_project only exist once at least one
    # report has gone through the public Citizen Chatbot's extended
    # schema (see sentinel.db.ensure_citizen_report_columns) — default them
    # in so older/base rows (and a merged frame missing the columns
    # entirely) still display cleanly instead of raising a KeyError.
    for col, default in [("channel", "web"), ("details", ""), ("is_unlisted_project", 0),
                          ("report_state", ""), ("report_district", "")]:
        if col not in merged.columns:
            merged[col] = default
    merged["channel"] = merged["channel"].fillna("web")
    merged["details"] = merged["details"].fillna("")
    merged["is_unlisted_project"] = merged["is_unlisted_project"].fillna(0)
    merged["source_district"] = merged["district"].fillna(merged["report_district"])

    view = merged.sort_values("composite_score", ascending=False)[
        ["report_id", "project_id", "complaint_type", "source_district", "mp_name", "status",
         "composite_score", "risk_band", "channel", "is_unlisted_project", "details"]
    ].copy()
    view["risk_band"] = view["risk_band"].fillna("LOW").apply(risk_pill)
    view["project_id"] = view.apply(
        lambda r: f'{svg_icon("flag", size=13)} {r["project_id"]}' if r["is_unlisted_project"] else r["project_id"], axis=1,
    )
    view["channel"] = view["channel"].apply(
        lambda c: f'{svg_icon("bot", size=14)} Chatbot' if c == "chatbot" else f'{svg_icon("globe", size=14)} Web form'
    )
    # escape=False below is needed so risk_pill()'s HTML badge renders — but
    # that means "Details" (raw citizen-typed text, e.g. from the Citizen
    # Chatbot) must be escaped by hand here first, or a "<" typed by a
    # citizen would be treated as real HTML in an officer's browser.
    view["details"] = view["details"].apply(lambda d: html.escape(d[:80] + "…") if len(d) > 80 else html.escape(d))
    view = view.drop(columns=["is_unlisted_project"])
    view.columns = [t("col_report_id"), t("col_project_id"), t("col_complaint_type"), t("district"),
                     t("col_mp_name"), t("status"), t("col_score"), t("risk"), "Channel", "Details"]
    st.caption(":material/flag: next to a project id means it isn't a project already tracked in the system — "
               "it was reported directly through the Citizen Chatbot and needs an officer to review it.")
    render_html_table(view.to_html(escape=False, index=False))
