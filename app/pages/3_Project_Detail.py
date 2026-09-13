"""Project Detail — full deep-dive on a single project: timeline, photos, contractor links, citizen reports."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
from PIL import Image

from common import get_scored_dataset, get_raw_tables, inject_base_style, risk_pill, require_login, show_user_badge, page_header, render_html_table
from sentinel.config import PHOTO_DIR, UPLOADED_PHOTO_DIR
from sentinel.utils.geo import haversine_meters
from sentinel.db import read_uploaded_evidence
from i18n import t

st.set_page_config(page_title="Project Detail — MPLADS Sentinel", page_icon=":material/search:", layout="wide")
inject_base_style()
user = require_login()
show_user_badge()
page_header("search", t("project_title"))

df = get_scored_dataset()
raw = get_raw_tables()

query_id = st.text_input(t("enter_project_id"), value="")
if not query_id:
    st.info(t("try_critical_tip"))
    st.stop()

matches = df[df["project_id"].str.upper() == query_id.strip().upper()]
if matches.empty:
    st.error(t("no_project_found", id=query_id))
    st.stop()

row = matches.iloc[0]

st.subheader(f"{row.project_id} — {row.category}")
st.markdown(risk_pill(row.risk_band), unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric(t("kpi_composite_score"), f"{row.composite_score:.1f} / 100")
c2.metric(t("col_sanctioned_amount"), f"₹{row.sanctioned_amount:,.0f}")
c3.metric(t("col_expenditure"), f"₹{row.expenditure:,.0f}")
c4.metric(t("status"), row.status)

st.markdown(f"**{t('col_mp_name')}:** {row.mp_name} ({row.constituency})  |  **{t('district')}:** {row.district}, {row.state}")
st.markdown(f"**{t('col_contractor')}:** {row.contractor_name} (`{row.contractor_id}`)")
st.markdown(f"**{t('description')}:** {row.description}")

st.divider()
st.subheader(t("why_score"))
st.code(row.explanation, language=None)

sub1, sub2, sub3, sub4, sub5 = st.columns(5)
sub1.metric(t("short_financial"), f"{row.financial_score:.0f}")
sub2.metric(t("engine_cost_benchmark"), f"{row.cost_benchmark_score:.0f}")
sub3.metric(t("short_geo_photo"), f"{row.geo_photo_score:.0f}")
sub4.metric(t("short_contractor_network"), f"{row.contractor_network_score:.0f}")
sub5.metric(t("short_documents"), f"{row.document_score:.0f}")

if "ml_risk_probability" in row and pd.notna(row.ml_risk_probability):
    st.markdown(f"**{t('ml_model_line')}**")
    ml_col1, ml_col2 = st.columns(2)
    ml_col1.metric(t("ml_prob_label"), f"{row.ml_risk_probability*100:.0f}%")
    ml_col2.metric(t("ml_top_signal_label"), str(row.get("ml_top_signal", "—")).replace("_", " ").title())
    st.caption(t("ml_caption"))

st.divider()
st.subheader(t("evidence_photos"))
photos = raw["photos"]
proj_photos = photos[photos["project_id"] == row.project_id]

if proj_photos.empty:
    st.info(t("no_photos"))
else:
    cols = st.columns(min(len(proj_photos), 4))
    for i, (_, prow) in enumerate(proj_photos.iterrows()):
        fpath = PHOTO_DIR / prow["filename"]
        with cols[i % len(cols)]:
            if fpath.exists():
                st.image(str(fpath), width='stretch')
            dist = haversine_meters(row.site_latitude, row.site_longitude, prow["latitude"], prow["longitude"])
            st.caption(f"{prow.photo_id} · {dist:,.0f}m from declared site")
            if dist > 500:
                st.error(t("location_mismatch"))

uploaded_evidence = read_uploaded_evidence(row.project_id)
if not uploaded_evidence.empty:
    st.divider()
    st.subheader(t("real_uploaded_evidence"))
    st.caption(t("real_uploaded_caption"))
    ecols = st.columns(min(len(uploaded_evidence), 4))
    for i, (_, erow) in enumerate(uploaded_evidence.iterrows()):
        epath = UPLOADED_PHOTO_DIR / erow["filename"]
        with ecols[i % len(ecols)]:
            if epath.exists():
                st.image(str(epath), width='stretch')
            st.caption(f"By {erow['uploaded_by']} · {str(erow['uploaded_at'])[:10]}")
            if erow["is_duplicate"]:
                st.error(t("duplicate_of", photo_id=erow['duplicate_of_photo']))
            if erow["geo_source"] == "exif" and pd.notna(erow["geo_distance_m"]):
                st.caption(f"{erow['geo_distance_m']:,.0f}m from site" + (" :material/warning:" if erow["geo_fail"] else " :material/check:"))
            else:
                st.caption(t("no_gps_metadata"))

st.divider()
st.subheader(t("citizen_reports_section"))
citizen = raw["citizen_reports"]
proj_reports = citizen[citizen["project_id"] == row.project_id]
if proj_reports.empty:
    st.caption(t("no_citizen_reports"))
else:
    view_reports = proj_reports[["report_id", "complaint_type", "report_date", "status"]].rename(columns={
        "report_id": t("col_report_id"), "complaint_type": t("col_complaint_type"),
        "report_date": t("col_report_date"), "status": t("status"),
    })
    render_html_table(view_reports.to_html(escape=False, index=False))

st.divider()
st.subheader(t("transaction_history"))
txns = raw["transactions"]
proj_txns = txns[txns["project_id"] == row.project_id].sort_values("installment_no")
view_txns = proj_txns[["transaction_id", "installment_no", "amount", "txn_date"]].rename(columns={
    "transaction_id": t("col_transaction_id"), "installment_no": t("col_installment"),
    "amount": t("col_amount"), "txn_date": t("col_txn_date"),
})
view_txns[t("col_amount")] = view_txns[t("col_amount")].apply(lambda v: f"₹{v:,.0f}" if pd.notna(v) else "—")
render_html_table(view_txns.to_html(escape=False, index=False))
