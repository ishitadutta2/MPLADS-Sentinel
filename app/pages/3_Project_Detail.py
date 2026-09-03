"""Project Detail — full deep-dive on a single project: timeline, photos, contractor links, citizen reports."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
from PIL import Image

from common import get_scored_dataset, get_raw_tables, inject_base_style, risk_pill, require_login, show_user_badge
from sentinel.config import PHOTO_DIR, UPLOADED_PHOTO_DIR
from sentinel.utils.geo import haversine_meters
from sentinel.db import read_uploaded_evidence

st.set_page_config(page_title="Project Detail — MPLADS Sentinel", page_icon="🔍", layout="wide")
inject_base_style()
user = require_login()
show_user_badge()
st.title("🔍 Project Detail")

df = get_scored_dataset()
raw = get_raw_tables()

query_id = st.text_input("Enter a Project ID (e.g. PRJ00363)", value="")
if not query_id:
    st.info("Tip: try a CRITICAL-risk project from the Home page's top-25 table, e.g. the first row shown there.")
    st.stop()

matches = df[df["project_id"].str.upper() == query_id.strip().upper()]
if matches.empty:
    st.error(f"No project found with ID '{query_id}'.")
    st.stop()

row = matches.iloc[0]

st.subheader(f"{row.project_id} — {row.category}")
st.markdown(risk_pill(row.risk_band), unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Composite Risk Score", f"{row.composite_score:.1f} / 100")
c2.metric("Sanctioned Amount", f"₹{row.sanctioned_amount:,.0f}")
c3.metric("Expenditure", f"₹{row.expenditure:,.0f}")
c4.metric("Status", row.status)

st.markdown(f"**MP:** {row.mp_name} ({row.constituency})  |  **District:** {row.district}, {row.state}")
st.markdown(f"**Contractor:** {row.contractor_name} (`{row.contractor_id}`)")
st.markdown(f"**Description:** {row.description}")

st.divider()
st.subheader("Why this score?")
st.code(row.explanation, language=None)

sub1, sub2, sub3, sub4, sub5 = st.columns(5)
sub1.metric("Financial", f"{row.financial_score:.0f}")
sub2.metric("Cost Benchmark", f"{row.cost_benchmark_score:.0f}")
sub3.metric("Geo/Photo", f"{row.geo_photo_score:.0f}")
sub4.metric("Contractor Network", f"{row.contractor_network_score:.0f}")
sub5.metric("Documents", f"{row.document_score:.0f}")

if "ml_risk_probability" in row and pd.notna(row.ml_risk_probability):
    st.markdown("**Supervised ML model (cross-validated, trained on top of the five signals above):**")
    ml_col1, ml_col2 = st.columns(2)
    ml_col1.metric("Model-predicted irregularity probability", f"{row.ml_risk_probability*100:.0f}%")
    ml_col2.metric("Top contributing signal (model-learned)", str(row.get("ml_top_signal", "—")).replace("_", " ").title())
    st.caption(
        "This is a genuinely trained `RandomForestClassifier`, not another rule — evaluated with "
        "5-fold cross-validation so every prediction is out-of-fold. See README for full model-card metrics."
    )

st.divider()
st.subheader("Evidence Photos")
photos = raw["photos"]
proj_photos = photos[photos["project_id"] == row.project_id]

if proj_photos.empty:
    st.info("No photos on file for this project.")
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
                st.error("⚠️ Location mismatch")

uploaded_evidence = read_uploaded_evidence(row.project_id)
if not uploaded_evidence.empty:
    st.divider()
    st.subheader("Real Uploaded Evidence")
    st.caption("Photos actually submitted by a logged-in user through Photo Verification → Upload & Check a Photo, not part of this demo's synthetic corpus.")
    ecols = st.columns(min(len(uploaded_evidence), 4))
    for i, (_, erow) in enumerate(uploaded_evidence.iterrows()):
        epath = UPLOADED_PHOTO_DIR / erow["filename"]
        with ecols[i % len(ecols)]:
            if epath.exists():
                st.image(str(epath), width='stretch')
            st.caption(f"By {erow['uploaded_by']} · {str(erow['uploaded_at'])[:10]}")
            if erow["is_duplicate"]:
                st.error(f"Duplicate of {erow['duplicate_of_photo']}")
            if erow["geo_source"] == "exif" and pd.notna(erow["geo_distance_m"]):
                st.caption(f"{erow['geo_distance_m']:,.0f}m from site" + (" ⚠️" if erow["geo_fail"] else " ✓"))
            else:
                st.caption("No GPS metadata in photo")

st.divider()
st.subheader("Citizen Reports")
citizen = raw["citizen_reports"]
proj_reports = citizen[citizen["project_id"] == row.project_id]
if proj_reports.empty:
    st.caption("No citizen reports filed for this project.")
else:
    st.dataframe(proj_reports[["report_id", "complaint_type", "report_date", "status"]], width='stretch', hide_index=True)

st.divider()
st.subheader("Transaction History")
txns = raw["transactions"]
proj_txns = txns[txns["project_id"] == row.project_id].sort_values("installment_no")
st.dataframe(proj_txns[["transaction_id", "installment_no", "amount", "txn_date"]], width='stretch', hide_index=True)
