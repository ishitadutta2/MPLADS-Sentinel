"""
District Officer Queue — the working screen for the human-in-the-loop
reviewer. Sentinel *never* auto-blocks funds; it only prioritises what a
district officer looks at first, and captures their verdict (this is the
"trust & explainability" surface the brief calls for).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from common import get_scored_dataset, inject_base_style, risk_pill, require_login, require_role, show_user_badge
from sentinel.db import insert_feedback, read_feedback
from sentinel.audit.hash_chain import append_event

st.set_page_config(page_title="District Officer Queue — MPLADS Sentinel", page_icon="🗂️", layout="wide")
inject_base_style()
user = require_login()
show_user_badge()
require_role(user, ["district_officer", "central_admin"])

st.title("🗂️ District Officer Review Queue")
st.caption(
    "Prioritised worklist of projects needing manual review, ranked by composite risk score. "
    "Sentinel flags — it never auto-blocks disbursal. Every verdict here is logged to the tamper-evident audit trail."
)

df = get_scored_dataset()
if user["role"] == "district_officer":
    df = df[df["district"] == user["scope_district"]]
    st.caption(f"Logged in as a District Officer account — scoped to {user['scope_district']} only.")

col1, col2, col3 = st.columns(3)
district_filter = col1.multiselect("District", sorted(df["district"].dropna().unique()))
band_filter = col2.multiselect("Risk Band", ["CRITICAL", "HIGH", "MEDIUM", "LOW"], default=["CRITICAL", "HIGH"])
category_filter = col3.multiselect("Category", sorted(df["category"].unique()))

queue = df.copy()
if district_filter:
    queue = queue[queue["district"].isin(district_filter)]
if band_filter:
    queue = queue[queue["risk_band"].isin(band_filter)]
if category_filter:
    queue = queue[queue["category"].isin(category_filter)]

queue = queue.sort_values("composite_score", ascending=False)

st.metric("Projects in Queue", len(queue))
st.divider()

feedback_df = read_feedback()
reviewed_ids = set(feedback_df["project_id"]) if len(feedback_df) else set()

for _, row in queue.head(30).iterrows():
    reviewed = row.project_id in reviewed_ids
    header = f"{'✅ ' if reviewed else ''}{row.project_id} — {row.category} · {row.district} — Score {row.composite_score}"
    with st.expander(header):
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(f"**MP:** {row.mp_name}  |  **Contractor:** {row.contractor_name}", unsafe_allow_html=False)
            st.markdown(f"**Sanctioned:** ₹{row.sanctioned_amount:,.0f}  |  **Status:** {row.status}")
            st.markdown(risk_pill(row.risk_band), unsafe_allow_html=True)
            st.markdown("**Why this project was flagged:**")
            st.code(row.explanation, language=None)
        with c2:
            st.markdown("**Sub-scores**")
            st.progress(min(int(row.financial_score), 100), text=f"Financial: {row.financial_score:.0f}")
            st.progress(min(int(row.cost_benchmark_score), 100), text=f"Cost benchmark: {row.cost_benchmark_score:.0f}")
            st.progress(min(int(row.geo_photo_score), 100), text=f"Geo/Photo: {row.geo_photo_score:.0f}")
            st.progress(min(int(row.contractor_network_score), 100), text=f"Contractor network: {row.contractor_network_score:.0f}")
            st.progress(min(int(row.document_score), 100), text=f"Documents: {row.document_score:.0f}")

        st.markdown("---")
        with st.form(key=f"feedback_{row.project_id}"):
            fc1, fc2, fc3 = st.columns([1, 1, 2])
            officer_name = fc1.text_input("Officer name", value=user["display_name"], key=f"officer_{row.project_id}")
            verdict = fc2.selectbox("Verdict", ["Confirmed irregularity", "False positive", "Needs field visit"], key=f"verdict_{row.project_id}")
            notes = fc3.text_input("Notes", key=f"notes_{row.project_id}")
            submitted = st.form_submit_button("Submit review")
            if submitted:
                insert_feedback(row.project_id, officer_name, verdict, notes)
                append_event(
                    event_type="OFFICER_REVIEW",
                    project_id=row.project_id,
                    actor=officer_name,
                    details={"verdict": verdict, "notes": notes, "composite_score": row.composite_score},
                )
                st.success("Review recorded and logged to the audit trail.")
                st.rerun()

if len(queue) > 30:
    st.caption(f"Showing top 30 of {len(queue)} matching projects — narrow the filters above to see more specific results.")
