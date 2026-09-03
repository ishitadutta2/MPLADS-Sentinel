"""MP Dashboard — self-service transparency view for an individual MP."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.express as px
import streamlit as st

from common import (
    get_scored_dataset, inject_base_style, risk_pill, RISK_COLORS, require_login,
    show_user_badge, apply_plot_theme, render_html_table,
)

st.set_page_config(page_title="MP Dashboard — MPLADS Sentinel", page_icon="🧑‍💼", layout="wide")
inject_base_style()
user = require_login()
show_user_badge()

st.title("🧑‍💼 MP Dashboard")
st.caption("A transparency view any MP's office can use to self-monitor their own fund utilisation and flagged projects.")

df = get_scored_dataset()

mp_options = df[["mp_id", "mp_name", "constituency"]].drop_duplicates().sort_values("mp_name")

if user["role"] == "mp":
    # An MP account is locked to its own constituency — not a hidden
    # default, an enforced restriction: the selectbox itself only ever
    # contains one option for this role.
    mp_options = mp_options[mp_options.mp_id == user["scope_mp_id"]]
    st.caption(f"Logged in as an MP account — scoped to {user['scope_mp_id']} only.")

mp_label = st.selectbox(
    "Select MP",
    mp_options.apply(lambda r: f"{r.mp_name} — {r.constituency} ({r.mp_id})", axis=1),
)
mp_id = mp_label.split("(")[-1].strip(")")

mp_df = df[df.mp_id == mp_id]
mp_row = mp_options[mp_options.mp_id == mp_id].iloc[0]

st.subheader(f"{mp_row.mp_name} — {mp_row.constituency}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Projects", len(mp_df))
c2.metric("Sanctioned", f"₹{mp_df.sanctioned_amount.sum()/1e7:,.2f} Cr")
c3.metric("Utilised", f"₹{mp_df.expenditure.sum()/1e7:,.2f} Cr")
c4.metric("Utilisation %", f"{mp_df.expenditure.sum()/max(mp_df.sanctioned_amount.sum(),1)*100:.1f}%")

flagged = mp_df[mp_df.risk_band.isin(["HIGH", "CRITICAL"])]
if len(flagged):
    st.warning(f"⚠️ {len(flagged)} of this MP's projects are currently flagged HIGH or CRITICAL risk and warrant review.")
else:
    st.success("✅ No HIGH or CRITICAL risk projects currently flagged for this MP.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Projects by Status")
    status_counts = mp_df["status"].value_counts().reset_index()
    status_counts.columns = ["status", "count"]
    fig = px.pie(status_counts, names="status", values="count", hole=0.45)
    fig.update_layout(height=320)
    fig = apply_plot_theme(fig)
    st.plotly_chart(fig, width='stretch')

with col2:
    st.subheader("Risk Band Breakdown")
    band_counts = mp_df["risk_band"].value_counts().reindex(["LOW","MEDIUM","HIGH","CRITICAL"]).fillna(0).reset_index()
    band_counts.columns = ["risk_band", "count"]
    fig2 = px.bar(band_counts, x="risk_band", y="count", color="risk_band", color_discrete_map=RISK_COLORS)
    fig2.update_layout(showlegend=False, height=320)
    fig2 = apply_plot_theme(fig2)
    st.plotly_chart(fig2, width='stretch')

st.subheader("All Projects")
category_filter = st.multiselect("Filter by category", sorted(mp_df["category"].unique()))
shown = mp_df if not category_filter else mp_df[mp_df["category"].isin(category_filter)]

table = shown.sort_values("composite_score", ascending=False)[
    ["project_id", "category", "district", "contractor_name", "sanctioned_amount",
     "status", "composite_score", "risk_band"]
].copy()
table["sanctioned_amount"] = table["sanctioned_amount"].apply(lambda v: f"₹{v:,.0f}")
table["risk_band"] = table["risk_band"].apply(risk_pill)
render_html_table(table.to_html(escape=False, index=False))
