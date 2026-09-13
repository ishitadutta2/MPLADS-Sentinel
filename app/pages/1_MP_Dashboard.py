"""MP Dashboard — self-service transparency view for an individual MP."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.express as px
import streamlit as st

from common import (
    get_scored_dataset, inject_base_style, risk_pill, RISK_COLORS, require_login,
    show_user_badge, apply_plot_theme, render_html_table, page_header,
)
from i18n import t, stage_label

st.set_page_config(page_title="MP Dashboard — MPLADS Sentinel", page_icon=":material/badge:", layout="wide")
inject_base_style()
user = require_login()
show_user_badge()

page_header("briefcase", t("mp_dash_title"), t("mp_dash_sub"))

df = get_scored_dataset()

mp_options = df[["mp_id", "mp_name", "constituency"]].drop_duplicates().sort_values("mp_name")

if user["role"] == "mp":
    # An MP account is locked to its own constituency — not a hidden
    # default, an enforced restriction: the selectbox itself only ever
    # contains one option for this role.
    mp_options = mp_options[mp_options.mp_id == user["scope_mp_id"]]
    st.caption(t("scoped_mp_caption", mp_id=user["scope_mp_id"]))

mp_label = st.selectbox(
    t("select_mp"),
    mp_options.apply(lambda r: f"{r.mp_name} — {r.constituency} ({r.mp_id})", axis=1),
)
mp_id = mp_label.split("(")[-1].strip(")")

mp_df = df[df.mp_id == mp_id]
mp_row = mp_options[mp_options.mp_id == mp_id].iloc[0]

st.subheader(f"{mp_row.mp_name} — {mp_row.constituency}")

c1, c2, c3, c4 = st.columns(4)
c1.metric(t("kpi_total_projects"), len(mp_df))
c2.metric(t("kpi_sanctioned"), f"₹{mp_df.sanctioned_amount.sum()/1e7:,.2f} Cr")
c3.metric(t("kpi_utilised"), f"₹{mp_df.expenditure.sum()/1e7:,.2f} Cr")
c4.metric(t("kpi_utilisation_pct"), f"{mp_df.expenditure.sum()/max(mp_df.sanctioned_amount.sum(),1)*100:.1f}%")

flagged = mp_df[mp_df.risk_band.isin(["HIGH", "CRITICAL"])]
if len(flagged):
    st.warning(t("flagged_warning", n=len(flagged)))
else:
    st.success(t("no_flagged_success"))

col1, col2 = st.columns(2)
with col1:
    st.subheader(t("projects_by_status"))
    status_counts = mp_df["status"].value_counts().reset_index()
    status_counts.columns = ["status", "count"]
    status_counts["status"] = status_counts["status"].apply(stage_label)
    fig = px.pie(
        status_counts, names="status", values="count", hole=0.5,
        color_discrete_sequence=["#0D9488", "#14B8A6", "#38BDF8", "#F59E0B", "#10B981"],
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=320, showlegend=False)
    fig = apply_plot_theme(fig)
    st.plotly_chart(fig, width='stretch')

with col2:
    st.subheader(t("risk_band_breakdown"))
    band_counts = mp_df["risk_band"].value_counts().reindex(["LOW","MEDIUM","HIGH","CRITICAL"]).fillna(0).reset_index()
    band_counts.columns = ["risk_band", "count"]
    fig2 = px.bar(band_counts, x="risk_band", y="count", color="risk_band", color_discrete_map=RISK_COLORS,
                   labels={"risk_band": t("axis_risk_band"), "count": t("axis_num_projects")})
    fig2.update_layout(showlegend=False, height=320)
    fig2 = apply_plot_theme(fig2)
    st.plotly_chart(fig2, width='stretch')

st.subheader(t("all_projects"))
category_filter = st.multiselect(t("filter_by_category"), sorted(mp_df["category"].unique()))
shown = mp_df if not category_filter else mp_df[mp_df["category"].isin(category_filter)]

table = shown.sort_values("composite_score", ascending=False)[
    ["project_id", "category", "district", "contractor_name", "sanctioned_amount",
     "status", "composite_score", "risk_band"]
].copy()
table["sanctioned_amount"] = table["sanctioned_amount"].apply(lambda v: f"₹{v:,.0f}")
table["status"] = table["status"].apply(stage_label)
table["risk_band"] = table["risk_band"].apply(risk_pill)
table.columns = [t("col_project_id"), t("category"), t("district"), t("col_contractor"),
                  t("col_sanctioned_amount"), t("status"), t("col_score"), t("risk")]
render_html_table(table.to_html(escape=False, index=False))
