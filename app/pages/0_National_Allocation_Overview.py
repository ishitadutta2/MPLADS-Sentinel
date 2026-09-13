"""
National Allocation Overview — the ONE page in this app built entirely on
real, published data: the eSAKSHI-portal allocated-limit dataset for the
18th Lok Sabha (543 MPs, their state, constituency, and annual MPLADS
entitlement). No risk scoring, no anomaly flags, no synthetic figures
appear anywhere on this page — it is a factual reference view only.

Everywhere else in this app (Home, MP Dashboard, District Officer Queue,
Photo Verification, Contractor Network, Project Detail, Citizen
Reporting) uses a synthetically generated project/contractor/photo
dataset to demonstrate the detection engines, precisely so that no
fabricated risk score or fraud flag is ever attached to a real, named,
sitting Member of Parliament.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.express as px
import pandas as pd
import streamlit as st

from common import (
    get_real_mp_allocations, inject_base_style, require_login, show_user_badge,
    apply_plot_theme, page_header, get_palette, render_html_table,
)
from i18n import t

st.set_page_config(page_title="National Allocation Overview — MPLADS Sentinel", page_icon=":material/account_balance:", layout="wide")
inject_base_style()
user = require_login()
show_user_badge()

page_header("landmark", t("national_title"))

df = get_real_mp_allocations()

c1, c2, c3, c4 = st.columns(4)
c1.metric(t("kpi_mps_covered"), f"{len(df):,}")
c2.metric(t("kpi_states_uts"), f"{df['state'].nunique()}")
c3.metric(t("kpi_total_entitlement"), f"₹{df['allocated_amount'].sum()/1e7:,.0f} Cr")
c4.metric(t("kpi_median_entitlement"), f"₹{df['allocated_amount'].median()/1e7:.2f} Cr")

st.caption(t("standard_entitlement_caption"))

st.divider()

col1, col2 = st.columns([1, 1.3])
with col1:
    st.subheader(t("entitlement_distribution"))
    pal = get_palette()
    fig = px.histogram(
        df, x="allocated_amount", nbins=40,
        labels={"allocated_amount": t("axis_allocated_amount"), "count": t("axis_num_projects")},
        color_discrete_sequence=[pal["primary"]],
    )
    fig.update_layout(height=360, showlegend=False)
    fig = apply_plot_theme(fig)
    st.plotly_chart(fig, width='stretch')

with col2:
    st.subheader(t("total_entitlement_by_state"))
    pal = get_palette()
    state_totals = df.groupby("state")["allocated_amount"].agg(["sum", "count"]).reset_index()
    state_totals.columns = ["state", "total_allocated", "n_mps"]
    state_totals["total_cr"] = state_totals["total_allocated"] / 1e7
    fig2 = px.bar(
        state_totals.sort_values("total_cr", ascending=True),
        x="total_cr", y="state", orientation="h",
        labels={"total_cr": t("axis_total_allocated_cr"), "state": ""},
        color_discrete_sequence=[pal["accent"]],
        height=560,
    )
    fig2 = apply_plot_theme(fig2)
    st.plotly_chart(fig2, width='stretch')

st.divider()
st.subheader(t("search_mps"))
search = st.text_input(t("search_placeholder"))
view = df.copy()
if search:
    mask = (
        df["mp_name"].str.contains(search, case=False, na=False, regex=False)
        | df["state"].str.contains(search, case=False, na=False, regex=False)
        | df["constituency"].str.contains(search, case=False, na=False, regex=False)
    )
    view = df[mask]

view = view.rename(columns={
    "mp_name": t("col_mp_name_full"), "state": t("col_state"), "constituency": t("constituency"),
    "allocated_amount": t("axis_allocated_amount"),
})[[t("col_mp_name_full"), t("col_state"), t("constituency"), t("axis_allocated_amount")]]
view[t("axis_allocated_amount")] = view[t("axis_allocated_amount")].apply(
    lambda v: f"₹{v:,.0f}" if pd.notna(v) else t("not_yet_published")
)
render_html_table(view.to_html(escape=False, index=False), max_height=420)

st.caption(t("showing_n_of_n_mps", shown=len(view), total=len(df)))
if df["allocated_amount"].isna().any():
    st.caption(t("note_missing_allocation", n=df["allocated_amount"].isna().sum()))
