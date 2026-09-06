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

from common import get_real_mp_allocations, inject_base_style, require_login, show_user_badge, apply_plot_theme, page_header
from i18n import t

st.set_page_config(page_title="National Allocation Overview — MPLADS Sentinel", page_icon="🇮🇳", layout="wide")
inject_base_style()
user = require_login()
show_user_badge()

page_header("🇮🇳", t("national_title"))
st.success(t("real_data_banner"))

with st.expander(t("about_esakshi_expander"), expanded=False):
    st.markdown("""
The MPLADS–eSAKSHI web portal (`mplads.mospi.gov.in`) has managed the MPLADS fund-flow process since
**1 April 2023**. Before that, the Scheme ran in physical mode, with District Authorities maintaining
recommendation/sanction records and disbursing through physical district bank accounts — which is why
recommendation and sanction data for the 17th Lok Sabha is only available on eSAKSHI from 2023-24 onward
(and not for 2019-20 through 2022-23), and Rajya Sabha MP data only from 2023-24 onward.

**The real workflow, stage by stage:**
1. **Recommendation** — an Hon'ble MP logs into the portal and recommends an eligible developmental work,
   earmarking funds from their annual entitlement. The total amount an MP has recommended is itself the
   indicator of how much of their entitlement they've put to use.
2. **Sanction** — the District Authority runs feasibility checks, sanctions the work, and designates an
   Implementing Agency (IA) to execute it. Sanctioned works are generally required to be completed
   **within one year** of sanction.
3. **Execution & payment** — the Implementing Agency raises vendor payment requests on the portal at each
   stage of execution, in line with the sanction order. The "expenditure" figures on any eSAKSHI dashboard
   are the total vendor payments released against completed/ongoing works — not necessarily the full
   sanctioned amount.
4. **Marking complete** — after the final payment, the IA must explicitly **"mark the work as complete"**
   on the portal. Only works marked complete this way show as completed on the public dashboard — a
   functionally finished work can still show as in-progress if this last step hasn't happened yet, which
   is why District Authorities are continuously pursued to ensure IAs comply.
5. **Evidence** — the IA uploads photographs of the asset at different stages, plus documents like the
   sanction order, so the public and other stakeholders can independently verify the work.

Everything updates in real time as each stakeholder — MP, District Authority, or Implementing Agency —
acts through their own login.

*(This explainer is currently only available in English.)*
""")

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
    fig = px.histogram(df, x="allocated_amount", nbins=40,
                        labels={"allocated_amount": t("axis_allocated_amount"), "count": t("axis_num_projects")})
    fig.update_layout(height=360, showlegend=False)
    fig = apply_plot_theme(fig)
    st.plotly_chart(fig, width='stretch')

with col2:
    st.subheader(t("total_entitlement_by_state"))
    state_totals = df.groupby("state")["allocated_amount"].agg(["sum", "count"]).reset_index()
    state_totals.columns = ["state", "total_allocated", "n_mps"]
    state_totals["total_cr"] = state_totals["total_allocated"] / 1e7
    fig2 = px.bar(
        state_totals.sort_values("total_cr", ascending=True),
        x="total_cr", y="state", orientation="h",
        labels={"total_cr": t("axis_total_allocated_cr"), "state": ""},
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
st.dataframe(view, width='stretch', hide_index=True, height=420)

st.caption(t("showing_n_of_n_mps", shown=len(view), total=len(df)))
if df["allocated_amount"].isna().any():
    st.caption(t("note_missing_allocation", n=df["allocated_amount"].isna().sum()))
