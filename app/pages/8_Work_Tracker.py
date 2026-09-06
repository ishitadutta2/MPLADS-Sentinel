"""Work Tracker — the one page in this app that WRITES to the projects
table instead of only reading a frozen snapshot of it.

This is deliberately separate from the read-only District Officer Queue
(risk-driven review) and Citizen Reporting (public complaints): this page
models the actual eSAKSHI recommend -> sanction -> execute -> pay ->
mark-complete pipeline itself, so an MP can recommend a real new work and
a District Officer/Central Admin can move it through that pipeline, and
everyone immediately sees it reflected everywhere else in the app (Home,
MP Dashboard, District Officer Queue) — not just here.
"""
import datetime

import pandas as pd
import streamlit as st

from common import (
    get_scored_dataset, inject_base_style, require_login, show_user_badge,
    scoped_query, risk_pill, render_html_table, page_header,
)
from sentinel.db import insert_project, update_project, next_work_id, read_table
from sentinel.audit.hash_chain import append_event
from i18n import t, stage_label

st.set_page_config(page_title="Work Tracker — MPLADS Sentinel", page_icon="🧭", layout="wide")
inject_base_style()
user = require_login()
show_user_badge()

page_header(
    "🧭", t("tracker_title"),
    t("tracker_sub"),
)

STAGES = ["Recommended", "Sanctioned", "Work In Progress", "Payment Released", "Completed"]
CATEGORIES = [
    "Community Hall", "Drinking Water Supply", "Public Toilet Complex", "Borewell / Handpump",
    "School Building Repair", "Sports Infrastructure", "Library / Reading Room",
    "Drainage System", "Street Lighting", "Road Construction",
]


def _next_stage(current: str):
    if current not in STAGES:
        return None
    i = STAGES.index(current)
    return STAGES[i + 1] if i + 1 < len(STAGES) else None


df = get_scored_dataset()
scoped = scoped_query(df, user)

# ---- Status board -------------------------------------------------------
st.subheader(t("status_board"))
counts = scoped["status"].value_counts()
cols = st.columns(len(STAGES))
for c, stage in zip(cols, STAGES):
    with c:
        st.metric(stage_label(stage), int(counts.get(stage, 0)))

st.divider()

# ---- MP: recommend a new work -------------------------------------------
if user["role"] in ("mp", "central_admin"):
    with st.expander(f"➕ {t('recommend_new')}", expanded=False):
        mps_df = read_table("mps")
        if user["role"] == "mp":
            mp_id = user["scope_mp_id"]
            mp_row = mps_df[mps_df["mp_id"] == mp_id]
        else:
            pick = st.selectbox(
                t("choose_mp"), mps_df["mp_name"] + " — " + mps_df["mp_id"], key="wt_mp_pick"
            )
            mp_id = pick.split(" — ")[-1]
            mp_row = mps_df[mps_df["mp_id"] == mp_id]

        own_projects = df[df["mp_id"] == mp_id]
        if len(own_projects):
            constituency = own_projects["constituency"].iloc[0]
            state = own_projects["state"].iloc[0]
            district = own_projects["district"].iloc[0]
        elif len(mp_row):
            constituency, state, district = mp_row["constituency"].iloc[0], mp_row["state"].iloc[0], ""
        else:
            constituency, state, district = "", "", ""

        st.caption(f"{t('constituency')}: **{constituency}**, {state} · {t('district')}: **{district or '—'}**")

        with st.form("recommend_work_form", clear_on_submit=True):
            category = st.selectbox(t("category"), CATEGORIES)
            description = st.text_area(t("description"), placeholder=t("description_ph"))
            proposed_amount = st.number_input(t("proposed_amount"), min_value=0, step=10000, format="%d")
            agency = st.text_input(t("implementing_agency"), placeholder=t("agency_ph"))
            submitted = st.form_submit_button(t("submit_recommendation"), width='stretch')
            if submitted:
                if not description.strip() or not district:
                    st.error(t("fill_required"))
                else:
                    wid = next_work_id()
                    insert_project({
                        "project_id": wid, "mp_id": mp_id, "constituency": constituency,
                        "state": state, "district": district, "category": category,
                        "description": description.strip(), "sanctioned_amount": proposed_amount,
                        "expenditure": 0, "status": "Recommended",
                        "recommendation_date": datetime.date.today().isoformat(),
                        "implementing_agency": agency.strip() or None,
                        "marked_complete_by_ia": 0, "is_seeded_anomalous": 0,
                    })
                    append_event("WORK_RECOMMENDED", wid, user["username"], {
                        "category": category, "district": district, "proposed_amount": proposed_amount,
                    })
                    get_scored_dataset.clear()
                    st.success(f"{t('recommended_ok')} — `{wid}`")
                    st.rerun()

# ---- District Officer / Central Admin: advance a work's status ----------
if user["role"] in ("district_officer", "central_admin"):
    with st.expander(f"➡️ {t('advance_status')}", expanded=False):
        advanceable = scoped[scoped["status"] != "Completed"].sort_values("recommendation_date", ascending=False)
        if advanceable.empty:
            st.caption(t("nothing_to_advance"))
        else:
            labels = (advanceable["project_id"] + " — " + advanceable["category"].fillna("") +
                      " (" + advanceable["status"].fillna("").apply(stage_label) + ")")
            pick = st.selectbox(t("choose_work"), labels, key="wt_advance_pick")
            pid = pick.split(" — ")[0]
            row = advanceable[advanceable["project_id"] == pid].iloc[0]
            nxt = _next_stage(row["status"])
            st.write(f"**{row['description']}**")
            st.caption(f"{t('current_status')}: **{stage_label(row['status'])}** · {row['district']}, {row['state']}")
            if nxt is None:
                st.caption(t("no_further_stage"))
            else:
                notes = st.text_area(t("notes_optional"), key="wt_advance_notes")
                if st.button(f"{t('advance_to')} \u201c{stage_label(nxt)}\u201d \u2192", key="wt_advance_btn"):
                    fields = {"status": nxt}
                    if nxt == "Sanctioned":
                        fields["sanction_date"] = datetime.date.today().isoformat()
                    if nxt == "Completed":
                        fields["marked_complete_by_ia"] = 1
                    update_project(pid, fields)
                    append_event("WORK_STATUS_ADVANCED", pid, user["username"], {
                        "from": row["status"], "to": nxt, "notes": notes,
                    })
                    get_scored_dataset.clear()
                    st.success(f"{pid} → {stage_label(nxt)}")
                    st.rerun()

st.divider()

# ---- Live list of scoped works ------------------------------------------
st.subheader(t("your_works"))
if scoped.empty:
    st.caption(t("no_works_yet"))
else:
    view = scoped.sort_values("recommendation_date", ascending=False).head(200).copy()
    is_new = view["project_id"].str.startswith("WRK-")
    rows_html = ""
    for _, r in view.iterrows():
        badge = f'<span style="opacity:.55; font-size:.72rem;">🆕 {t("added_here")}</span>' if r["project_id"].startswith("WRK-") else ""
        pill = risk_pill(r["risk_band"]) if pd.notna(r.get("scored_at")) else (
            f'<span style="opacity:.55; font-size:.78rem;">{t("not_yet_scored")}</span>'
        )
        rows_html += (
            f"<tr><td>{r['project_id']}</td><td>{r['category']}</td>"
            f"<td>{r['description'][:70]}{'…' if len(str(r['description']))>70 else ''}</td>"
            f"<td>{stage_label(r['status'])}</td><td>{r['district']}</td><td>{pill} {badge}</td></tr>"
        )
    render_html_table(
        f"<table><thead><tr><th>{t('col_project_id')}</th><th>{t('category')}</th><th>{t('description')}</th>"
        f"<th>{t('status')}</th><th>{t('district')}</th><th>{t('risk')}</th></tr></thead>"
        f"<tbody>{rows_html}</tbody></table>"
    )
