"""
Public Citizen Chatbot — a no-login, conversational front door for
citizens: report a problem (or share feedback) about an MPLADS-funded
project near them, or flag something that isn't even in the system yet.

Flow: what's up today -> where are you reporting from -> which project
(or "none of these, it's something else") -> your details -> the report
is registered into the SAME citizen_reports table the District Officer
Queue and Citizen Reporting dashboard already read from, so an existing
project's report shows up right alongside ones filed through the old
web form, and an unlisted one shows up flagged for an officer to triage.

Deliberately English-only for now, same honest trade-off as the photo
verification upload explainer elsewhere in this app — a translated
chat experience is future work, not faked here.
"""
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from common import get_scored_dataset, inject_base_style, header_controls, get_palette, svg_icon
from sentinel.db import insert_citizen_report_full, next_report_id
from sentinel.audit.hash_chain import append_event

st.set_page_config(page_title="Citizen Assistant — MPLADS Sentinel", page_icon=":material/smart_toy:", layout="wide")
inject_base_style()
header_controls()

# This page is deliberately for the public, not for signed-in officials —
# an MP office, district officer, or admin filing a "citizen" report against
# their own projects would defeat the point of an independent citizen
# verification channel. Anyone already signed in gets turned back here,
# before ever seeing the chat, and pointed at where officials actually
# raise things: recommending work in Work Tracker.
if st.session_state.get("user") is not None:
    accent = get_palette()["accent"]
    st.markdown(
        f"""
        <div style="text-align:center; margin:2.4rem 0 1.2rem;">
          <div style="opacity:.85;">{svg_icon('lock', size=34)}</div>
          <h1 class="sentinel-hero-title" style="margin:.3rem 0 0;">Not available while signed in</h1>
          <div style="width:40px; height:3px; background:{accent}; margin:.5rem auto 0; border-radius:2px;"></div>
          <p style="opacity:.72; margin-top:.6rem; font-size:.95rem; max-width:52ch; margin-left:auto; margin-right:auto;">
            The Citizen Assistant is an independent channel for the public to report problems —
            it isn't for officials to file reports on their own projects. If you want to raise or
            track a work item, use Work Tracker instead.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, mid, right = st.columns([1, 1, 1])
    with mid:
        if st.button("Go to Work Tracker", width='stretch', icon=":material/schedule:"):
            st.switch_page("pages/8_Work_Tracker.py")
        if st.button("Back to Home", width='stretch', icon=":material/home:"):
            st.switch_page("Home.py")
    st.stop()

COMPLAINT_TYPES = [
    "Work not started despite sanction",
    "Poor quality of construction material",
    "Marked complete but incomplete on the ground",
    "Facility not usable / not functional",
    "Suspected overpricing or fund misuse",
    "Something else",
]

NONE_OF_THESE = ":material/flag: None of these — my issue is about something else"
PLACEHOLDER_PICK = "— Select a project —"
ALL_TYPES = "All types of work"

# Every stage in the happy path, in order — used only to render a "Step X
# of Y" caption so people can see how close they are to done; it has no
# effect on navigation.
STEP_ORDER = ["start", "ask_location", "ask_project", "existing_details", "done"]


def _step_caption(stage_name: str):
    stage_for_count = "existing_details" if stage_name == "new_details" else stage_name
    if stage_for_count in STEP_ORDER:
        n = STEP_ORDER.index(stage_for_count) + 1
        st.caption(f"Step {n} of {len(STEP_ORDER)}")


def _init_state():
    if "pc_stage" not in st.session_state:
        st.session_state.pc_stage = "start"
        st.session_state.pc_data = {}
        st.session_state.pc_transcript = [{
            "role": "assistant",
            "text": (
                ":material/waving_hand: Hi, I'm the **MPLADS Sentinel** citizen assistant. I can help you report a "
                "problem, or share feedback, about a government-funded project near you.\n\n"
                "What's up today?"
            ),
        }]


def _say(text: str):
    st.session_state.pc_transcript.append({"role": "assistant", "text": text})


def _echo(text: str):
    st.session_state.pc_transcript.append({"role": "user", "text": text})


def _reset():
    for k in ("pc_stage", "pc_data", "pc_transcript"):
        st.session_state.pop(k, None)


_init_state()

accent = get_palette()["accent"]
st.markdown(
    f"""
    <div style="text-align:center; margin:.2rem 0 1.2rem;">
      <div style="opacity:.85;">{svg_icon('bot', size=34)}</div>
      <h1 class="sentinel-hero-title" style="margin:.3rem 0 0;">Citizen Assistant</h1>
      <div style="width:40px; height:3px; background:{accent}; margin:.5rem auto 0; border-radius:2px;"></div>
      <p style="opacity:.72; margin-top:.6rem; font-size:.95rem;">
        No login needed — report a problem or check in on a project near you.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.button("Back to Home", icon=":material/home:"):
    _reset()
    st.switch_page("Home.py")

for msg in st.session_state.pc_transcript:
    with st.chat_message(msg["role"], avatar=":material/smart_toy:" if msg["role"] == "assistant" else ":material/person:"):
        st.markdown(msg["text"])

df = get_scored_dataset()
stage = st.session_state.pc_stage
data = st.session_state.pc_data

# ---------------------------------------------------------------------
if stage == "start":
    c1, c2 = st.columns(2)
    if c1.button("I have a complaint / feedback", width='stretch', icon=":material/forum:"):
        _echo("I have a complaint / feedback")
        _say("Got it. Which **state** and **district** are you reporting from?")
        st.session_state.pc_stage = "ask_location"
        st.rerun()
    if c2.button("Just exploring", width='stretch', icon=":material/info:"):
        _echo("Just exploring")
        _say(
            "No problem! I'm here whenever you want to report a problem with an MPLADS-funded "
            "project, or flag one you don't see listed at all. Come back anytime."
        )
        st.session_state.pc_stage = "explore_end"
        st.rerun()

elif stage == "explore_end":
    if st.button("Actually, I do want to report something", icon=":material/forum:"):
        _echo("Actually, I do want to report something")
        _say("Sure — which **state** and **district** are you reporting from?")
        st.session_state.pc_stage = "ask_location"
        st.rerun()

elif stage == "ask_location":
    _step_caption(stage)
    states = sorted(df["state"].dropna().unique().tolist())
    # Streamlit drops a widget's session state once a run goes by without
    # rendering it — which happens here whenever "Change state /
    # district" sends someone back from ask_project. Re-deriving the
    # default index from our own pc_data (which we control and never
    # drop) means coming back here re-shows their last pick instead of
    # silently resetting to the alphabetically-first state/district.
    state_idx = states.index(data["state"]) if data.get("state") in states else 0
    state = st.selectbox("State", states, index=state_idx, key="pc_state_sel")
    districts = sorted(df.loc[df["state"] == state, "district"].dropna().unique().tolist())
    district_idx = districts.index(data["district"]) if data.get("district") in districts else 0
    district = st.selectbox("District", districts, index=district_idx, key="pc_district_sel") if districts else None
    if st.button("Continue", icon=":material/arrow_forward:"):
        data["state"] = state
        data["district"] = district
        _echo(f"{district}, {state}" if district else state)
        matches = df[(df["state"] == state) & (df["district"] == district)] if district else df.iloc[0:0]
        if matches.empty:
            _say(
                f"I don't have any projects on file for **{district or state}** yet. "
                "No problem — let's file this as a new report with your own details."
            )
            st.session_state.pc_stage = "new_details"
        else:
            n = len(matches)
            _say(
                f"I found **{n} project{'s' if n != 1 else ''}** on file for **{district}, {state}**. "
                "Narrow it down by type of work below, then pick the one this is about — or say it's "
                "not listed at all."
            )
            st.session_state.pc_stage = "ask_project"
        st.rerun()

elif stage == "ask_project":
    _step_caption(stage)
    matches = df[(df["state"] == data.get("state")) & (df["district"] == data.get("district"))]
    # A district can have anywhere from ~40 to 120 projects on file — far
    # too many to lay out as a wall of radio buttons (that was the
    # original design here, and it was genuinely unusable at real scale).
    # A category filter to narrow things down, feeding a single searchable
    # dropdown (Streamlit's selectbox supports type-to-search natively),
    # scales to hundreds of projects with no more effort than picking from
    # a handful.
    categories = [ALL_TYPES] + sorted(matches["category"].dropna().unique().tolist())
    cat_filter = st.selectbox(
        f"Narrow down by type of work ({len(matches)} total)", categories, key="pc_cat_filter",
    )
    if cat_filter != ALL_TYPES:
        matches = matches[matches["category"] == cat_filter]
    options = {
        f"{r.project_id} — {r.category} · {r.status}": r.project_id
        for r in matches.itertuples()
    }
    labels = [PLACEHOLDER_PICK] + list(options.keys()) + [NONE_OF_THESE]
    choice = st.selectbox(
        f"Which project is this about? (type to search {len(matches)} shown)", labels,
        key="pc_project_choice",
    )
    b1, b2 = st.columns([1, 1])
    if b1.button("Select", width='stretch', icon=":material/check:"):
        if choice == PLACEHOLDER_PICK:
            st.warning("Pick a project from the list first — or choose the last option if it isn't listed.")
        else:
            _echo(choice)
            if choice == NONE_OF_THESE:
                _say("No problem — tell me a bit about it and I'll register it as a new report.")
                data["reached_new_via_project_list"] = True
                st.session_state.pc_stage = "new_details"
            else:
                data["project_id"] = options[choice]
                _say("Thanks. Could you tell me a bit more, and how an officer can reach you?")
                st.session_state.pc_stage = "existing_details"
            st.rerun()
    if b2.button("Change state / district", width='stretch', icon=":material/arrow_back:"):
        st.session_state.pc_stage = "ask_location"
        _say("Sure — let's pick your state and district again.")
        st.rerun()

elif stage == "existing_details":
    _step_caption(stage)
    with st.form("pc_existing_form"):
        complaint_type = st.selectbox("What's the issue?", COMPLAINT_TYPES)
        details = st.text_area("Anything else you'd like to add? (optional)")
        name = st.text_input("Your name (optional)")
        contact = st.text_input("Phone or email (optional, so an officer can follow up)")
        submitted = st.form_submit_button("Submit report")
    if st.button("Back to project list", icon=":material/arrow_back:"):
        st.session_state.pc_stage = "ask_project"
        st.rerun()
    if submitted:
        report_id = next_report_id("CIT")
        masked = f"Citizen_{abs(hash(name or 'anon')) % 9000 + 1000}"
        insert_citizen_report_full({
            "report_id": report_id,
            "project_id": data["project_id"],
            "complaint_type": complaint_type,
            "report_date": datetime.date.today().isoformat(),
            "citizen_name_masked": masked,
            "status": "Open",
            "details": details,
            "citizen_contact": contact,
            "report_state": data.get("state"),
            "report_district": data.get("district"),
            "channel": "chatbot",
            "is_unlisted_project": 0,
        })
        append_event(
            event_type="CITIZEN_REPORT", project_id=data["project_id"], actor=masked,
            details={"complaint_type": complaint_type, "details": details, "channel": "chatbot"},
        )
        st.cache_data.clear()
        _echo(f"{complaint_type} — {details or '(no extra details)'}")
        _say(
            f":material/check_circle: Thanks! Your report has been registered — reference **{report_id}**. "
            "District officers and admins will see this on the Citizen Reports dashboard."
        )
        data["last_report_id"] = report_id
        st.session_state.pc_stage = "done"
        st.rerun()

elif stage == "new_details":
    _step_caption(stage)
    categories = sorted(df["category"].dropna().unique().tolist()) + ["Other / not sure"]
    with st.form("pc_new_form"):
        title = st.text_input("In a few words, what is this project/work about?")
        category = st.selectbox("Closest category", categories)
        locality = st.text_input("Village / area / landmark (optional, helps an officer locate it)")
        complaint_type = st.selectbox("What's the issue?", COMPLAINT_TYPES)
        details = st.text_area("Describe the problem")
        name = st.text_input("Your name (optional)")
        contact = st.text_input("Phone or email (optional, so an officer can follow up)")
        submitted = st.form_submit_button("Submit report")
    if data.get("reached_new_via_project_list") and st.button("Back to project list", icon=":material/arrow_back:"):
        st.session_state.pc_stage = "ask_project"
        st.rerun()
    if submitted:
        if not title.strip() and not details.strip():
            st.error("Please give at least a short title or description so an officer knows what this is about.")
        else:
            report_id = next_report_id("NEW")
            masked = f"Citizen_{abs(hash(name or 'anon')) % 9000 + 1000}"
            full_details = f"{locality + ' — ' if locality else ''}{details}"
            insert_citizen_report_full({
                "report_id": report_id,
                "project_id": report_id,  # no real project to key on — id doubles as a stable handle
                "complaint_type": complaint_type,
                "report_date": datetime.date.today().isoformat(),
                "citizen_name_masked": masked,
                "status": "Open",
                "details": full_details,
                "citizen_contact": contact,
                "report_state": data.get("state"),
                "report_district": data.get("district"),
                "category": category,
                "unlisted_title": title,
                "channel": "chatbot",
                "is_unlisted_project": 1,
            })
            append_event(
                event_type="CITIZEN_REPORT", project_id=report_id, actor=masked,
                details={
                    "complaint_type": complaint_type, "details": details, "channel": "chatbot",
                    "unlisted_title": title, "category": category,
                },
            )
            st.cache_data.clear()
            _echo(f"{title} — {complaint_type}")
            _say(
                f":material/check_circle: Thanks! I've registered this as a new report — reference **{report_id}**. "
                "Since it isn't a project already tracked in the system, it's flagged for an "
                "officer to review and follow up on directly."
            )
            data["last_report_id"] = report_id
            st.session_state.pc_stage = "done"
            st.rerun()

elif stage == "done":
    _step_caption(stage)
    c1, c2 = st.columns(2)
    if c1.button("File another report", width='stretch', icon=":material/edit_note:"):
        _reset()
        st.rerun()
    if c2.button("Back to Home", width='stretch', icon=":material/home:"):
        _reset()
        st.switch_page("Home.py")
