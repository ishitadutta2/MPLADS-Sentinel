"""Shared helpers used across every page of the Streamlit app."""
import sys
from pathlib import Path
from string import Template

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from sentinel.db import read_table, load_csvs_into_db
from sentinel.services.scoring_pipeline import run_full_pipeline
from sentinel.config import DATA_DIR, PHOTO_DIR
from sentinel.auth import authenticate, seed_demo_users, scoped_query


@st.cache_resource(show_spinner=False)
def _ensure_demo_users_seeded():
    """Runs once per server process (st.cache_resource, not cache_data —
    this has a side effect, not a return value worth caching by content)
    rather than on every page load."""
    load_csvs_into_db()
    return seed_demo_users()


def require_login() -> dict:
    """Login gate. Every page calls this immediately after
    inject_base_style(). Returns the logged-in user's record; halts page
    execution (st.stop()) and renders a login form if no session exists.
    See sentinel/auth.py for what's actually enforced vs. what's a
    documented upgrade path."""
    demo_creds = _ensure_demo_users_seeded()

    if st.session_state.get("user") is None:
        theme_toggle_ui()
        left, mid, right = st.columns([1, 1.3, 1])
        with mid:
            st.markdown(
                """
                <div style="text-align:center; margin-top:1.2rem; margin-bottom:0.6rem;">
                  <div style="font-size:2.6rem; line-height:1;">🛡️</div>
                  <h1 class="sentinel-hero-title" style="margin:.3rem 0 0;">MPLADS Sentinel</h1>
                  <p style="opacity:.72; margin-top:.15rem; font-size:.95rem;">
                    Sign in to the AI-assisted transparency &amp; fraud-detection dashboard
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(
                "This is a hackathon demo — accounts and passwords below are intentionally simple and "
                "shown in the open rather than hidden, since pretending this is production-hardened would "
                "be dishonest. See README 'Demo login credentials' for the same list."
            )
            with st.expander("🔑 Demo credentials (click to expand)"):
                st.code(
                    f"Central Admin (sees everything):\n"
                    f"  username: {demo_creds['admin'][0]}   password: {demo_creds['admin'][1]}\n\n"
                    f"MP office (scoped to {demo_creds['mp'][2]} only):\n"
                    f"  username: {demo_creds['mp'][0]}   password: {demo_creds['mp'][1]}\n\n"
                    f"District Officer (scoped to {demo_creds['district_officer'][2]} only):\n"
                    f"  username: {demo_creds['district_officer'][0]}   password: {demo_creds['district_officer'][1]}",
                    language=None,
                )
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign in →", width='stretch')
                if submitted:
                    user = authenticate(username, password)
                    if user is None:
                        st.error("Invalid username or password.")
                    else:
                        st.session_state.user = user
                        st.rerun()
        st.stop()

    return st.session_state.user


def sidebar_brand():
    """Small branded header block shown at the top of the sidebar (below
    Streamlit's auto-generated page nav, which always renders first)."""
    st.sidebar.markdown(
        """
        <div class="sentinel-brand">
          <div class="brand-title">🛡️&nbsp; MPLADS Sentinel</div>
          <div class="brand-sub">AI-assisted fund transparency</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def theme_toggle_ui():
    """Renders the sidebar brand block plus a light/dark toggle, and keeps
    st.session_state.theme in sync with it. Safe to call once per page
    render (require_login() calls it pre-login, show_user_badge() calls it
    post-login — never both in the same script run)."""
    sidebar_brand()
    current = get_theme()
    is_dark = st.sidebar.toggle(
        "🌙 Dark mode", value=(current == "dark"), key="_theme_toggle_switch",
        help="Switch between light and dark appearance.",
    )
    new_theme = "dark" if is_dark else "light"
    if new_theme != current:
        st.session_state.theme = new_theme
        st.rerun()
    st.sidebar.markdown("<div style='margin-bottom:.4rem;'></div>", unsafe_allow_html=True)


def show_user_badge():
    theme_toggle_ui()
    user = st.session_state.get("user")
    if not user:
        return
    role_label = {"mp": "MP Office", "district_officer": "District Officer", "central_admin": "Central Admin"}
    name = user.get("display_name", "?")
    initials = "".join([p[0] for p in name.split() if p][:2]).upper() or "?"
    st.sidebar.markdown(
        f"""
        <div class="sentinel-user-card">
          <div class="sentinel-user-avatar">{initials}</div>
          <div>
            <div class="sentinel-user-name">{name}</div>
            <div class="sentinel-user-role">{role_label.get(user['role'], user['role'])}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.sidebar.button("↪ Log out", width='stretch'):
        st.session_state.user = None
        st.rerun()


def require_role(user: dict, allowed_roles: list):
    """Call after require_login() on pages that shouldn't be open to
    every logged-in role (e.g. an MP account has no business submitting
    official district-officer verdicts). Halts with a clear message
    rather than silently hiding controls — a role check that only hides
    a button in the UI isn't a role check."""
    if user["role"] not in allowed_roles:
        st.error(
            f"This page is restricted to: {', '.join(allowed_roles)}. "
            f"Your account role is '{user['role']}'."
        )
        st.stop()


@st.cache_data(show_spinner=False)
def get_scored_dataset() -> pd.DataFrame:
    """Loads projects joined with their latest risk scores. Runs the full
    scoring pipeline once (cached) if scores don't exist yet."""
    load_csvs_into_db()
    try:
        scores = read_table("risk_scores")
        if scores.empty:
            raise ValueError("empty")
    except Exception:
        scores = None

    if scores is None or len(scores) == 0:
        composite = run_full_pipeline(verbose=False)
        scores = composite

    projects = read_table("projects")
    mps = read_table("mps")
    contractors = read_table("contractors")

    merged = projects.merge(scores, on="project_id", how="left", suffixes=("", "_score"))
    merged = merged.merge(mps[["mp_id", "mp_name", "party"]], on="mp_id", how="left")
    merged = merged.merge(
        contractors[["contractor_id", "contractor_name", "cartel_cluster"]],
        on="contractor_id", how="left",
    )
    merged["composite_score"] = merged["composite_score"].fillna(0)
    merged["risk_band"] = merged["risk_band"].fillna("LOW")
    return merged


@st.cache_data(show_spinner=False)
def get_raw_tables():
    load_csvs_into_db()
    return {
        "mps": read_table("mps"),
        "contractors": read_table("contractors"),
        "transactions": read_table("transactions"),
        "photos": read_table("photos"),
        "citizen_reports": read_table("citizen_reports"),
    }


@st.cache_data(show_spinner=False)
def get_real_mp_allocations() -> pd.DataFrame:
    """Loads the REAL, published eSAKSHI allocated-limit dataset for the
    18th Lok Sabha (543 MPs). This is factual public data — distinct from
    every other table in this app, which is synthetically generated for
    the fraud-detection demo. See app/pages/0_National_Allocation_Overview.py."""
    path = DATA_DIR / "real_mp_allocated_limits.csv"
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def get_vision_geo_results(_df: pd.DataFrame, _photos: pd.DataFrame) -> pd.DataFrame:
    """Cached wrapper around the vision/geo engine. This trains a
    cross-validated image classifier plus perceptual-hash duplicate
    matching over every evidence photo — expensive (tens of seconds).
    Streamlit reruns the whole script on *any* widget interaction, so
    without caching this recomputed from scratch every time someone so
    much as typed into a text box or flipped the theme toggle on this
    page, which looked exactly like an infinite loading loop. Args are
    prefixed with `_` so Streamlit doesn't try to hash the (large)
    DataFrames themselves — this cache is invalidated by clearing the
    Streamlit cache or restarting the app, which is the right lifetime
    for a scoring pass over a fixed demo dataset."""
    from sentinel.engines.vision_geo import run_vision_geo_engine
    return run_vision_geo_engine(_df, _photos)


@st.cache_resource(show_spinner=False)
def get_vision_corpus():
    """Cached corpus (trained classifier + phash/thumbnail index) used to
    score a brand-new photo — e.g. one a person uploads through the
    'Upload & Check a Photo' tab — against the existing evidence-photo
    corpus. st.cache_resource (not cache_data) because the payload
    includes a fitted sklearn model object we want to reuse by reference,
    not re-pickle on every access. Loads its own photos table directly
    (rather than taking one as an argument) so its cache key is simply
    'has this been built once this process', which is the right lifetime
    for a fixed demo corpus."""
    from sentinel.engines.vision_geo import build_vision_corpus
    load_csvs_into_db()
    photos = read_table("photos")
    return build_vision_corpus(photos)


@st.cache_data(show_spinner=False)
def get_contractor_graph(_contractors: pd.DataFrame):
    """Cached NetworkX graph of contractor shared-identity links. Cheap in
    absolute terms, but cached anyway so it isn't silently rebuilt (twice
    — once here, once inside get_contractor_network_scores) on every
    widget interaction on the Contractor Network page."""
    from sentinel.engines.contractor_graph import build_contractor_graph
    return build_contractor_graph(_contractors)


@st.cache_data(show_spinner=False)
def get_contractor_network_scores(_df: pd.DataFrame, _contractors: pd.DataFrame):
    from sentinel.engines.contractor_graph import run_contractor_network_engine
    return run_contractor_network_engine(_df, _contractors)


# ---------------------------------------------------------------------------
# Theming — light/dark palettes + a big, theme-aware CSS re-skin of Streamlit
# ---------------------------------------------------------------------------
THEMES = {
    "light": {
        "bg_gradient": "linear-gradient(180deg, #F5F7FB 0%, #ECF0F9 100%)",
        "bg": "#F5F7FB",
        "card": "#FFFFFF",
        "card_border": "#E3E7F1",
        "sidebar_bg": "#FFFFFF",
        "sidebar_border": "#E3E7F1",
        "text": "#1A1F36",
        "text_muted": "#5B6472",
        "heading": "#372E8A",
        "primary": "#4F46E5",
        "primary_dark": "#372E8A",
        "accent": "#0EA5B5",
        "shadow": "0 2px 10px rgba(20,25,50,0.06)",
        "shadow_hover": "0 10px 24px rgba(20,25,50,0.12)",
        "hover_bg": "#EEF1FA",
        "input_bg": "#FFFFFF",
        "divider": "#E3E7F1",
        "code_bg": "#F0F2F8",
    },
    "dark": {
        "bg_gradient": "linear-gradient(180deg, #0E1220 0%, #131A2E 100%)",
        "bg": "#0E1220",
        "card": "#161D31",
        "card_border": "#262F49",
        "sidebar_bg": "#10152B",
        "sidebar_border": "#232C46",
        "text": "#E7EBF7",
        "text_muted": "#96A0BD",
        "heading": "#C7D2FE",
        "primary": "#818CF8",
        "primary_dark": "#6366F1",
        "accent": "#2DD4DA",
        "shadow": "0 2px 10px rgba(0,0,0,0.35)",
        "shadow_hover": "0 12px 28px rgba(0,0,0,0.5)",
        "hover_bg": "#1B2340",
        "input_bg": "#141B2E",
        "divider": "#232C46",
        "code_bg": "#111729",
    },
}


def get_theme() -> str:
    return st.session_state.get("theme", "light")


def get_palette() -> dict:
    return THEMES[get_theme()]


_CSS_TEMPLATE = Template(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Manrope:wght@700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* ---- App background & header ---- */
[data-testid="stAppViewContainer"] { background: ${bg_gradient} !important; }
[data-testid="stHeader"] {
    background: transparent !important;
}
[data-testid="stAppViewContainer"], .main, .main p, .main li, .main span, .main label {
    color: ${text};
}
[data-testid="stCaptionContainer"], .stCaption, small {
    color: ${text_muted} !important;
}
/* Widget labels (the "Username", "Select a district" text above every
   input) come from Streamlit's own internal styling, which reads the
   FIXED light-theme textColor from .streamlit/config.toml rather than
   our dark/light toggle — so in dark mode these were rendering
   near-black text on a near-black background: readable-in-theory,
   invisible-in-practice. Force them to follow our actual active theme. */
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span,
[data-testid="stWidgetLabel"] label {
    color: ${text} !important;
}

/* ---- Layout / responsiveness ---- */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
    animation: sentinelFadeIn .45s ease;
}
@keyframes sentinelFadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
@media (max-width: 768px) {
    .main .block-container { padding-left: 1rem; padding-right: 1rem; padding-top: 1.1rem; }
    h1 { font-size: 1.5rem !important; }
    h2 { font-size: 1.2rem !important; }
    [data-testid="stMetricValue"] { font-size: 1.1rem !important; }
}
@media (min-width: 1800px) {
    .main .block-container { max-width: 1600px; }
}

/* ---- Headings ---- */
h1, h2, h3 {
    font-family: 'Manrope', 'Inter', sans-serif !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
    color: ${heading};
}
h1 {
    font-size: clamp(1.5rem, 1.1rem + 1.6vw, 2.3rem) !important;
}
/* Reserved for the actual brand name (login hero) — everywhere else uses
   a solid heading color. A gradient on every single page title (this app
   has one on all ten pages) reads as noisy rather than "branded"; used
   once, at the one moment that's genuinely a logo/wordmark, it reads as
   intentional. */
.sentinel-hero-title {
    background: linear-gradient(90deg, ${primary}, ${accent});
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: ${sidebar_bg} !important;
    border-right: 1px solid ${sidebar_border};
}
[data-testid="stSidebar"] > div { color: ${text}; }
[data-testid="stSidebarNavLink"] p, [data-testid="stSidebarNavLink"] span {
    color: ${text} !important;
}
[data-testid="stSidebarNavLink"] {
    border-radius: 8px !important;
}
[data-testid="stSidebarNavLink"]:hover { background: ${hover_bg} !important; }
[data-testid="stSidebarNavLink"][aria-current="page"] { background: ${hover_bg} !important; }
[data-testid="stSidebarNavLink"][aria-current="page"] p {
    color: ${primary} !important; font-weight: 700 !important;
}
.sentinel-brand {
    padding: 6px 4px 14px; margin-bottom: 10px;
    border-bottom: 1px solid ${sidebar_border};
}
.sentinel-brand .brand-title {
    font-family: 'Manrope', sans-serif; font-weight: 800; font-size: 1.05rem; color: ${text};
}
.sentinel-brand .brand-sub { font-size: .72rem; color: ${text_muted}; margin-top: 2px; }
.sentinel-user-card {
    background: ${hover_bg}; border: 1px solid ${card_border}; border-radius: 12px;
    padding: 10px 12px; margin: 4px 0 10px; display: flex; gap: 10px; align-items: center;
}
.sentinel-user-avatar {
    width: 34px; height: 34px; border-radius: 50%; flex-shrink: 0;
    background: linear-gradient(135deg, ${primary}, ${accent});
    color: #fff; display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: .82rem;
}
.sentinel-user-name { font-weight: 700; font-size: .85rem; color: ${text}; }
.sentinel-user-role { font-size: .72rem; color: ${text_muted}; }

/* ---- Metric cards ---- */
[data-testid="stMetric"] {
    background: ${card} !important;
    border: 1px solid ${card_border};
    border-radius: 16px;
    padding: 16px 18px 14px;
    box-shadow: ${shadow};
    position: relative; overflow: hidden;
    transition: transform .18s ease, box-shadow .18s ease;
}
[data-testid="stMetric"]::before {
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, ${primary}, ${accent});
}
[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: ${shadow_hover};
}
[data-testid="stMetricLabel"] {
    color: ${text_muted} !important; font-size: .74rem !important;
    text-transform: uppercase; letter-spacing: .06em; font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    color: ${text} !important; font-weight: 800 !important;
    font-size: clamp(1.2rem, .9rem + 1vw, 1.8rem) !important;
}

/* ---- Buttons ---- */
.stButton button, .stDownloadButton button, .stFormSubmitButton button {
    background: linear-gradient(135deg, ${primary}, ${primary_dark}) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.2rem !important;
    box-shadow: 0 2px 8px ${primary}55;
    transition: transform .15s ease, box-shadow .15s ease, filter .15s ease;
}
.stButton button:hover, .stDownloadButton button:hover, .stFormSubmitButton button:hover {
    transform: translateY(-2px);
    filter: brightness(1.08);
    box-shadow: 0 6px 18px ${primary}66;
}
.stButton button:active, .stFormSubmitButton button:active { transform: translateY(0); }

/* ---- Inputs ---- */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stNumberInput"] input {
    background: ${input_bg} !important;
    color: ${text} !important;
    border: 1.5px solid ${card_border} !important;
    border-radius: 10px !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus,
[data-testid="stNumberInput"] input:focus {
    border-color: ${primary} !important;
    box-shadow: 0 0 0 3px ${primary}30 !important;
}
/* Streamlit overlays a "Press Enter to..." hint *inside* the input box
   itself (transparent background, absolutely positioned) without
   reserving any space for it — so it sits directly on top of whatever
   was just typed. In Streamlit's own default theme this is pale gray on
   white and barely noticeable; our global text-color rules above made it
   render in full-contrast text color instead, which made an
   always-slightly-broken-looking overlap into a glaring one. There's no
   layout fix that actually prevents the overlap (Streamlit doesn't
   reserve the space), so this hides it outright rather than half-fixing
   the symptom. */
[data-testid="InputInstructions"] { display: none !important; }
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div,
[data-testid="stSelectbox"] [role="group"],
[data-testid="stMultiSelect"] [role="group"] {
    background: ${input_bg} !important;
    border-color: ${card_border} !important;
    border: 1px solid ${card_border} !important;
    border-radius: 10px !important;
    color: ${text} !important;
}
[data-testid="stSelectbox"] input,
[data-testid="stMultiSelect"] input {
    background: transparent !important;
    color: ${text} !important;
}
[data-testid="stSelectbox"] svg, [data-testid="stMultiSelect"] svg {
    fill: ${text_muted} !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: ${hover_bg} !important;
    border: 1.5px dashed ${card_border} !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background: ${card} !important;
    color: ${text} !important;
    border: 1.5px solid ${card_border} !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    border-color: ${primary} !important;
    color: ${primary} !important;
}
/* Same root cause as InputInstructions above: Streamlit renders this
   hint ("200MB per file...") using the fixed config.toml light-theme
   text color at 60% opacity, regardless of our dark/light toggle — on a
   dark dropzone background that's dark-on-dark, nearly unreadable. */
[data-testid="stFileUploaderDropzoneInstructions"] span {
    color: ${text_muted} !important;
}

/* ---- Expanders ---- */
[data-testid="stExpander"] {
    border: 1px solid ${card_border} !important;
    border-radius: 14px !important;
    background: ${card} !important;
    box-shadow: ${shadow};
    overflow: hidden;
}
[data-testid="stExpander"] summary { font-weight: 600; color: ${text} !important; }
[data-testid="stExpander"] summary:hover { background: ${hover_bg}; }

/* ---- Tabs ---- */
/* Streamlit's tabs are `[data-testid="stTab"]` divs (not <button
   role="tab">, an older internal structure this app's CSS briefly
   targeted and which no longer exists in this Streamlit version) with
   aria-selected/data-selected marking the active one. */
[data-testid="stTab"] { color: ${text_muted} !important; font-weight: 600; }
[data-testid="stTab"] p { color: inherit !important; }
[data-testid="stTab"][aria-selected="true"],
[data-testid="stTab"][data-selected="true"] {
    color: ${primary} !important;
}
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid ${divider};
}
[data-testid="stTab"] .react-aria-SelectionIndicator {
    background: ${primary} !important;
}

/* ---- Alerts / expanders / dataframe / plotly / images as "cards" ---- */
[data-testid="stAlert"] { border-radius: 12px !important; }
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {
    border: 1px solid ${card_border};
    border-radius: 14px;
    padding: 6px;
    background: ${card};
    box-shadow: ${shadow};
}
[data-testid="stImage"] img { border-radius: 12px; box-shadow: ${shadow}; }

/* ---- Divider / code ---- */
hr { border-color: ${divider} !important; margin: 1.3rem 0 !important; }
/* Streamlit's code block wrapper testid is "stCode" in this version
   ("stCodeBlock" — this rule's original selector — doesn't exist here,
   so this was silently a no-op, leaving st.code() blocks in their
   default near-white styling regardless of theme). */
[data-testid="stCode"] pre, [data-testid="stCode"] code {
    background: ${code_bg} !important;
    color: ${text} !important;
    border-radius: 10px !important;
}
code { background: ${code_bg} !important; border-radius: 4px !important; }

/* ---- Risk pills ---- */
.risk-pill {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 13px; border-radius: 999px;
    font-weight: 700; font-size: .78rem; color: #fff;
    letter-spacing: .02em; box-shadow: 0 2px 6px rgba(0,0,0,.18);
    white-space: nowrap;
}
.risk-pill.critical { animation: sentinelPulse 1.8s ease-in-out infinite; }
@keyframes sentinelPulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(198,40,40,.55); }
    50% { box-shadow: 0 0 0 6px rgba(198,40,40,0); }
}

/* ---- Raw HTML tables (to_html output) ---- */
.sentinel-table-wrap {
    overflow-x: auto; -webkit-overflow-scrolling: touch;
    border: 1px solid ${card_border}; border-radius: 14px;
    box-shadow: ${shadow}; background: ${card}; margin: .5rem 0 1.2rem;
}
.sentinel-table-wrap table { width: 100%; border-collapse: collapse; font-size: .87rem; }
.sentinel-table-wrap thead th {
    position: sticky; top: 0; background: ${hover_bg}; color: ${text};
    text-align: left; padding: 10px 14px; font-weight: 700;
    border-bottom: 2px solid ${card_border}; white-space: nowrap;
}
.sentinel-table-wrap tbody td {
    padding: 9px 14px; border-bottom: 1px solid ${card_border};
    color: ${text}; white-space: nowrap;
}
.sentinel-table-wrap tbody tr:hover { background: ${hover_bg}; }
@media (max-width: 640px) { .sentinel-table-wrap table { font-size: .76rem; } }

/* ---- Scrollbar ---- */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: ${bg}; }
::-webkit-scrollbar-thumb { background: ${card_border}; border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: ${primary}; }
</style>
""")


def inject_base_style():
    pal = get_palette()
    st.markdown(_CSS_TEMPLATE.substitute(pal), unsafe_allow_html=True)


def apply_plot_theme(fig, height: int = None):
    """Strips a plotly figure's background so it sits flush inside our
    themed card container, and colors text/gridlines to match the active
    light/dark palette. Call right before st.plotly_chart(fig, ...)."""
    pal = get_palette()
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=pal["text"], family="Inter, -apple-system, sans-serif"),
        legend=dict(font=dict(color=pal["text"])),
    )
    fig.update_xaxes(gridcolor=pal["divider"], zerolinecolor=pal["divider"], color=pal["text_muted"])
    fig.update_yaxes(gridcolor=pal["divider"], zerolinecolor=pal["divider"], color=pal["text_muted"])
    if height:
        fig.update_layout(height=height)
    return fig


def render_html_table(html: str):
    """Wraps a df.to_html(...) string in a styled, horizontally-scrollable
    card so raw HTML tables match the rest of the theme on any screen size."""
    st.markdown(f'<div class="sentinel-table-wrap">{html}</div>', unsafe_allow_html=True)


RISK_COLORS = {"LOW": "#2e7d32", "MEDIUM": "#f9a825", "HIGH": "#ef6c00", "CRITICAL": "#c62828"}
RISK_ICONS = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}


def risk_pill(band: str) -> str:
    color = RISK_COLORS.get(band, "#888")
    icon = RISK_ICONS.get(band, "⚪")
    extra_class = " critical" if band == "CRITICAL" else ""
    return f'<span class="risk-pill{extra_class}" style="background:{color}">{icon} {band}</span>'
