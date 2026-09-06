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
from i18n import t, language_switcher_sidebar


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
            accent = get_palette()["accent"]
            st.markdown(
                f"""
                <div style="text-align:center; margin-top:1.2rem; margin-bottom:1.3rem;">
                  <div style="font-size:2.2rem; line-height:1; opacity:.85;">🛡️</div>
                  <h1 class="sentinel-hero-title" style="margin:.35rem 0 0;">MPLADS Sentinel</h1>
                  <div style="width:40px; height:3px; background:{accent};
                              margin:.55rem auto 0; border-radius:2px;"></div>
                  <p style="opacity:.72; margin-top:.65rem; font-size:.95rem;">
                    {t('login_tagline')}
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            tab_signin, tab_demo = st.tabs([t("sign_in"), t("demo_accounts")])

            with tab_signin:
                with st.form("login_form"):
                    username = st.text_input(t("username_lbl"))
                    password = st.text_input(t("password_lbl"), type="password")
                    submitted = st.form_submit_button(t("sign_in_btn"), width='stretch')
                    if submitted:
                        user = authenticate(username, password)
                        if user is None:
                            st.error("Invalid username or password.")
                        else:
                            st.session_state.user = user
                            st.rerun()

            with tab_demo:
                st.markdown(
                    "<p class='sentinel-cred-note'>This is a hackathon demo — the accounts below are "
                    "intentionally simple and shown in the open rather than hidden, since pretending "
                    "this is production-hardened would be dishonest. Pick a role and copy its username "
                    "and password into the Sign in tab.</p>",
                    unsafe_allow_html=True,
                )
                roles = [
                    ("👑", t("central_admin"), "Sees every project nationally",
                     demo_creds["admin"][0], demo_creds["admin"][1]),
                    ("🧑‍💼", t("mp_office"), f"Scoped to {demo_creds['mp'][2]} only",
                     demo_creds["mp"][0], demo_creds["mp"][1]),
                    ("🗂️", t("district_officer"), f"Scoped to {demo_creds['district_officer'][2]} only",
                     demo_creds["district_officer"][0], demo_creds["district_officer"][1]),
                ]
                cards_html = "".join(
                    '<div class="sentinel-cred-card">'
                    f'<div class="role-row"><span class="role-icon">{icon}</span>{role}</div>'
                    f'<div class="scope">{scope}</div>'
                    f'<div class="cred-row"><span class="sentinel-cred-label">{t("username_lbl")}</span>'
                    f'<span class="sentinel-cred-value">{u}</span></div>'
                    f'<div class="cred-row"><span class="sentinel-cred-label">{t("password_lbl")}</span>'
                    f'<span class="sentinel-cred-value">{p}</span></div>'
                    '</div>'
                    for icon, role, scope, u, p in roles
                )
                st.markdown(f'<div class="sentinel-cred-grid">{cards_html}</div>', unsafe_allow_html=True)
        st.stop()

    return st.session_state.user


def sidebar_brand():
    """Small branded header block shown at the top of the sidebar (below
    Streamlit's auto-generated page nav, which always renders first).
    Uses the ambient `st` rather than `st.sidebar` so it correctly nests
    inside theme_toggle_ui()'s st.sidebar.container() — calling
    st.sidebar.xxx explicitly here would instead escape that container
    and land directly in the sidebar root, breaking the scroll-split
    described in the CSS."""
    st.markdown(
        f"""
        <div class="sentinel-brand">
          <div class="brand-title">🛡️&nbsp; MPLADS Sentinel</div>
          <div class="brand-sub">{t('brand_tagline')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def theme_toggle_ui():
    """Renders the sidebar brand block plus a light/dark toggle and the
    language switcher, inside an explicit st.sidebar.container() — this
    becomes the FIRST of exactly two sibling containers under the
    sidebar's "user content" area (the second, added by show_user_badge()
    right after this returns, holds the profile card + log-out button).
    That's not cosmetic: the CSS in _CSS_TEMPLATE targets these two
    containers by position (first-of-type / last-of-type) to make this
    one scroll independently while the profile+logout container below it
    stays pinned to the bottom of the sidebar — see the "Sidebar
    scroll-split" comment in the CSS for the full mechanism.

    Keeps st.session_state.theme in sync. Safe to call once per page
    render (require_login() calls it pre-login, show_user_badge() calls
    it post-login — never both in the same script run)."""
    with st.sidebar.container():
        sidebar_brand()
        current = get_theme()
        is_dark = st.toggle(
            f"🌙 {t('dark_mode')}", value=(current == "dark"), key="_theme_toggle_switch",
            help="Switch between light and dark appearance.",
        )
        new_theme = "dark" if is_dark else "light"
        if new_theme != current:
            st.session_state.theme = new_theme
            st.rerun()
        language_switcher_sidebar()


def show_user_badge():
    theme_toggle_ui()
    user = st.session_state.get("user")
    if not user:
        return
    role_label = {"mp": t("mp_office"), "district_officer": t("district_officer"), "central_admin": t("central_admin")}
    name = user.get("display_name", "?")
    initials = "".join([p[0] for p in name.split() if p][:2]).upper() or "?"
    with st.sidebar.container():
        st.markdown(
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
        if st.button(t("log_out"), width='stretch'):
            st.session_state.user = None
            st.rerun()


def page_header(icon: str, title: str, subtitle: str = None):
    """Renders the shared page masthead: icon + serif title, a single
    accent rule, and an optional subtitle — used in place of a bare
    st.title()/st.caption() pair so every page in the app shares one
    consistent identity instead of nine independent headings."""
    sub_html = f'<div class="masthead-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="sentinel-masthead">
          <div class="kicker">
            <span class="kicker-icon">{icon}</span>
            <h1>{title}</h1>
          </div>
          {sub_html}
          <div class="masthead-rule"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
        "bg_gradient": "linear-gradient(180deg, #F6F7F9 0%, #F0F2F5 100%)",
        "bg": "#F6F7F9",
        "card": "#FFFFFF",
        "card_border": "#E3E6EB",
        "sidebar_bg": "#FFFFFF",
        "sidebar_border": "#E3E6EB",
        "text": "#1A1D24",
        "text_muted": "#66707C",
        "heading": "#11141B",
        "primary": "#2952CC",
        "primary_dark": "#1E3D99",
        "accent": "#3F6FE0",
        "shadow": "0 1px 2px rgba(16,20,30,.05), 0 1px 0 rgba(16,20,30,.04)",
        "shadow_hover": "0 8px 20px rgba(16,20,30,.10)",
        "hover_bg": "#EEF1F6",
        "input_bg": "#FFFFFF",
        "divider": "#E3E6EB",
        "code_bg": "#EEF1F6",
    },
    "dark": {
        "bg_gradient": "linear-gradient(180deg, #0A0D13 0%, #10141C 100%)",
        "bg": "#0A0D13",
        "card": "#141822",
        "card_border": "#262C3A",
        "sidebar_bg": "#0F131B",
        "sidebar_border": "#242A38",
        "text": "#E7E9EE",
        "text_muted": "#8B93A7",
        "heading": "#F5F6F9",
        "primary": "#5B84E8",
        "primary_dark": "#3457C4",
        "accent": "#7098F0",
        "shadow": "0 1px 3px rgba(0,0,0,.5)",
        "shadow_hover": "0 10px 24px rgba(0,0,0,.55)",
        "hover_bg": "#1A2030",
        "input_bg": "#141A24",
        "divider": "#242A38",
        "code_bg": "#121722",
    },
}


def get_theme() -> str:
    return st.session_state.get("theme", "light")


def get_palette() -> dict:
    return THEMES[get_theme()]


_CSS_TEMPLATE = Template(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Sora:wght@500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-feature-settings: "tnum" 1;
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
/* A distinct heading face (Sora) from the Inter body text gives the app
   one considered typographic identity instead of the flat single-font
   look — this cascades to every st.title/st.subheader on all nine pages
   from this one rule. Clean geometric sans, not serif/italic: the goal
   here is "confident modern dashboard", not "gazette". */
h1, h2, h3 {
    font-family: 'Sora', 'Inter', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.015em;
    color: ${heading};
}
h1 {
    font-size: clamp(1.5rem, 1.15rem + 1.4vw, 2.2rem) !important;
    font-weight: 700 !important;
}
.sentinel-hero-title {
    color: ${heading} !important;
}
.sentinel-masthead {
    border-bottom: 1px solid ${divider};
    padding-bottom: .85rem;
    margin-bottom: 1.4rem;
}
.sentinel-masthead .kicker {
    display: flex; align-items: center; gap: 10px;
}
.sentinel-masthead .kicker-icon { font-size: 1.5rem; line-height: 1; }
.sentinel-masthead h1 {
    margin: 0 !important;
}
.sentinel-masthead .masthead-sub {
    color: ${text_muted}; font-size: .92rem; margin-top: .35rem; max-width: 74ch;
}
.sentinel-masthead .masthead-rule {
    width: 40px; height: 3px; background: ${primary}; margin-top: .7rem; border-radius: 2px;
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
    position: relative;
}
[data-testid="stSidebarNavLink"]:hover { background: ${hover_bg} !important; }
[data-testid="stSidebarNavLink"][aria-current="page"] { background: ${hover_bg} !important; }
[data-testid="stSidebarNavLink"][aria-current="page"] p {
    color: ${primary} !important; font-weight: 700 !important;
}
/* Active-page indicator: a short accent bar on the left edge of the
   current nav item, the common "you are here" pattern in professional
   dashboard sidebars (Linear, Notion, Vercel etc.) rather than relying
   on background tint alone. */
[data-testid="stSidebarNavLink"][aria-current="page"]::before {
    content: ""; position: absolute; left: -8px; top: 6px; bottom: 6px; width: 3px;
    background: ${primary}; border-radius: 2px;
}
.sentinel-brand {
    padding: 6px 4px 14px; margin-bottom: 10px;
    border-bottom: 1px solid ${sidebar_border};
}
.sentinel-brand .brand-title {
    font-family: 'Sora', sans-serif; font-weight: 700; font-size: 1.05rem; color: ${text};
    letter-spacing: -0.01em;
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

/* ---- Sidebar scroll-split ----
   Streamlit renders everything after st.sidebar.* calls into one
   [data-testid="stSidebarUserContent"] region, and by default the WHOLE
   sidebar ([data-testid="stSidebarContent"]) is one scrolling box — nav
   links, brand, theme/language controls, profile card and log-out
   button all in a single scroll, so on a short window the log-out
   button is the last thing you can scroll down to reach. That's the
   exact complaint this fixes.

   common.py wraps the two sidebar sections in their own explicit
   st.sidebar.container() calls: theme_toggle_ui() (brand/theme/language)
   is the first container, show_user_badge()'s profile-card+log-out is
   the second and last — giving the CSS below two concrete, positional
   hooks (first-of-type / last-of-type) instead of anything fragile like
   a generated class name.

   The sidebar becomes a fixed-height flex column instead of one long
   scrolling block: the page-nav list keeps its natural size (scrolling
   internally only in the unlikely case it's very long itself), the
   first container (brand/theme/language) gets whatever space is left
   over and scrolls internally ONLY if it doesn't fit, and the second
   container (profile+log-out) is flex-shrink:0 — meaning it's always
   rendered at full size and always fully visible with zero scrolling.
   A position:sticky element would NOT guarantee that: sticky only stops
   an element sliding away *after* you've already scrolled to it, it
   doesn't pull something into view from further away — which is
   exactly the case here on a short window, so this needs the real
   flex split below rather than the simpler-looking sticky shortcut. */
[data-testid="stSidebarContent"] {
    display: flex;
    flex-direction: column;
    overflow: hidden !important;
    height: 100vh;
}
[data-testid="stSidebarHeader"] { flex-shrink: 0; }
[data-testid="stSidebarNav"] {
    flex-shrink: 0;
    max-height: 46vh;
    overflow-y: auto;
}
[data-testid="stSidebarUserContent"] {
    flex: 1 1 auto;
    min-height: 0;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    padding-bottom: .5rem;
}
[data-testid="stSidebarUserContent"] > div {
    flex: 1 1 auto; min-height: 0; display: flex; flex-direction: column; overflow: hidden;
}
[data-testid="stSidebarUserContent"] > div > [data-testid="stVerticalBlock"] {
    flex: 1 1 auto; min-height: 0; display: flex; flex-direction: column; overflow: hidden;
}
[data-testid="stSidebarUserContent"] > div > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:first-of-type {
    flex: 1 1 auto;
    min-height: 0;
    overflow-y: auto;
}
[data-testid="stSidebarUserContent"] > div > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:last-of-type {
    flex-shrink: 0;
    padding-top: .5rem;
}

/* ---- Metric cards ---- */
[data-testid="stMetric"] {
    background: ${card} !important;
    border: 1px solid ${card_border};
    border-left: 3px solid ${primary};
    border-radius: 10px;
    padding: 14px 18px 12px;
    box-shadow: ${shadow};
    position: relative;
    transition: box-shadow .18s ease, transform .18s ease;
}
[data-testid="stMetric"]:hover {
    box-shadow: ${shadow_hover};
    transform: translateY(-1px);
}
[data-testid="stMetricLabel"] {
    color: ${text_muted} !important; font-size: .74rem !important;
    text-transform: uppercase; letter-spacing: .05em; font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    color: ${text} !important; font-weight: 700 !important;
    font-variant-numeric: tabular-nums;
    font-size: clamp(1.2rem, .9rem + 1vw, 1.75rem) !important;
}

/* ---- Buttons ---- */
.stButton button, .stDownloadButton button, .stFormSubmitButton button {
    background: ${primary} !important;
    color: ${card} !important;
    border: 1px solid ${primary} !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.2rem !important;
    box-shadow: none;
    transition: background .15s ease, border-color .15s ease;
}
.stButton button:hover, .stDownloadButton button:hover, .stFormSubmitButton button:hover {
    background: ${accent} !important;
    border-color: ${accent} !important;
}
.stButton button:active, .stFormSubmitButton button:active { background: ${primary_dark} !important; }

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
    border-radius: 10px !important;
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
    border-radius: 10px;
    padding: 6px;
    background: ${card};
    box-shadow: ${shadow};
}
[data-testid="stImage"] img { border-radius: 6px; box-shadow: ${shadow}; }

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
    letter-spacing: .02em; box-shadow: 0 1px 3px rgba(0,0,0,.15);
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
    border: 1px solid ${card_border}; border-radius: 10px;
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

/* ---- Auth screen: sign-in form card + demo-account credential cards ---- */
[data-testid="stForm"] {
    background: ${card} !important;
    border: 1px solid ${card_border} !important;
    border-radius: 10px !important;
    padding: 1.3rem 1.4rem 0.6rem !important;
    box-shadow: ${shadow};
}
.sentinel-cred-note {
    font-size: .82rem; color: ${text_muted}; line-height: 1.5;
    margin: .2rem 0 1rem;
}
.sentinel-cred-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 12px;
}
.sentinel-cred-card {
    border: 1px solid ${card_border}; border-radius: 10px;
    background: ${card}; padding: 14px 16px; box-shadow: ${shadow};
}
.sentinel-cred-card .role-row {
    display: flex; align-items: center; gap: 8px;
    font-weight: 700; color: ${heading}; font-size: .95rem;
}
.sentinel-cred-card .role-icon { font-size: 1.05rem; }
.sentinel-cred-card .scope {
    font-size: .76rem; color: ${text_muted}; margin: 2px 0 10px;
}
.sentinel-cred-card .cred-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 7px 0; border-top: 1px dashed ${divider};
}
.sentinel-cred-card .cred-row:first-of-type { border-top: 1px solid ${divider}; }
.sentinel-cred-label {
    color: ${text_muted}; font-size: .72rem; text-transform: uppercase; letter-spacing: .04em;
}
.sentinel-cred-value {
    font-family: 'SF Mono', 'Consolas', 'Menlo', monospace; font-size: .82rem;
    color: ${text}; background: ${hover_bg}; padding: 2px 8px; border-radius: 5px;
}

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
