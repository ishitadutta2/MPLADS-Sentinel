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
from i18n import t, language_switcher


# ---------------------------------------------------------------------------
# Icon system: this app uses no emoji anywhere. Everywhere Streamlit renders
# plain text (buttons, tabs, alerts, chat avatars, popover labels, toggle
# labels) we use its built-in ":material/name:" shorthand, which Streamlit
# itself renders as a proper vector icon — see each call site below.
# That shorthand is NOT parsed inside raw unsafe_allow_html markup though
# (verified empirically), so the handful of places that build their own
# HTML (the masthead icon, the brand mark, risk badges, login role cards)
# instead pull a small hand-drawn icon from ICON_SVGS via svg_icon().
# ---------------------------------------------------------------------------
ICON_SVGS = {
    # stroke-style (outline) icons, 24x24 viewbox, Feather-icon-inspired
    "shield": '<path d="M12 2 20 6 V11 C20 16 16.5 20.5 12 22 7.5 20.5 4 16 4 11 V6 Z"/>',
    "lock": '<rect x="4" y="11" width="16" height="10" rx="2"/><path d="M7.5 11V7.5a4.5 4.5 0 0 1 9 0V11"/>',
    "chat": '<path d="M21 11.5a8.38 8.38 0 0 1-4.9 7.6 8.5 8.5 0 0 1-7.6-.1L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 8.5-8.5h.5a8.48 8.48 0 0 1 8 8z"/>',
    "briefcase": '<rect x="2" y="7" width="20" height="14" rx="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/>',
    "folder": '<path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>',
    "award": '<circle cx="12" cy="8" r="6.5"/><path d="M8.4 13.7 7 22l5-3 5 3-1.4-8.3"/>',
    "landmark": '<path d="M3 21h18M4 10h16M12 3 2 9h20zM5 10v8M9.5 10v8M14.5 10v8M19 10v8"/>',
    "link": '<path d="M10 13a5 5 0 0 0 7.5.5l2.5-2.5a5 5 0 0 0-7-7l-1.5 1.5"/><path d="M14 11a5 5 0 0 0-7.5-.5L4 13a5 5 0 0 0 7 7l1.5-1.5"/>',
    "camera": '<path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/>',
    "hub": '<circle cx="18" cy="5" r="2.7"/><circle cx="6" cy="12" r="2.7"/><circle cx="18" cy="19" r="2.7"/><path d="M8.4 13.4l7 3.7M15.4 7.1l-7 3.7"/>',
    "search": '<circle cx="11" cy="11" r="7.5"/><path d="M21 21l-4.8-4.8"/>',
    "clock": '<circle cx="12" cy="12" r="9.5"/><path d="M12 7v5l3.5 2"/>',
    "bot": '<rect x="4" y="8" width="16" height="12" rx="2.5"/><circle cx="9" cy="14" r="1.3"/><circle cx="15" cy="14" r="1.3"/><path d="M12 8V4.5"/><circle cx="12" cy="3.2" r="1.2"/>',
    "person": '<path d="M19.5 21v-2a4 4 0 0 0-4-4h-7a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    "list": '<path d="M9 6h12M9 12h12M9 18h12M4 6h.01M4 12h.01M4 18h.01"/>',
    "flag": '<path d="M4 21V4h1.5l13 5-13 5"/>',
    "add": '<path d="M12 5v14M5 12h14"/>',
    "arrow-right": '<path d="M5 12h14M13 5l7 7-7 7"/>',
    "home": '<path d="M3 11.5 12 4l9 7.5"/><path d="M5.5 10v10h13V10"/>',
    "check-circle": '<circle cx="12" cy="12" r="9.5"/><path d="M7.5 12.5l3 3 6-6.5"/>',
    "delete": '<path d="M4 7h16M9 7V4.5A1.5 1.5 0 0 1 10.5 3h3A1.5 1.5 0 0 1 15 4.5V7m2 0-.8 12.4A2 2 0 0 1 14.2 21H9.8a2 2 0 0 1-2-1.6L7 7"/>',
    "explore": '<circle cx="12" cy="12" r="9.5"/><path d="M15.5 8.5 13.7 13.7 8.5 15.5l1.8-5.2z"/>',
    "globe": '<circle cx="12" cy="12" r="9.5"/><path d="M2.5 12h19"/><path d="M12 2.5c2.5 2.7 3.9 6.1 3.9 9.5s-1.4 6.8-3.9 9.5c-2.5-2.7-3.9-6.1-3.9-9.5S9.5 5.2 12 2.5z"/>',
}


def svg_icon(name: str, size: int = 18, color: str = "currentColor", stroke_width: float = 2) -> str:
    """Returns an inline <svg> for the handful of raw-HTML (unsafe_allow_html)
    spots that can't use Streamlit's ":material/name:" text shorthand.
    Fill-style icons (currently just "dot", the risk-severity marker) take
    `color` as a fill; everything else in ICON_SVGS is a stroke outline."""
    if name == "dot":
        return (f'<svg viewBox="0 0 24 24" width="{size}" height="{size}" '
                 f'style="display:inline-block;vertical-align:-2px;"><circle cx="12" cy="12" r="9" '
                 f'fill="{color}"/></svg>')
    inner = ICON_SVGS.get(name, ICON_SVGS["shield"])
    return (
        f'<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" stroke="{color}" '
        f'stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round" '
        f'style="display:inline-block;vertical-align:-3px;">{inner}</svg>'
    )


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
        header_controls()
        left, mid, right = st.columns([1, 1.3, 1])
        with mid:
            accent = get_palette()["accent"]
            st.markdown(
                f"""
                <div style="text-align:center; margin-top:1.2rem; margin-bottom:1.3rem;">
                  <div style="opacity:.85;">{svg_icon('shield', size=40)}</div>
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
                    submitted = st.form_submit_button(t("sign_in_btn"), width='stretch', icon=":material/login:")
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
                    ("award", t("central_admin"), "Sees every project nationally",
                     demo_creds["admin"][0], demo_creds["admin"][1]),
                    ("briefcase", t("mp_office"), f"Scoped to {demo_creds['mp'][2]} only",
                     demo_creds["mp"][0], demo_creds["mp"][1]),
                    ("folder", t("district_officer"), f"Scoped to {demo_creds['district_officer'][2]} only",
                     demo_creds["district_officer"][0], demo_creds["district_officer"][1]),
                ]
                cards_html = "".join(
                    '<div class="sentinel-cred-card">'
                    f'<div class="role-row"><span class="role-icon">{svg_icon(icon, size=18)}</span>{role}</div>'
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


def render_landing_gate():
    """First screen anyone hitting the app sees, before any account
    exists in the session — a choice between the official/staff sign-in
    flow (require_login(), unchanged) and the no-login public Citizen
    Chatbot. Only shown from Home.py, and only until either button is
    clicked once this session — see the `show_login_form` /
    `pc_stage` flags this sets."""
    header_controls()
    accent = get_palette()["accent"]
    st.markdown(
        f"""
        <div style="text-align:center; margin-top:1.2rem; margin-bottom:1.4rem;">
          <div style="opacity:.85;">{svg_icon('shield', size=40)}</div>
          <h1 class="sentinel-hero-title" style="margin:.35rem 0 0;">MPLADS Sentinel</h1>
          <div style="width:40px; height:3px; background:{accent};
                      margin:.55rem auto 0; border-radius:2px;"></div>
          <p style="opacity:.72; margin-top:.65rem; font-size:.95rem;">
            Who's using the app today?
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, mid1, mid2, right = st.columns([1, 1.15, 1.15, 1])
    with mid1:
        with st.container(border=True):
            st.markdown(f"### {svg_icon('lock', size=22)}&nbsp; Official / Staff", unsafe_allow_html=True)
            st.caption(
                "MP office, district officer, or central admin — sign in to see the scored "
                "dashboards, forensics tools, and audit trail."
            )
            if st.button("Sign in", width='stretch', key="_gate_login_btn", icon=":material/login:"):
                st.session_state.show_login_form = True
                st.rerun()
    with mid2:
        with st.container(border=True):
            st.markdown(f"### {svg_icon('chat', size=22)}&nbsp; Public / Citizen", unsafe_allow_html=True)
            st.caption(
                "No account needed — tell the assistant about a problem with a project near you, "
                "or one that isn't even listed yet."
            )
            if st.button("Continue as public user", width='stretch', key="_gate_public_btn", icon=":material/forum:"):
                st.switch_page("pages/9_Public_Citizen_Chatbot.py")


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
          <div class="brand-title">{svg_icon('shield', size=18)}&nbsp; MPLADS Sentinel</div>
          <div class="brand-sub">{t('brand_tagline')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def theme_toggle_ui():
    """Renders the sidebar brand block, inside an explicit
    st.sidebar.container() — this becomes the FIRST of exactly two
    sibling containers under the sidebar's "user content" area (the
    second, added by show_user_badge() right after this returns, holds
    the profile card + log-out button). That's not cosmetic: the CSS in
    _CSS_TEMPLATE targets these two containers by position (first-of-type
    / last-of-type) so the profile+logout container always stays fully
    visible at the bottom regardless of how long the page-nav list gets
    — see the "Sidebar scroll-split" comment in the CSS for the full
    mechanism.

    The dark-mode toggle and language switcher used to live here too,
    but moved to header_controls() in the main content's top-right —
    controls people reach for constantly shouldn't require opening or
    scrolling the sidebar to use. Safe to call once per page render
    (require_login() calls it pre-login, show_user_badge() calls it
    post-login — never both in the same script run)."""
    with st.sidebar.container():
        sidebar_brand()


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
        if st.button(t("log_out"), width='stretch', icon=":material/logout:"):
            st.session_state.user = None
            st.rerun()


def _compute_notifications(user: dict) -> list:
    """Builds a short, role-scoped list of things worth a person's
    attention, from data already live in the app — not placeholder
    content. Kept intentionally small (a handful of items) since this
    is a notification feed, not another data table."""
    notifs = []
    try:
        df = get_scored_dataset()
        if user and user.get("role") != "central_admin":
            df = scoped_query(df, user)

        critical = df[df["risk_band"] == "CRITICAL"].sort_values("composite_score", ascending=False)
        for _, r in critical.head(3).iterrows():
            notifs.append({
                "icon": ":material/error:",
                "text": f"{r['project_id']} ({r['category']}) flagged CRITICAL — score {r['composite_score']:.0f}",
            })

        if user and user.get("role") in ("district_officer", "central_admin"):
            pending = df[df["status"] == "Recommended"]
            for _, r in pending.head(3).iterrows():
                notifs.append({"icon": ":material/explore:", "text": f"{r['project_id']} ({r['category']}) awaiting sanction"})

        raw = get_raw_tables()
        reports = raw.get("citizen_reports")
        if reports is not None and len(reports):
            in_scope = reports[reports["project_id"].isin(df["project_id"])]
            open_reports = in_scope[in_scope["status"] == "Open"] if "status" in in_scope.columns else in_scope
            for _, r in open_reports.head(3).iterrows():
                notifs.append({"icon": ":material/forum:", "text": f"New citizen report on {r['project_id']}: {r['complaint_type']}"})
    except Exception:
        # Notifications are a convenience, never a reason to break a page —
        # if scoring/data isn't ready yet (e.g. first-ever run), just show
        # an empty bell rather than crashing the page underneath it.
        return []
    return notifs[:8]


def header_controls():
    """Top-right row (notifications, language, dark mode) shown above
    every page's masthead via page_header(), and above the login card
    pre-login — kept in the main content area rather than the sidebar,
    since these are controls people reach for constantly and shouldn't
    need to open or scroll the sidebar to use."""
    user = st.session_state.get("user")
    st.markdown('<div class="sentinel-header-row-marker"></div>', unsafe_allow_html=True)
    cols = st.columns([5, 1.5, 2.4, 1.4] if user else [6, 2.4, 1.4])
    idx = 1
    if user:
        with cols[idx]:
            notifs = _compute_notifications(user)
            label = f":material/notifications: {len(notifs)}" if notifs else ":material/notifications:"
            with st.popover(label, width='stretch'):
                st.markdown(f"**{t('notifications')}**")
                if not notifs:
                    st.caption(t("no_notifications"))
                else:
                    for n in notifs:
                        st.markdown(f"{n['icon']} {n['text']}")
        idx += 1
    with cols[idx]:
        language_switcher(compact=True)
    idx += 1
    with cols[idx]:
        current = get_theme()
        is_dark = st.toggle(
            ":material/dark_mode:", value=(current == "dark"), key="_theme_toggle_switch", help=t("dark_mode"),
        )
        new_theme = "dark" if is_dark else "light"
        if new_theme != current:
            st.session_state.theme = new_theme
            st.rerun()


def page_header(icon: str, title: str, subtitle: str = None):
    """Renders the shared header controls row, then the page masthead:
    icon + title, a single accent rule, and an optional subtitle — used
    in place of a bare st.title()/st.caption() pair so every page in the
    app shares one consistent identity instead of nine independent
    headings.

    `icon` is a key into ICON_SVGS (e.g. "shield", "search", "camera") —
    this renders as raw HTML, so it goes through svg_icon() rather than
    Streamlit's ":material/name:" text shorthand (see the ICON_SVGS
    comment up top for why)."""
    header_controls()
    sub_html = f'<div class="masthead-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="sentinel-masthead">
          <div class="kicker">
            <span class="kicker-icon">{svg_icon(icon, size=26)}</span>
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
        "bg_gradient": "linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 100%)",
        "bg": "#F8FAFC",
        "card": "#FFFFFF",
        "card_border": "#E2E8F0",
        "sidebar_bg": "#FFFFFF",
        "sidebar_border": "#E2E8F0",
        "text": "#0F172A",
        "text_muted": "#64748B",
        "heading": "#0F172A",
        "primary": "#0F9F7B",
        "primary_dark": "#0C7C60",
        "accent": "#10B981",
        "shadow": "0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.02)",
        "shadow_hover": "0 10px 25px -3px rgba(0,0,0,0.08), 0 4px 6px -2px rgba(0,0,0,0.04)",
        "hover_bg": "#F1F5F9",
        "input_bg": "#FFFFFF",
        "divider": "#E2E8F0",
        "code_bg": "#F1F5F9",
    },
    "dark": {
        "bg_gradient": "linear-gradient(160deg, #060A13 0%, #0A101C 55%, #0B1220 100%)",
        "bg": "#070B14",
        "card": "#0F1729",
        "card_border": "#1D2840",
        "sidebar_bg": "#080C16",
        "sidebar_border": "#1A2338",
        "text": "#E5EAF3",
        "text_muted": "#8B96AC",
        "heading": "#F8FAFC",
        "primary": "#10B981",
        "primary_dark": "#059669",
        "accent": "#34D399",
        "shadow": "0 4px 20px rgba(0,0,0,0.45)",
        "shadow_hover": "0 14px 34px rgba(0,0,0,0.6), 0 0 0 1px rgba(16,185,129,0.12)",
        "hover_bg": "#141E33",
        "input_bg": "#0C1320",
        "divider": "#1D2840",
        "code_bg": "#0A0F1C",
    },
}


def get_theme() -> str:
    return st.session_state.get("theme", "dark")


def get_palette() -> dict:
    return THEMES[get_theme()]


_CSS_TEMPLATE = Template(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

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
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span,
[data-testid="stWidgetLabel"] label {
    color: ${text} !important;
    font-weight: 600 !important;
    font-size: .86rem !important;
}

/* ---- Layout / responsiveness ---- */
.main .block-container {
    padding-top: 1.8rem;
    padding-bottom: 3.5rem;
    max-width: 1440px;
    animation: sentinelFadeIn .35s ease;
}
@keyframes sentinelFadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
@media (max-width: 768px) {
    .main .block-container { padding-left: 1rem; padding-right: 1rem; padding-top: 1rem; }
    h1 { font-size: 1.45rem !important; }
    h2 { font-size: 1.15rem !important; }
    [data-testid="stMetricValue"] { font-size: 1.1rem !important; }
}
@media (max-width: 1180px) {
    [data-testid="stMetric"] { padding: 12px 14px 10px !important; }
    [data-testid="stMetricLabel"] { font-size: .68rem !important; letter-spacing: .04em; }
}
/* Streamlit's own column-stacking breakpoint is narrower than most tablets
   (verified: a real device/viewport at 700-900px keeps a 5-column KPI row
   and 2-column chart row side by side rather than stacking them) — at that
   width each column gets squeezed hard enough that metric VALUES visibly
   truncate (e.g. "395" rendering as just "5") and multiselect/selectbox
   filters clip off the edge of the screen. Forcing stHorizontalBlock to
   wrap in this range makes columns reflow onto as many rows as they need,
   the same safety net React Grid frameworks give you by default, which
   Streamlit's fixed-ratio columns don't. Below 640px Streamlit already
   stacks natively, so this covers the 640-1200px gap.  */
@media (max-width: 1200px) {
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
        row-gap: 12px;
    }
    [data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
        min-width: 200px !important;
        flex: 1 1 200px !important;
        width: auto !important;
    }
    /* The header controls row (notifications / language / theme) uses
       narrow fixed-ratio columns that go negative-width before a generic
       200px floor would even kick in — give it its own, smaller floor so
       it wraps into a tidy second line instead of clipping off-screen.
       header_controls() drops an empty .sentinel-header-row-marker div
       immediately before its st.columns() call so this can find exactly
       that row (verified against the actual rendered DOM: the marker's
       own .stElementContainer is followed by a sibling
       [data-testid="stLayoutWrapper"] that wraps the columns). */
    .stElementContainer:has(.sentinel-header-row-marker) + [data-testid="stLayoutWrapper"] [data-testid="stColumn"] {
        min-width: 120px !important;
    }
}
@media (min-width: 1800px) {
    .main .block-container { max-width: 1650px; }
}

/* ---- Headings ---- */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
    color: ${heading};
}
h1 {
    font-size: clamp(1.6rem, 1.2rem + 1.3vw, 2.3rem) !important;
    font-weight: 800 !important;
}
.sentinel-hero-title {
    color: ${heading} !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 800 !important;
    letter-spacing: -0.025em;
}
.sentinel-masthead {
    border-bottom: 1px solid ${divider};
    padding-bottom: 1rem;
    margin-bottom: 1.6rem;
}
.sentinel-masthead .kicker {
    display: flex; align-items: center; gap: 12px;
}
.sentinel-masthead .kicker-icon {
    display: inline-flex; align-items: center; justify-content: center;
    color: ${primary};
    background: linear-gradient(135deg, ${primary}22, ${accent}11);
    border: 1px solid ${primary}40;
    padding: 9px 12px;
    border-radius: 12px;
    box-shadow: ${shadow};
}
.sentinel-masthead h1 {
    margin: 0 !important;
}
.sentinel-masthead .masthead-sub {
    color: ${text_muted}; font-size: .92rem; margin-top: .4rem; max-width: 78ch; line-height: 1.5;
}
.sentinel-masthead .masthead-rule {
    width: 48px; height: 3.5px;
    background: linear-gradient(90deg, ${primary}, ${accent});
    margin-top: .8rem; border-radius: 999px;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: ${sidebar_bg} !important;
    border-right: 1px solid ${sidebar_border};
}
[data-testid="stSidebar"] > div { color: ${text}; }
[data-testid="stSidebarNavLink"] p, [data-testid="stSidebarNavLink"] span {
    color: ${text} !important;
    font-weight: 500;
}
[data-testid="stSidebarNavLink"] {
    border-radius: 10px !important;
    margin-bottom: 3px;
    padding-top: 8px !important;
    padding-bottom: 8px !important;
    position: relative;
    transition: background .15s ease;
}
[data-testid="stSidebarNavLink"]:hover { background: ${hover_bg} !important; }
[data-testid="stSidebarNavLink"][aria-current="page"] {
    background: linear-gradient(90deg, ${primary}26, ${primary}08) !important;
    border: 1px solid ${primary}35;
}
[data-testid="stSidebarNavLink"][aria-current="page"] p {
    color: ${accent} !important; font-weight: 700 !important;
}
[data-testid="stSidebarNavLink"][aria-current="page"]::before {
    content: ""; position: absolute; left: -8px; top: 6px; bottom: 6px; width: 3.5px;
    background: ${primary}; border-radius: 999px;
    box-shadow: 0 0 8px ${primary}80;
}
.sentinel-brand {
    padding: 8px 6px 16px; margin-bottom: 12px;
    border-bottom: 1px solid ${sidebar_border};
}
.sentinel-brand .brand-title {
    font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800; font-size: 1.12rem; color: ${heading};
    letter-spacing: -0.02em; display: flex; align-items: center; gap: 6px;
}
.sentinel-brand .brand-title svg { color: ${primary}; }
.sentinel-brand .brand-sub { font-size: .74rem; color: ${text_muted}; margin-top: 2px; }
.sentinel-user-card {
    background: ${hover_bg}; border: 1px solid ${card_border}; border-radius: 14px;
    padding: 12px 14px; margin: 4px 0 12px; display: flex; gap: 12px; align-items: center;
    box-shadow: ${shadow};
}
.sentinel-user-avatar {
    width: 36px; height: 36px; border-radius: 50%; flex-shrink: 0;
    background: linear-gradient(135deg, ${primary}, ${accent});
    color: #fff; display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: .84rem;
    box-shadow: 0 2px 8px ${primary}40;
}
.sentinel-user-name { font-weight: 700; font-size: .88rem; color: ${heading}; }
.sentinel-user-role { font-size: .74rem; color: ${text_muted}; }

/* ---- Sidebar scroll-split ---- */
[data-testid="stSidebarContent"] {
    display: flex;
    flex-direction: column;
    overflow: hidden !important;
    height: 100vh;
}
[data-testid="stSidebarHeader"] { flex-shrink: 0; }
[data-testid="stSidebarNav"] {
    /* The nav list is now the flexible, scrollable one — it takes
       whatever space is left over after the profile card + logout
       button below claim theirs, rather than the other way around, so
       "flex-shrink: 0" + a % cap could squeeze the fixed section off
       the bottom of a short window with no way to reach it. */
    flex: 1 1 auto;
    min-height: 0;
    overflow-y: auto;
}
[data-testid="stSidebarUserContent"] {
    /* Fixed, not flexible: this holds the brand block, the profile
       card, and the Log out button, and it should always render at its
       full natural size — never shrink, never need its own scrollbar.
       (stSidebarNav above is what flexes/scrolls instead — see its
       comment.) */
    flex: 0 0 auto;
    display: flex;
    flex-direction: column;
    padding-bottom: .5rem;
}
[data-testid="stSidebarUserContent"] > div {
    flex: 0 0 auto; display: flex; flex-direction: column;
}
[data-testid="stSidebarUserContent"] > div > [data-testid="stVerticalBlock"] {
    flex: 0 0 auto; display: flex; flex-direction: column;
}
[data-testid="stSidebarUserContent"] > div > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:first-of-type {
    flex-shrink: 0;
}
[data-testid="stSidebarUserContent"] > div > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:last-of-type {
    flex-shrink: 0;
    padding-top: .5rem;
}

/* ---- Metric cards ---- */
[data-testid="stMetric"] {
    background: linear-gradient(160deg, ${card} 0%, ${card} 100%) !important;
    border: 1px solid ${card_border} !important;
    border-top: 3px solid ${primary} !important;
    border-radius: 16px !important;
    padding: 18px 20px 16px !important;
    box-shadow: ${shadow} !important;
    position: relative;
    transition: box-shadow .2s ease, transform .2s ease, border-color .2s ease;
}
[data-testid="stMetric"]:hover {
    box-shadow: ${shadow_hover} !important;
    transform: translateY(-2px);
    border-color: ${primary}55 !important;
    border-top-color: ${accent} !important;
}
[data-testid="stMetricLabel"] {
    color: ${text_muted} !important; font-size: .74rem !important;
    text-transform: uppercase; letter-spacing: .06em; font-weight: 700 !important;
    line-height: 1.3;
}
[data-testid="stMetricLabel"] div,
[data-testid="stMetricLabel"] p {
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important;
    word-break: normal !important; overflow-wrap: normal !important; hyphens: none !important;
}
[data-testid="stMetricValue"] {
    color: ${heading} !important; font-weight: 800 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-variant-numeric: tabular-nums;
    font-size: clamp(1.2rem, .95rem + 1vw, 1.85rem) !important;
    letter-spacing: -0.02em;
}
[data-testid="stMetricValue"] div,
[data-testid="stMetricValue"] p {
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important;
    overflow-wrap: anywhere !important; word-break: normal !important;
}

/* ---- Buttons ---- */
.stButton button, .stDownloadButton button, .stFormSubmitButton button {
    background: linear-gradient(135deg, ${primary}, ${primary_dark}) !important;
    color: #ffffff !important;
    border: 1px solid ${primary_dark} !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    padding: 0.52rem 1.25rem !important;
    box-shadow: 0 2px 6px ${primary}30;
    transition: all .18s ease;
}
.stButton button:hover, .stDownloadButton button:hover, .stFormSubmitButton button:hover {
    background: linear-gradient(135deg, ${accent}, ${primary}) !important;
    border-color: ${accent} !important;
    box-shadow: 0 4px 14px ${primary}45;
    transform: translateY(-1px);
}
.stButton button:active, .stFormSubmitButton button:active {
    background: ${primary_dark} !important;
    transform: translateY(0);
}

/* ---- Header controls (notifications, language, theme) ---- */
/* Streamlit nests the actual <button> two levels inside [data-testid="stPopover"]
   (div > div > button), so a direct-child ">" selector here never matched — the
   notifications bell rendered with zero theming (a black box in light mode,
   invisible "6" badge) because it fell back to Streamlit's native chrome. */
[data-testid="stPopoverButton"] {
    background: ${card} !important;
    color: ${text} !important;
    border: 1px solid ${card_border} !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    box-shadow: ${shadow} !important;
    transition: all .15s ease;
}
[data-testid="stPopoverButton"]:hover {
    background: ${hover_bg} !important;
    border-color: ${primary} !important;
    color: ${primary} !important;
}
[data-testid="stPopoverButton"] p { color: inherit !important; }
[data-testid="stPopoverBody"] {
    background: ${card} !important;
    border: 1px solid ${card_border} !important;
    color: ${text} !important;
}
.stElementContainer:has(.sentinel-header-row-marker) + [data-testid="stLayoutWrapper"] [data-testid="stSelectbox"] > div {
    margin-top: 2px;
}
/* Theme toggle switch: give the track a visible border in BOTH themes — the
   unchecked track is a pale native gray that all but disappeared against a
   light background. */
label:has(input[role="switch"]) > div:first-of-type {
    border: 1px solid ${card_border} !important;
    box-shadow: none !important;
}
label:has(input[role="switch"]) [data-testid="stTooltipHoverTarget"] svg {
    fill: ${text_muted} !important;
}

/* ---- Inputs ---- */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stNumberInput"] input {
    background: ${input_bg} !important;
    color: ${text} !important;
    border: 1.5px solid ${card_border} !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    transition: border-color .15s ease, box-shadow .15s ease;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus,
[data-testid="stNumberInput"] input:focus {
    border-color: ${primary} !important;
    box-shadow: 0 0 0 3px ${primary}25 !important;
}
[data-testid="InputInstructions"] { display: none !important; }
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div,
[data-testid="stSelectbox"] [role="group"],
[data-testid="stMultiSelect"] [role="group"] {
    background: ${input_bg} !important;
    border-color: ${card_border} !important;
    border: 1.5px solid ${card_border} !important;
    border-radius: 10px !important;
    color: ${text} !important;
}
[data-testid="stSelectbox"] input,
[data-testid="stMultiSelect"] input {
    background: transparent !important;
    color: ${text} !important;
}
html body [data-testid="stMultiSelectTagsContainer"] {
    flex-wrap: wrap !important;
    overflow: visible !important;
    height: auto !important;
    max-height: none !important;
    min-height: 38px;
    row-gap: 4px;
    padding-top: 4px; padding-bottom: 4px;
}
[data-testid="stSelectbox"] svg, [data-testid="stMultiSelect"] svg {
    fill: ${text_muted} !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: ${hover_bg} !important;
    border: 1.5px dashed ${card_border} !important;
    border-radius: 14px !important;
    transition: border-color .18s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: ${primary} !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background: ${card} !important;
    color: ${text} !important;
    border: 1.5px solid ${card_border} !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    border-color: ${primary} !important;
    color: ${primary} !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span {
    color: ${text_muted} !important;
}

/* ---- Expanders ---- */
[data-testid="stExpander"] {
    border: 1px solid ${card_border} !important;
    border-radius: 12px !important;
    background: ${card} !important;
    box-shadow: ${shadow};
    overflow: hidden;
    margin-bottom: 10px;
    transition: border-color .18s ease;
}
[data-testid="stExpander"]:hover {
    border-color: ${primary}60 !important;
}
[data-testid="stExpander"] summary {
    font-weight: 600;
    color: ${heading} !important;
    padding: 12px 16px !important;
}
[data-testid="stExpander"] summary:hover { background: ${hover_bg}; }

/* ---- Tabs ---- */
[data-testid="stTab"] {
    color: ${text_muted} !important;
    font-weight: 600;
    font-family: 'Plus Jakarta Sans', sans-serif;
    padding-top: 8px; padding-bottom: 10px;
}
[data-testid="stTab"] p { color: inherit !important; }
[data-testid="stTab"][aria-selected="true"],
[data-testid="stTab"][data-selected="true"] {
    color: ${primary} !important;
    font-weight: 700;
}
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1.5px solid ${divider};
    gap: 8px;
}
[data-testid="stTab"] .react-aria-SelectionIndicator {
    background: ${primary} !important;
    height: 3px !important;
    border-radius: 3px 3px 0 0;
}

/* ---- Alerts / expanders / dataframe / plotly / images as "cards" ---- */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border: 1px solid ${card_border} !important;
}
/* Streamlit's default alert colors (especially "warning") don't track our
   palette and read as a muddy, low-contrast olive box in dark mode. Give
   each kind the same soft-tint treatment as the risk pills instead. */
[data-testid="stAlertContainer"] { border-radius: 12px !important; }
[data-testid="stAlertContentInfo"] {
    background: rgba(59,130,246,0.12) !important;
    border: 1px solid rgba(59,130,246,0.35) !important;
    border-radius: 12px !important;
}
[data-testid="stAlertContentInfo"] p { color: #3B82F6 !important; }
[data-testid="stAlertContentSuccess"] {
    background: rgba(16,185,129,0.12) !important;
    border: 1px solid rgba(16,185,129,0.35) !important;
    border-radius: 12px !important;
}
[data-testid="stAlertContentSuccess"] p { color: #10B981 !important; }
[data-testid="stAlertContentWarning"] {
    background: rgba(245,158,11,0.14) !important;
    border: 1px solid rgba(245,158,11,0.4) !important;
    border-radius: 12px !important;
}
[data-testid="stAlertContentWarning"] p { color: #F59E0B !important; }
[data-testid="stAlertContentError"] {
    background: rgba(239,68,68,0.12) !important;
    border: 1px solid rgba(239,68,68,0.35) !important;
    border-radius: 12px !important;
}
[data-testid="stAlertContentError"] p { color: #EF4444 !important; }
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {
    border: 1px solid ${card_border};
    border-radius: 14px;
    padding: 10px;
    background: ${card};
    box-shadow: ${shadow};
    transition: box-shadow .2s ease;
}
[data-testid="stPlotlyChart"]:hover, [data-testid="stDataFrame"]:hover {
    box-shadow: ${shadow_hover};
}
[data-testid="stImage"] img { border-radius: 10px; box-shadow: ${shadow}; }

/* ---- Divider / code ---- */
hr { border-color: ${divider} !important; margin: 1.4rem 0 !important; }
[data-testid="stCode"] pre, [data-testid="stCode"] code {
    background: ${code_bg} !important;
    color: ${text} !important;
    border-radius: 12px !important;
    border: 1px solid ${card_border};
}
code { background: ${code_bg} !important; border-radius: 4px !important; }

/* ---- Risk pills ---- */
.risk-pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 999px;
    font-weight: 700; font-size: .74rem;
    letter-spacing: .04em; text-transform: uppercase;
    white-space: nowrap;
}
.risk-pill.critical {
    animation: sentinelPulse 2s ease-in-out infinite;
}
@keyframes sentinelPulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,.35); }
    50% { box-shadow: 0 0 0 6px rgba(239,68,68,0); }
}

/* ---- Raw HTML tables (to_html output) ---- */
.sentinel-table-wrap {
    overflow-x: auto; -webkit-overflow-scrolling: touch;
    border: 1px solid ${card_border}; border-radius: 14px;
    box-shadow: ${shadow}; background: ${card}; margin: .8rem 0 1.4rem;
}
.sentinel-table-wrap table { width: 100%; border-collapse: collapse; font-size: .88rem; }
.sentinel-table-wrap thead th {
    position: sticky; top: 0; background: ${hover_bg}; color: ${heading};
    text-align: left; padding: 12px 16px; font-weight: 700;
    font-family: 'Plus Jakarta Sans', sans-serif;
    border-bottom: 2px solid ${card_border}; white-space: nowrap;
    letter-spacing: .02em; font-size: .80rem; text-transform: uppercase;
}
.sentinel-table-wrap tbody td {
    padding: 11px 16px; border-bottom: 1px solid ${card_border};
    color: ${text}; white-space: nowrap;
}
.sentinel-table-wrap tbody tr:hover { background: ${hover_bg}; }
@media (max-width: 640px) { .sentinel-table-wrap table { font-size: .76rem; } }

/* ---- Auth screen: sign-in form card + demo-account credential cards ---- */
[data-testid="stForm"] {
    background: ${card} !important;
    border: 1px solid ${card_border} !important;
    border-radius: 14px !important;
    padding: 1.5rem 1.6rem 0.8rem !important;
    box-shadow: ${shadow_hover};
}
.sentinel-cred-note {
    font-size: .84rem; color: ${text_muted}; line-height: 1.5;
    margin: .2rem 0 1.1rem;
}
.sentinel-cred-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 12px;
}
.sentinel-cred-card {
    border: 1px solid ${card_border}; border-radius: 12px;
    background: ${card}; padding: 14px 16px; box-shadow: ${shadow};
    transition: transform .18s ease;
}
.sentinel-cred-card:hover { transform: translateY(-2px); }
.sentinel-cred-card .role-row {
    display: flex; align-items: center; gap: 8px;
    font-weight: 700; color: ${heading}; font-size: .95rem;
    font-family: 'Plus Jakarta Sans', sans-serif;
}
.sentinel-cred-card .role-icon { display: inline-flex; color: ${primary}; }
.sentinel-cred-card .scope {
    font-size: .76rem; color: ${text_muted}; margin: 2px 0 10px;
}
.sentinel-cred-card .cred-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 7px 0; border-top: 1px dashed ${divider};
}
.sentinel-cred-card .cred-row:first-of-type { border-top: 1px solid ${divider}; }
.sentinel-cred-label {
    color: ${text_muted}; font-size: .72rem; text-transform: uppercase; letter-spacing: .04em; font-weight: 600;
}
.sentinel-cred-value {
    font-family: 'SF Mono', 'Consolas', 'Menlo', monospace; font-size: .82rem;
    color: ${text}; background: ${hover_bg}; padding: 2px 8px; border-radius: 6px;
    border: 1px solid ${card_border};
}

/* ---- Scrollbar ---- */
::-webkit-scrollbar { width: 8px; height: 8px; }
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
        margin=dict(l=20, r=20, t=30, b=20),
    )
    fig.update_xaxes(
        gridcolor=pal["divider"], zerolinecolor=pal["divider"],
        color=pal["text_muted"], tickfont=dict(size=11),
    )
    fig.update_yaxes(
        gridcolor=pal["divider"], zerolinecolor=pal["divider"],
        color=pal["text_muted"], tickfont=dict(size=11),
    )
    if height:
        fig.update_layout(height=height)
    return fig


def render_html_table(html: str, max_height: int | None = None):
    """Wraps a df.to_html(...) string in a styled, horizontally-scrollable
    card so raw HTML tables match the rest of the theme on any screen size.

    Pass max_height (px) for long tables that used to rely on st.dataframe's
    built-in scroll window — the header stays pinned via the existing
    `position: sticky` rule on thead th, so scrolling behaves the same."""
    style = f' style="max-height:{max_height}px; overflow-y:auto;"' if max_height else ""
    st.markdown(f'<div class="sentinel-table-wrap"{style}>{html}</div>', unsafe_allow_html=True)


RISK_COLORS = {"LOW": "#10B981", "MEDIUM": "#3B82F6", "HIGH": "#F59E0B", "CRITICAL": "#EF4444"}


def _hex_to_rgb(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def risk_pill(band: str) -> str:
    """Soft, translucent status pill (tinted background + solid-colored
    text/border, with a small solid-color dot standing in for what used
    to be a colored emoji circle) matching the reference dashboard's
    badge style, rather than a solid block of color with white text."""
    color = RISK_COLORS.get(band, "#888888")
    extra_class = " critical" if band == "CRITICAL" else ""
    r, g, b = _hex_to_rgb(color)
    bg = f"rgba({r},{g},{b},0.16)"
    border = f"rgba({r},{g},{b},0.4)"
    dot = svg_icon("dot", size=9, color=color)
    return (
        f'<span class="risk-pill{extra_class}" '
        f'style="background:{bg}; color:{color}; border:1px solid {border}">{dot} {band}</span>'
    )
