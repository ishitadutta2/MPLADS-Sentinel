"""
Audit Trail — shows the tamper-evident SHA-256 hash-chain log of every
scoring run and officer action, and lets a user verify chain integrity
live in the UI.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from common import inject_base_style, require_login, require_role, show_user_badge, page_header, render_html_table
from sentinel.audit.hash_chain import read_chain, verify_chain
from i18n import t

st.set_page_config(page_title="Audit Trail — MPLADS Sentinel", page_icon=":material/link:", layout="wide")
inject_base_style()
user = require_login()
show_user_badge()
require_role(user, ["district_officer", "central_admin"])
page_header("link", t("audit_title"), t("audit_sub"))

entries = read_chain()
is_valid, broken_at = verify_chain()

c1, c2 = st.columns(2)
c1.metric(t("kpi_total_logged_events"), len(entries))
with c2:
    if is_valid:
        st.success(t("chain_verified_success"))
    else:
        st.error(t("chain_failed_error", n=broken_at))

st.divider()

if entries:
    view = pd.DataFrame(entries)
    view = view[["timestamp", "event_type", "project_id", "actor", "details", "entry_hash"]]
    view["entry_hash"] = view["entry_hash"].str[:16] + "…"
    view["details"] = view["details"].apply(lambda d: str(d))
    view.columns = [t("col_timestamp"), t("col_event_type"), t("col_project_id"), t("col_actor"),
                     t("col_details"), t("col_entry_hash")]
    view = view.sort_values(t("col_timestamp"), ascending=False)
    # escape=True (default) here, unlike other tables in this app — the
    # "Details" column can contain raw citizen-typed text (via the Citizen
    # Chatbot), so it must be HTML-escaped rather than trusted like the
    # risk_pill() markup other pages render into their tables.
    render_html_table(view.to_html(index=False), max_height=520)
else:
    st.info(t("no_audit_events_yet"))

with st.expander(t("how_hash_chain_works")):
    st.markdown("""
Each entry is a JSON record:
```
{ timestamp, event_type, project_id, actor, details, prev_hash, entry_hash }
```
`entry_hash = SHA256(entry without entry_hash)`, and `prev_hash` is the previous entry's `entry_hash`.
The first entry's `prev_hash` is `000...0` (genesis).

To verify: walk the chain from the start, recompute each entry's hash, and check it matches both the
stored `entry_hash` **and** the next entry's `prev_hash`. If even one character in one historical entry
changes, every hash after it stops matching — the tampering is immediately visible without needing a
central authority or blockchain network, just this log file and this verification function.

*(This technical explainer is currently only available in English.)*
""")
