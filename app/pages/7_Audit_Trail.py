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

from common import inject_base_style, require_login, require_role, show_user_badge
from sentinel.audit.hash_chain import read_chain, verify_chain

st.set_page_config(page_title="Audit Trail — MPLADS Sentinel", page_icon="🔗", layout="wide")
inject_base_style()
user = require_login()
show_user_badge()
require_role(user, ["district_officer", "central_admin"])
st.title("🔗 Tamper-Evident Audit Trail")
st.caption(
    "Every scoring run and every officer verdict is appended to a SHA-256 hash chain — each entry embeds "
    "the hash of the one before it. Editing or deleting a past entry breaks every hash after it, making "
    "silent tampering with the audit record detectable."
)

entries = read_chain()
is_valid, broken_at = verify_chain()

c1, c2 = st.columns(2)
c1.metric("Total Logged Events", len(entries))
with c2:
    if is_valid:
        st.success("✅ Chain integrity verified — no tampering detected.")
    else:
        st.error(f"⚠️ Chain integrity FAILED at entry #{broken_at}. The log may have been tampered with.")

st.divider()

if entries:
    view = pd.DataFrame(entries)
    view = view[["timestamp", "event_type", "project_id", "actor", "details", "entry_hash"]]
    view["entry_hash"] = view["entry_hash"].str[:16] + "…"
    view["details"] = view["details"].apply(lambda d: str(d))
    st.dataframe(view.sort_values("timestamp", ascending=False), width='stretch', hide_index=True)
else:
    st.info("No events logged yet — run the scoring pipeline or submit an officer review to generate audit entries.")

with st.expander("How the hash chain works"):
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
""")
