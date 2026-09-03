"""
Tamper-evident audit trail.

Every significant action (a project gets scored, an officer records a
verdict, a risk score is recalculated) is appended as a JSON line to
`data/audit_chain.jsonl`, where each entry embeds the SHA-256 hash of the
*previous* entry — a simple blockchain-style hash chain. This means:

  - the log is append-only in practice: editing or deleting a past entry
    changes its hash, which breaks every subsequent link
  - `verify_chain()` can prove, cheaply and offline, whether the log has
    been tampered with since it was written — exactly the property an
    anti-corruption audit trail needs, without needing an actual
    blockchain/consensus network.
"""
import hashlib
import json
import datetime
from pathlib import Path

from sentinel.config import AUDIT_LOG_PATH

GENESIS_HASH = "0" * 64


def _hash_entry(entry: dict) -> str:
    payload = json.dumps(entry, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _last_hash() -> str:
    if not AUDIT_LOG_PATH.exists():
        return GENESIS_HASH
    last_line = None
    with open(AUDIT_LOG_PATH, "r") as f:
        for line in f:
            if line.strip():
                last_line = line
    if last_line is None:
        return GENESIS_HASH
    return json.loads(last_line)["entry_hash"]


def append_event(event_type: str, project_id: str, actor: str, details: dict):
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    prev_hash = _last_hash()
    entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "event_type": event_type,
        "project_id": project_id,
        "actor": actor,
        "details": details,
        "prev_hash": prev_hash,
    }
    entry["entry_hash"] = _hash_entry(entry)
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    return entry


def read_chain():
    if not AUDIT_LOG_PATH.exists():
        return []
    entries = []
    with open(AUDIT_LOG_PATH, "r") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def verify_chain():
    """Returns (is_valid: bool, first_broken_index_or_None)."""
    entries = read_chain()
    prev_hash = GENESIS_HASH
    for i, entry in enumerate(entries):
        expected_prev = entry["prev_hash"]
        if expected_prev != prev_hash:
            return False, i
        claimed_hash = entry["entry_hash"]
        recomputed = _hash_entry({k: v for k, v in entry.items() if k != "entry_hash"})
        if claimed_hash != recomputed:
            return False, i
        prev_hash = claimed_hash
    return True, None
