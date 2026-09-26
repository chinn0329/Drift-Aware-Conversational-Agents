"""IO helpers for reading/writing conversation transcripts and feature logs
under data/processed/ (Section 8.2)."""

import json
from pathlib import Path


def save_conversation_log(log: dict, out_dir: str) -> str:
    """Write one conversation's transcript + feature log as JSON."""
    out_path = Path(out_dir) / f"{log.get('conversation_id', 'unknown')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(log, f, indent=2)
    return str(out_path)


def load_conversation_logs(log_dir: str) -> list:
    """Load all conversation JSON logs from a directory."""
    logs = []
    for path in Path(log_dir).glob("*.json"):
        with open(path) as f:
            logs.append(json.load(f))
    return logs
