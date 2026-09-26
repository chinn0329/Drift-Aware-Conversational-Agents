"""
Section 8.2 / 11.1 — Offline simulation loop.

1. Clone the "Lost in Conversation" framework and set API credentials.
2. Reproduce the paper's baseline once, unmodified, as a sanity check.
3. Instrument the per-turn loop: after every assistant turn, compute and
   log the four drift features before the next shard is revealed.
4. Run across all sharded instructions; each run produces a full
   conversation transcript plus a per-turn feature log.
5. At conversation end, record the framework's own pass/fail verdict.
"""


def run_single_conversation(sharded_instruction, backbone, log_dir: str) -> dict:
    """Simulate one full multi-turn conversation, logging per-turn features
    and the final framework pass/fail verdict.

    Returns:
        dict with keys: conversation_id, turns (list of feature dicts), verdict
    """
    raise NotImplementedError


def run_batch(sharded_instructions: list, backbone, log_dir: str, n_runs: int = 1):
    """Run run_single_conversation across all instructions, n_runs times each
    (Section 8.3: repeated runs per task)."""
    raise NotImplementedError
