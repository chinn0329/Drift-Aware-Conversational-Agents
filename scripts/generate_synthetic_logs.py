"""Create explicitly synthetic, benchmark-shaped feature logs for prototyping.

The generated eventual_success field is a proxy label, not an observed outcome.
Do not report model scores on this data as real benchmark performance.
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = ROOT / "data/raw/lost_in_conversation/data/sharded_instructions_600.json"
DEFAULT_LOGS = ROOT / "data/processed/raw_logs.jsonl"
FIELDS = (
    "conversation_id",
    "turn_index",
    "self_consistency_variance",
    "contradiction_score",
    "confidence",
    "confidence_delta",
    "silent_commitment_ratio",
    "eventual_success",
    "source",
)
FEATURE_FIELDS = FIELDS[2:7]
SUCCESS_PRIORS = {
    "code": 0.52,
    "database": 0.43,
    "actions": 0.55,
    "math": 0.67,
    "data2text": 0.64,
    "summary": 0.61,
}
SYNTHETIC_SOURCE_PREFIX = "synthetic_augmentation/"


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def get_shards(task: dict) -> list[tuple[int, str, bool]]:
    shards = []
    for index, item in enumerate(task.get("shards", []), start=1):
        if isinstance(item, dict):
            text = item.get("shard") or item.get("shard_text") or ""
            explicitly_underspecified = item.get("is_underspecified")
        elif isinstance(item, str):
            text = item
            explicitly_underspecified = None
        else:
            continue

        if not text:
            continue
        if explicitly_underspecified is None:
            words = text.lower().split()
            explicitly_underspecified = len(words) <= 8 or any(
                token in {"it", "that", "them", "more", "same"}
                for token in words
            )
        shards.append((index, text, bool(explicitly_underspecified)))
    return shards


def load_real_records(path: Path) -> dict[tuple[str, int], dict]:
    records = {}
    if not path.exists():
        return records

    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                key = (row["conversation_id"], int(row["turn_index"]))
                if key not in records:
                    records[key] = {field: row.get(field) for field in FIELDS}
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid existing log at {path}:{line_number}") from exc
    return records


def build_donors(records: dict[tuple[str, int], dict]) -> list[tuple[int, list[dict]]]:
    conversations = defaultdict(list)
    for (_, turn_index), row in records.items():
        if all(isinstance(row.get(field), (int, float)) for field in FEATURE_FIELDS):
            conversations[row["conversation_id"]].append((turn_index, row))
    donors = []
    for rows in conversations.values():
        rows.sort(key=lambda pair: pair[0])
        donors.append((len(rows), [row for _, row in rows]))
    return donors


def sample_donor(
    donors: list[tuple[int, list[dict]]],
    target_turn_count: int,
    turn_position: int,
) -> dict:
    if not donors:
        return {
            "self_consistency_variance": 0.25,
            "contradiction_score": 0.25,
            "confidence": 0.75,
            "confidence_delta": 0.0,
            "silent_commitment_ratio": 0.0,
        }

    _, rows = min(donors, key=lambda item: abs(item[0] - target_turn_count))
    relative_position = turn_position / max(target_turn_count, 1)
    donor_index = min(
        range(len(rows)),
        key=lambda index: abs(
            (index + 1) / max(len(rows), 1) - relative_position
        ),
    )
    return rows[donor_index]


def task_difficulty(task: dict, turn_count: int) -> float:
    task_type = task.get("task", "general")
    explicit = str(
        task.get("difficulty", task.get("spider_difficulty", "medium"))
    ).lower()
    explicit_score = {
        "easy": 0.2,
        "medium": 0.5,
        "hard": 0.8,
        "very hard": 0.95,
    }.get(explicit, 0.5)
    length_score = clamp((turn_count - 2) / 10)
    text_fields = (
        task.get("question")
        or task.get("fully_specified_question")
        or task.get("prompt")
        or task.get("query")
        or ""
    )
    content_score = clamp(len(str(text_fields)) / 3000)
    complexity = 0.50 * explicit_score + 0.30 * length_score + 0.20 * content_score
    return clamp(complexity + (0.03 if task_type in {"code", "database"} else 0.0))


def synthetic_success(task: dict, difficulty: float, rng: random.Random) -> bool:
    task_type = task.get("task", "general")
    probability = SUCCESS_PRIORS.get(task_type, 0.55) - 0.12 * (difficulty - 0.5)
    probability = clamp(probability, 0.25, 0.82)
    return rng.random() < probability


def augment_to_target(
    records: dict[tuple[str, int], dict],
    target_rows: int,
    seed: int,
) -> None:
    if len(records) > target_rows:
        raise ValueError(
            f"Existing benchmark-shaped rows ({len(records)}) exceed target "
            f"({target_rows}); refusing to drop records."
        )
    if len(records) == target_rows:
        return

    by_conversation = defaultdict(list)
    for (conversation_id, turn_index), row in records.items():
        by_conversation[conversation_id].append((turn_index, row))
    donors = []
    source_counts = defaultdict(int)
    for conversation_id, rows in by_conversation.items():
        rows.sort(key=lambda item: item[0])
        if not rows:
            continue
        source = str(rows[0][1].get("source") or "general")
        if source.startswith(SYNTHETIC_SOURCE_PREFIX):
            continue
        donors.append((source, [row for _, row in rows]))
        source_counts[source] += 1
    if not donors:
        raise ValueError("No benchmark-derived feature conversations available to augment.")

    rng = random.Random(seed ^ 0x5A17C0DE)
    sources = sorted(source_counts)
    source_weights = [source_counts[source] for source in sources]
    lengths = [len(rows) for _, rows in donors]
    added_count = 0
    synthetic_index = 1

    while len(records) < target_rows:
        source = rng.choices(sources, weights=source_weights, k=1)[0]
        source_donors = [rows for donor_source, rows in donors if donor_source == source]
        donor_rows = rng.choice(source_donors)
        conversation_length = min(rng.choice(lengths), target_rows - len(records))
        if conversation_length < 1:
            break

        domain = source.split("_")[0]
        if domain == "humaneval" or source.startswith("lcb_"):
            domain = "code"
        elif domain not in SUCCESS_PRIORS:
            domain = "general"

        donor_label = bool(donor_rows[0].get("eventual_success"))
        prior = SUCCESS_PRIORS.get(domain, 0.55)
        p_success = min(0.82, max(0.25, 0.65 * float(donor_label) + 0.35 * prior))
        label = rng.random() < p_success
        conversation_id = f"synthetic-augmentation-{synthetic_index:05d}"
        synthetic_index += 1

        previous_confidence = clamp(
            float(donor_rows[0]["confidence"]) + rng.gauss(0.0, 0.07), 0.05, 0.99
        )
        for turn_index in range(1, conversation_length + 1):
            donor = donor_rows[min(turn_index - 1, len(donor_rows) - 1)]
            variance = clamp(
                float(donor["self_consistency_variance"])
                + rng.gauss(0.0, 0.07)
            )
            contradiction = clamp(
                float(donor["contradiction_score"])
                + rng.gauss(0.0, 0.05)
            )
            confidence = clamp(
                0.65 * previous_confidence
                + 0.35 * float(donor["confidence"])
                + rng.gauss(0.025 if label else -0.025, 0.055),
                0.05,
                0.99,
            )
            confidence_delta = confidence - previous_confidence
            previous_confidence = confidence
            commitment_ratio = clamp(
                float(donor["silent_commitment_ratio"])
                + rng.gauss(-0.025 if label else 0.025, 0.06)
            )
            row = {
                "conversation_id": conversation_id,
                "turn_index": turn_index,
                "self_consistency_variance": variance,
                "contradiction_score": contradiction,
                "confidence": confidence,
                "confidence_delta": confidence_delta,
                "silent_commitment_ratio": commitment_ratio,
                "eventual_success": label,
                "source": f"{SYNTHETIC_SOURCE_PREFIX}{source}",
            }
            records[(conversation_id, turn_index)] = row
            added_count += 1

    print(
        f"Added {added_count} explicitly synthetic turn rows in "
        f"{synthetic_index - 1} synthetic conversations."
    )


def make_synthetic_row(
    task: dict,
    turn_index: int,
    turn_position: int,
    underspecified: bool,
    donor: dict,
    difficulty: float,
    previous_confidence: float,
    underspecified_count: int,
    silent_commitments: int,
    rng: random.Random,
) -> tuple[dict, float, int, int]:
    variance = clamp(
        float(donor["self_consistency_variance"])
        + rng.gauss(0.015 * difficulty, 0.035)
    )
    contradiction = clamp(
        float(donor["contradiction_score"])
        + rng.gauss(0.02 * difficulty, 0.025)
    )

    donor_confidence = clamp(float(donor["confidence"]), 0.05, 0.99)
    confidence = clamp(
        0.62 * previous_confidence
        + 0.38 * donor_confidence
        + rng.gauss(-0.012 * difficulty, 0.045),
        0.05,
        0.99,
    )
    confidence_delta = confidence - previous_confidence

    if underspecified:
        underspecified_count += 1
        clarification_probability = clamp(0.50 - 0.30 * difficulty, 0.12, 0.50)
        if rng.random() > clarification_probability:
            silent_commitments += 1
    commitment_ratio = (
        silent_commitments / underspecified_count if underspecified_count else 0.0
    )

    source = task.get("source", task.get("task", "general"))
    row = {
        "conversation_id": task.get("task_id", "conv_unk"),
        "turn_index": turn_index,
        "self_consistency_variance": variance,
        "contradiction_score": contradiction,
        "confidence": confidence,
        "confidence_delta": confidence_delta,
        "silent_commitment_ratio": commitment_ratio,
        "eventual_success": None,
        "source": source,
    }
    return row, confidence, underspecified_count, silent_commitments


def generate(
    tasks_path: Path,
    logs_path: Path,
    seed: int,
    replace: bool,
    target_rows: int | None = None,
) -> int:
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise ValueError(f"Expected task list in {tasks_path}")

    real_records = load_real_records(logs_path)
    donors = build_donors(real_records)
    output_records = {}
    for task in tasks:
        conversation_id = task.get("task_id", "conv_unk")
        shards = get_shards(task)
        if not shards:
            continue

        task_seed = int.from_bytes(
            hashlib.sha256(conversation_id.encode("utf-8")).digest()[:8], "big"
        )
        task_rng = random.Random(seed ^ task_seed)
        difficulty = task_difficulty(task, len(shards))
        label = synthetic_success(task, difficulty, task_rng)
        previous_confidence = 1.0
        underspecified_count = 0
        silent_commitments = 0

        for position, (turn_index, _, underspecified) in enumerate(shards, start=1):
            key = (conversation_id, turn_index)
            existing = real_records.get(key)
            if existing and all(existing.get(field) is not None for field in FEATURE_FIELDS):
                row = dict(existing)
                previous_confidence = float(row["confidence"])
                if underspecified:
                    underspecified_count += 1
                    silent_commitments = round(
                        float(row["silent_commitment_ratio"] or 0.0)
                        * underspecified_count
                    )
            else:
                donor = sample_donor(donors, len(shards), position)
                row, previous_confidence, underspecified_count, silent_commitments = (
                    make_synthetic_row(
                        task,
                        turn_index,
                        position,
                        underspecified,
                        donor,
                        difficulty,
                        previous_confidence,
                        underspecified_count,
                        silent_commitments,
                        task_rng,
                    )
                )
            row["eventual_success"] = label
            output_records[key] = {field: row.get(field) for field in FIELDS}

    if target_rows is not None:
        augment_to_target(output_records, target_rows, seed)

    ordered = sorted(
        output_records.values(),
        key=lambda row: (row["conversation_id"], row["turn_index"]),
    )
    if not ordered:
        raise RuntimeError("No dataset rows were generated.")

    nulls = [
        row["conversation_id"]
        for row in ordered
        if any(row.get(field) is None for field in FIELDS)
    ]
    if nulls:
        raise ValueError(f"Generated rows have missing fields: {nulls[:5]}")

    logs_path.parent.mkdir(parents=True, exist_ok=True)
    if logs_path.exists():
        if not replace:
            raise FileExistsError(
                f"{logs_path} exists; pass --replace to create a backup and replace it."
            )
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        backup = logs_path.with_name(
            f"{logs_path.stem}.backup-{timestamp}{logs_path.suffix}"
        )
        shutil.copy2(logs_path, backup)
        print(f"Backed up current logs to {backup}")

    temp_path = logs_path.with_name(logs_path.name + ".tmp")
    with temp_path.open("w", encoding="utf-8") as output:
        for row in ordered:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temp_path, logs_path)

    success_count = sum(bool(row["eventual_success"]) for row in ordered)
    print(
        f"Wrote {len(ordered)} rows for {len({r['conversation_id'] for r in ordered})} "
        f"tasks to {logs_path}"
    )
    print(
        f"Synthetic proxy labels: {success_count} success "
        f"({success_count / len(ordered):.1%} of turn rows)."
    )
    print(
        "Warning: labels for this generated dataset are synthetic task-level proxies, "
        "not measured task outcomes."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a benchmark-shaped synthetic feature JSONL for prototyping."
    )
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_LOGS)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument(
        "--target-rows",
        type=int,
        help="Expand to this many rows with explicitly marked synthetic augmentation.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Back up an existing output file before replacing it.",
    )
    args = parser.parse_args()
    return generate(
        args.tasks,
        args.output,
        args.seed,
        args.replace,
        target_rows=args.target_rows,
    )


if __name__ == "__main__":
    raise SystemExit(main())
