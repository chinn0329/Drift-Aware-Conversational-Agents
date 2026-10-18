"""Generate pipeline feature logs in resumable batches using the Groq API."""

import argparse
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEFAULT_INPUT = Path("data/raw/lost_in_conversation/data/sharded_instructions_600.json")
DEFAULT_OUTPUT_DIR = Path("data/processed/groq_labeled_batches")
DEFAULT_MERGED_OUTPUT = Path("data/processed/groq_raw_logs.jsonl")
DEFAULT_RAW_LOGS = Path("data/processed/raw_logs.jsonl")
MAX_RETRIES = 5
_worker_local = threading.local()


def load_api_keys() -> list[tuple[str, str]]:
    numbered_names = sorted(
        (
            name
            for name in os.environ
            if re.fullmatch(r"GROQ_API_KEY_\d+", name)
        ),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    if numbered_names:
        keys = [
            (name, os.environ[name])
            for name in numbered_names
            if os.environ[name]
        ]
    else:
        single_key = os.environ.get("GROQ_API_KEY")
        keys = [("GROQ_API_KEY", single_key)] if single_key else []

    if not keys:
        raise RuntimeError(
            "No Groq API key found. Set GROQ_API_KEY or numbered variables "
            "such as GROQ_API_KEY_1 and GROQ_API_KEY_2 in your environment or .env."
        )
    return keys


class GroqBackend:
    def __init__(self, api_keys: list[tuple[str, str]], model_name: str):
        from openai import OpenAI

        self.clients = [
            OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
                timeout=120.0,
                max_retries=0,
            )
            for _, api_key in api_keys
        ]
        self.key_names = [name for name, _ in api_keys]
        self.model_name = model_name
        self._next_client = 0
        self._client_lock = threading.Lock()
        self._invalid_clients = set()

    def _get_client(self):
        with self._client_lock:
            for _ in range(len(self.clients)):
                index = self._next_client
                self._next_client = (self._next_client + 1) % len(self.clients)
                if index not in self._invalid_clients:
                    return index, self.clients[index]
        raise RuntimeError(
            "All configured Groq API keys have been rejected. Check the "
            "GROQ_API_KEY variable(s) loaded from .env or your shell."
        )

    def generate(self, messages: list, temperature: float = 0.7) -> str:
        for attempt in range(MAX_RETRIES):
            client_index, client = self._get_client()
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                )
                content = response.choices[0].message.content
                if content is None:
                    raise RuntimeError("Groq returned an empty response.")
                return content
            except Exception as exc:
                status_code = getattr(exc, "status_code", None)
                incorrect_key = (
                    status_code == 400
                    and "incorrect api key" in str(exc).lower()
                )
                if incorrect_key or status_code in (401, 403):
                    with self._client_lock:
                        self._invalid_clients.add(client_index)
                        remaining = len(self.clients) - len(self._invalid_clients)
                    logging.error(
                        "Groq rejected the key from %s; trying another configured "
                        "key if available.",
                        self.key_names[client_index],
                    )
                    if remaining:
                        continue
                    raise RuntimeError(
                        "Groq rejected every configured API key. Check that the "
                        "key values in .env are current Groq API keys."
                    ) from exc

                retryable = status_code is None or status_code == 429 or status_code >= 500
                if status_code is not None and 400 <= status_code < 500 and status_code != 429:
                    raise RuntimeError(
                        f"Groq rejected model {self.model_name!r} with HTTP "
                        f"{status_code}: {exc}"
                    ) from exc
                if not retryable or attempt == MAX_RETRIES - 1:
                    raise RuntimeError(
                        f"Groq request failed after {attempt + 1} attempt(s)"
                        + (f" (HTTP {status_code})" if status_code is not None else "")
                        + "."
                    ) from exc

                delay = min(60, 2**attempt)
                logging.warning(
                    "Temporary Groq request failure%s; retrying in %s seconds.",
                    f" (HTTP {status_code})" if status_code is not None else "",
                    delay,
                )
                time.sleep(delay)

        raise RuntimeError("Groq request retry loop ended unexpectedly.")


def valid_shards(task: dict) -> list[tuple[int, str, bool]]:
    shards = []
    for turn_index, shard_item in enumerate(task.get("shards", []), start=1):
        if isinstance(shard_item, dict):
            shard_text = shard_item.get("shard") or shard_item.get("shard_text") or ""
            underspecified = shard_item.get("is_underspecified", False)
        elif isinstance(shard_item, str):
            shard_text = shard_item
            underspecified = False
        else:
            continue

        if shard_text:
            shards.append((turn_index, shard_text, underspecified))
    return shards


def load_saved_turns(path: Path) -> set[tuple[str, int]]:
    saved_turns = set()
    if not path.exists():
        return saved_turns

    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                saved_turns.add(
                    (record["conversation_id"], int(record["turn_index"]))
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                logging.warning(
                    "Skipping invalid JSONL record at %s:%s: %s",
                    path,
                    line_number,
                    exc,
                )
    return saved_turns


def process_task(
    task: dict,
    backbone: GroqBackend,
    saved_turns: set[tuple[str, int]],
) -> tuple[str, list[dict]]:
    from src.signals.feature_fusion import FeatureBuilder

    if not hasattr(_worker_local, "feature_builder"):
        _worker_local.feature_builder = FeatureBuilder()

    feature_builder = _worker_local.feature_builder
    feature_builder.reset()

    conversation_id = task.get("task_id", "conv_unk")
    source = task.get("source", task.get("task", "general"))
    history = []
    records = []

    for turn_index, shard_text, is_underspecified in valid_shards(task):
        history.append({"role": "user", "content": shard_text})
        assistant_response = backbone.generate(history, temperature=0.7)
        history.append({"role": "assistant", "content": assistant_response})

        features = feature_builder.compute_turn_features(
            conversation_id=conversation_id,
            turn_index=turn_index,
            conversation_history=history,
            assistant_response=assistant_response,
            is_underspecified=is_underspecified,
            backbone=backbone,
        )

        if (conversation_id, turn_index) not in saved_turns:
            record = features.to_dict()
            record["source"] = source
            records.append(record)

    success = judge_eventual_success(task, history, backbone)
    for record in records:
        record["eventual_success"] = success

    return conversation_id, records


def judge_eventual_success(task: dict, history: list[dict], backbone: GroqBackend) -> bool:
    """Label task success from benchmark references and the conversation's answer."""
    assistant_responses = [
        message["content"]
        for message in history
        if message.get("role") == "assistant" and message.get("content")
    ]
    if not assistant_responses:
        return False

    task_type = task.get("task", task.get("source", "unknown"))
    context_fields = {
        "code": (
            "prompt",
            "question_content",
            "test",
            "public_test_cases",
            "starter_code",
        ),
        "database": (
            "fully_specified_question",
            "reference_sql",
            "schema_sql",
            "spider_difficulty",
        ),
        "actions": (
            "fully_specified_question",
            "function",
            "reference_answer",
            "test_category",
        ),
        "math": ("question", "answer"),
        "data2text": (
            "raw_table",
            "table_highlighted_html",
            "references",
        ),
        "summary": ("query", "insights", "documents", "topic"),
    }
    context = {}
    for field in context_fields.get(task_type, ()):
        value = task.get(field)
        if value is not None:
            serialized = json.dumps(value, ensure_ascii=False, default=str)
            context[field] = serialized[:12000]

    judge_prompt = [
        {
            "role": "system",
            "content": (
                "You are a strict evaluator for a research dataset. Decide whether "
                "the assistant's FINAL answer fulfills the original benchmark task. "
                "Use the supplied reference/tests when present; allow equivalent "
                "correct solutions, but do not reward plausible-sounding answers "
                "that omit required results. Return exactly true or false, with no "
                "other text. Return true only when the final answer is substantially "
                "correct and complete."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task_type": task_type,
                    "original_task": context,
                    "assistant_responses_in_order": [
                        answer[:4000] for answer in assistant_responses[-8:]
                    ],
                    "instruction": (
                        "Judge the completed solution at the end of this conversation. "
                        "Earlier assistant responses may be incomplete or corrected later."
                    ),
                },
                ensure_ascii=False,
            ),
        },
    ]
    verdict = backbone.generate(judge_prompt, temperature=0.0).strip().lower()
    if verdict in {"true", "true.", "```true```"}:
        return True
    if verdict in {"false", "false.", "```false```"}:
        return False
    raise ValueError(
        f"Success judge returned an invalid label for {task.get('task_id')}: "
        f"{verdict[:100]!r}"
    )


def run_batch(args: argparse.Namespace, backbone: GroqBackend) -> int:
    if args.batch_index < 0:
        raise ValueError("--batch-index must be zero or greater.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be greater than zero.")
    if args.workers < 1:
        raise ValueError("--workers must be greater than zero.")

    if not args.input_path.exists():
        raise FileNotFoundError(f"Input instructions not found: {args.input_path}")

    all_tasks = json.loads(args.input_path.read_text(encoding="utf-8"))
    if not isinstance(all_tasks, list):
        raise ValueError(f"Expected a JSON array of tasks in {args.input_path}")

    start = args.batch_index * args.batch_size
    tasks = all_tasks[start : start + args.batch_size]
    output_path = args.output_dir / f"batch_{args.batch_index:03d}.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    saved_turns = load_saved_turns(output_path)

    tasks_to_run = []
    for task in tasks:
        conversation_id = task.get("task_id", "conv_unk")
        expected_turns = {
            (conversation_id, turn_index)
            for turn_index, _, _ in valid_shards(task)
        }
        if not expected_turns.issubset(saved_turns):
            tasks_to_run.append(task)

    print(
        f"Batch {args.batch_index}: {len(tasks_to_run)} unfinished task(s) "
        f"of {len(tasks)}; output: {output_path}"
    )
    if not tasks_to_run:
        print("This batch is already complete.")
        return 0

    import torch

    torch.set_num_threads(1)
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_tasks = {
            executor.submit(process_task, task, backbone, saved_turns): task
            for task in tasks_to_run
        }

        with output_path.open("a", encoding="utf-8") as output_file:
            for future in as_completed(future_tasks):
                task = future_tasks[future]
                conversation_id = task.get("task_id", "conv_unk")
                try:
                    _, records = future.result()
                except Exception as exc:
                    logging.exception("Failed to generate conversation %s", conversation_id)
                    failures.append(conversation_id)
                    continue

                for record in records:
                    output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    saved_turns.add(
                        (record["conversation_id"], record["turn_index"])
                    )
                output_file.flush()
                print(f"Saved {conversation_id}: {len(records)} new turn(s)")

    if failures:
        raise RuntimeError(
            "Some conversations failed and were not fully saved: "
            + ", ".join(failures)
            + ". Re-run this batch index to retry them."
        )

    print(f"Batch {args.batch_index} complete.")
    return 0


def merge_batches(batch_dir: Path, output_path: Path) -> int:
    batch_files = sorted(batch_dir.glob("batch_*.jsonl"))
    if not batch_files:
        raise FileNotFoundError(f"No batch_*.jsonl files found in {batch_dir}")

    records_by_turn = {}
    for path in batch_files:
        with path.open("r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    key = (record["conversation_id"], int(record["turn_index"]))
                    records_by_turn.setdefault(key, record)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    logging.warning("Skipping invalid JSONL record at %s:%s: %s", path, line_number, exc)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        for record in records_by_turn.values():
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary_path, output_path)
    print(f"Merged {len(records_by_turn)} unique turns into {output_path}")
    return 0


def run_all(args: argparse.Namespace, backbone: GroqBackend) -> int:
    if not args.input_path.exists():
        raise FileNotFoundError(f"Input instructions not found: {args.input_path}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be greater than zero.")

    tasks = json.loads(args.input_path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise ValueError(f"Expected a JSON array of tasks in {args.input_path}")

    import torch

    torch.set_num_threads(1)
    total_batches = (len(tasks) + args.batch_size - 1) // args.batch_size
    print(f"Generating {len(tasks)} tasks in {total_batches} resumable batches.")

    for batch_index in range(total_batches):
        batch_args = argparse.Namespace(
            input_path=args.input_path,
            output_dir=args.output_dir,
            batch_index=batch_index,
            batch_size=args.batch_size,
            workers=args.workers,
        )
        run_batch(batch_args, backbone)

    merge_batches(args.output_dir, args.merged_output)

    if args.replace_raw_logs:
        if args.merged_output.resolve() == DEFAULT_RAW_LOGS.resolve():
            raise ValueError("Merged output must not be the raw log path before backup.")
        if DEFAULT_RAW_LOGS.exists():
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            backup_path = DEFAULT_RAW_LOGS.with_name(
                f"{DEFAULT_RAW_LOGS.stem}.backup-{timestamp}{DEFAULT_RAW_LOGS.suffix}"
            )
            os.replace(DEFAULT_RAW_LOGS, backup_path)
            print(f"Backed up existing raw logs to {backup_path}")
        DEFAULT_RAW_LOGS.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = DEFAULT_RAW_LOGS.with_name(
            DEFAULT_RAW_LOGS.name + ".tmp"
        )
        with args.merged_output.open("rb") as source, temporary_path.open("wb") as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
        os.replace(temporary_path, DEFAULT_RAW_LOGS)
        print(f"Replaced {DEFAULT_RAW_LOGS} with labeled Groq-generated data.")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate resumable, concurrent Groq-backed pipeline batches."
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--merged-output", type=Path, default=DEFAULT_MERGED_OUTPUT)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every task in the input file, resuming completed batches.",
    )
    parser.add_argument(
        "--replace-raw-logs",
        action="store_true",
        help="After a successful --all run, back up and replace data/processed/raw_logs.jsonl.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.merge:
        return merge_batches(args.output_dir, args.merged_output)

    from dotenv import load_dotenv

    load_dotenv()
    api_keys = load_api_keys()
    print(
        f"Loaded {len(api_keys)} Groq API key variable(s): "
        f"{', '.join(name for name, _ in api_keys)}; using model {args.model}."
    )
    backbone = GroqBackend(api_keys, args.model)
    if args.replace_raw_logs and not args.all:
        raise ValueError("--replace-raw-logs requires --all.")
    if args.all:
        return run_all(args, backbone)
    return run_batch(args, backbone)


if __name__ == "__main__":
    raise SystemExit(main())
