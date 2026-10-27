import argparse
import json
import os
import time
from tqdm import tqdm

from src.pipeline.backends import LLMBackend, LLMError, DailyLimitError
from src.signals.feature_fusion import FeatureBuilder
from src.pipeline.dataset_builder import build_dataset

# Step 3+ adds src/pipeline/evaluators.py. Until it exists, everything falls
# back to the LLM judge so this file runs today.
try:
    from src.pipeline.evaluators import evaluate as exec_evaluate
except ImportError:
    exec_evaluate = None

EXTRACTOR_VERSION = "v5"

# Per-family output budgets for the assistant's turn (tokens).
# Code gets a big budget so solutions are not cut off mid-function.
MAX_TOKENS = {
    "livecodebench": 2048,
    "humaneval": 1024,
    "code": 1024,
    "spider": 512,
    "database": 512,
    "bfcl": 512,
    "actions": 512,
    "math": 512,
    "data2text": 400,
    "summary": 400,
    "default": 512,
    "lcb": 2048,
}


def max_tokens_for(source: str) -> int:
    s = str(source).lower()
    for key, val in MAX_TOKENS.items():
        if key != "default" and key in s:
            return val
    return MAX_TOKENS["default"]


def normalize_shards(task: dict) -> list[dict]:
    """Return shards as [{'shard': str, 'is_underspecified': bool}, ...]."""
    out = []
    for s in task.get("shards", []):
        if isinstance(s, str):
            s = {"shard": s}
        if not isinstance(s, dict):
            continue
        text = (s.get("shard") or s.get("shard_text") or s.get("content") or "").strip()
        if text:
            out.append({"shard": text, "is_underspecified": s.get("is_underspecified")})
    n = len(out)
    for i, sh in enumerate(out):
        if sh["is_underspecified"] is None:      # flag missing: info is incomplete until the last shard
            sh["is_underspecified"] = i < n - 1
        sh["is_underspecified"] = bool(sh["is_underspecified"])
    return out


def get_completed_keys(output_logs_path: str, legacy_model: str) -> set:
    """Keys of (conversation_id, assistant_model, run_index) already logged.
    Old rows without assistant_model are attributed to `legacy_model`."""
    done = set()
    if os.path.exists(output_logs_path):
        with open(output_logs_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "conversation_id" in r:
                    done.add(
                        (
                            r["conversation_id"],
                            r.get("assistant_model") or legacy_model,
                            r.get("run_index", 0),
                        )
                    )
    return done


def judge_eventual_success(task: dict, history: list[dict], backbone: LLMBackend) -> bool:
    """LLM judge, used only for families with no executable check."""
    assistant_responses = [
        m["content"] for m in history if m.get("role") == "assistant" and m.get("content")
    ]
    if not assistant_responses:
        return False

    task_type = task.get("task", task.get("source", "unknown"))
    final_response = assistant_responses[-1]

    reference_info = []
    for field in (
        "prompt", "question", "test", "reference_sql", "reference_answer",
        "answer", "public_test_cases", "starter_code",
    ):
        val = task.get(field)
        if val:
            reference_info.append(
                f"{field}: {json.dumps(val, ensure_ascii=False, default=str)[:1000]}"
            )
    ref_text = (
        "\n".join(reference_info) if reference_info else str(task.get("shards", []))[:1000]
    )

    judge_prompt = [
        {
            "role": "system",
            "content": (
                "You are an objective judge evaluating whether an AI assistant's final response "
                "correctly and completely solves the user's multi-turn task. "
                "Check requirements and references carefully. "
                "Reply with ONLY the word TRUE if the final response is correct and complete, "
                "or FALSE if it is incomplete, incorrect, or committed to a wrong assumption. "
                "Output strictly TRUE or FALSE."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Task Type: {task_type}\n"
                f"Reference/Requirements:\n{ref_text[:3000]}\n\n"
                f"Assistant Final Answer:\n{final_response[:3000]}\n\n"
                "Verdict (TRUE or FALSE):"
            ),
        },
    ]
    # 300 tokens so a thinking model can finish before it answers; the backend
    # strips <think> blocks and raises LLMError on an empty reply.
    verdict_raw = backbone.generate(judge_prompt, temperature=0.0, max_tokens=300).strip().lower()
    return "true" in verdict_raw


PROMPT_VERSION = "lcb_codeonly_v1"


def first_shard_text(task, shard_text):
    """Prepend a format instruction for LiveCodeBench tasks (shard 1 only)."""
    if str(task.get("source", "")).lower().startswith("lcb"):
        starter = (task.get("starter_code") or "").rstrip()
        return (
            "You will receive a programming problem in pieces over several messages. "
            "In EVERY reply, respond with ONLY one complete Python solution in a single "
            "```python code block, with no explanation or other text. "
            "Use exactly this starter signature:\n"
            f"```python\n{starter}\n```\n"
            "Update your solution as new details arrive.\n\n"
            f"First detail: {shard_text}"
        )
    return shard_text


def simulate_one(task, backbone, feature_builder, n_resamples, run_index):
    """Runs one conversation. Returns (records, transcript). Raises LLMError on failure."""
    conv_id = task.get("task_id", "conv_unk")
    task_source = task.get("source", task.get("task", "general"))
    shards = normalize_shards(task)
    turn_budget = max_tokens_for(task_source)

    history = []
    feature_builder.reset()
    task_records = []
    last_finish = None
    any_truncated = False

    for t_idx, shard in enumerate(shards, start=1):
        text = first_shard_text(task, shard["shard"]) if t_idx == 1 else shard["shard"]
        history.append({"role": "user", "content": text})
        assistant_response = backbone.generate(history, temperature=0.7, max_tokens=turn_budget)
        last_finish = backbone.last_finish_reason
        truncated = last_finish == "length"
        any_truncated = any_truncated or truncated
        history.append({"role": "assistant", "content": assistant_response})

        features = feature_builder.compute_turn_features(
            conversation_id=conv_id,
            turn_index=t_idx,
            conversation_history=list(history),
            assistant_response=assistant_response,
            is_underspecified=shard["is_underspecified"],
            backbone=backbone,
            n_resamples=n_resamples,
            resample_max_tokens=turn_budget,
        )

        record = features.to_dict()
        record["source"] = task_source
        record["finish_reason"] = last_finish
        record["truncated"] = truncated
        task_records.append(record)
        print(
            f"  [{conv_id}] Turn {t_idx}/{len(shards)} "
            f"(var={features.self_consistency_variance:.2f}, "
            f"contra={features.contradiction_score:.2f}, finish={last_finish})",
            flush=True,
        )

    if not any(m["role"] == "assistant" and m["content"].strip() for m in history):
        raise LLMError("no assistant replies recorded")

    # ---- label ----
    final_text = next(
        (m["content"] for m in reversed(history) if m["role"] == "assistant"), ""
    )
    success, valid, reason = None, True, "llm_judge"
    if exec_evaluate is not None:
        success, valid, reason = exec_evaluate(task, final_text, last_finish)
    if success is None:  # no executable check for this family
        success = judge_eventual_success(task, history, backbone)
        valid, reason = (not (last_finish == "length")), "llm_judge"
        if not valid:
            reason = "truncated"

    for r in task_records:
        r.update(
            eventual_success=bool(success),
            label_valid=bool(valid),
            label_reason=reason,
            assistant_model=backbone.model_name,
            run_index=run_index,
            extractor_version=EXTRACTOR_VERSION,
            n_resamples=n_resamples,
            any_turn_truncated=any_truncated,
            prompt_version=PROMPT_VERSION,
        )

    transcript = {
        "conversation_id": conv_id,
        "assistant_model": backbone.model_name,
        "run_index": run_index,
        "source": task_source,
        "history": history,
        "eventual_success": bool(success),
        "label_valid": bool(valid),
        "label_reason": reason,
    }
    return task_records, transcript


def run_simulation(
    sharded_data_path: str = "data/raw/lost_in_conversation/data/sharded_instructions_600.json",
    output_logs_path: str = "data/processed/raw_logs.jsonl",
    transcripts_path: str = "data/processed/transcripts.jsonl",
    splits_dir: str = "data/splits",
    provider: str = None,
    model_name: str = None,
    batch_size: int = 5,
    n_resamples: int = 2,
    cool_down_seconds: float = 0.2,
    sources: str = None,
    run_index: int = 0,
    legacy_model: str = "qwen/qwen3.8-27b",
):
    os.makedirs(os.path.dirname(output_logs_path), exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)

    with open(sharded_data_path, "r", encoding="utf-8") as f:
        all_tasks = json.load(f)
        if sources:
            wanted = {s.strip().lower() for s in sources.split(",") if s.strip()}
            all_tasks = [
                t for t in all_tasks
                if str(t.get("task", "")).lower() in wanted
                or str(t.get("source", "")).lower() in wanted
            ]
            print(f"--sources {sorted(wanted)} -> {len(all_tasks)} tasks", flush=True)

    backbone = LLMBackend(provider=provider, model_name=model_name)
    print(f"Using backbone: provider={backbone.provider}, model={backbone.model_name}, run={run_index}", flush=True)

    completed = get_completed_keys(output_logs_path, legacy_model)
    remaining = [
        t for t in all_tasks
        if (t.get("task_id", "conv_unk"), backbone.model_name, run_index) not in completed
    ]

    if not remaining:
        print("All tasks for this model/run are already processed!", flush=True)
        build_dataset(input_path=output_logs_path, output_dir=splits_dir)
        return

    actual_batch = min(batch_size, len(remaining)) if batch_size > 0 else len(remaining)
    print(f"Already processed (all models/runs): {len(completed)}. Remaining for this model/run: {len(remaining)}.", flush=True)
    print(f"Processing next batch of {actual_batch} tasks...\n", flush=True)

    feature_builder = FeatureBuilder()
    done_count = 0
    skipped = 0

    try:
        for task in tqdm(remaining[:actual_batch], desc="Simulating Batch"):
            conv_id = task.get("task_id", "conv_unk")
            try:
                records, transcript = simulate_one(
                    task, backbone, feature_builder, n_resamples, run_index
                )
            except DailyLimitError as e:
                print(f"\nDaily token limit reached, stopping cleanly.\n{str(e)[:300]}", flush=True)
                break
            except LLMError as e:
                # Do not write partial/garbage conversations. They will be retried next run.
                skipped += 1
                print(f"[{conv_id}] SKIPPED due to LLM error: {str(e)[:200]}", flush=True)
                continue

            with open(output_logs_path, "a", encoding="utf-8") as out_f:
                for r in records:
                    out_f.write(json.dumps(r) + "\n")
                out_f.flush()
            with open(transcripts_path, "a", encoding="utf-8") as tf:
                tf.write(json.dumps(transcript, ensure_ascii=False) + "\n")

            done_count += 1
            print(
                f"[{conv_id}] completed {len(records)} turns | success={transcript['eventual_success']} "
                f"| label_valid={transcript['label_valid']} ({transcript['label_reason']})",
                flush=True,
            )
            if cool_down_seconds > 0:
                time.sleep(cool_down_seconds)
    finally:
        # Build splits once at the end, not after every task.
        build_dataset(input_path=output_logs_path, output_dir=splits_dir)

    print(f"\nDone. Logged {done_count} tasks, skipped {skipped}.", flush=True)
    print(f"Remaining for this model/run: {len(remaining) - done_count}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="Simulate multi-turn tasks and log drift features.")
    p.add_argument("--provider", type=str, default=None, help="LLM provider: groq or ollama (auto-detects)")
    p.add_argument("--model", type=str, default=None, help="LLM model name")
    p.add_argument("--batch-size", type=int, default=5, help="Number of tasks to simulate in this run")
    p.add_argument("--n-resamples", type=int, default=2, help="Self-consistency resamples per turn")
    p.add_argument("--cool-down", type=float, default=0.5, help="Cooldown between tasks in seconds")
    p.add_argument("--sharded-data", type=str, default="data/raw/lost_in_conversation/data/sharded_instructions_600.json")
    p.add_argument("--output-logs", type=str, default="data/processed/raw_logs.jsonl")
    p.add_argument("--transcripts", type=str, default="data/processed/transcripts.jsonl")
    p.add_argument("--splits-dir", type=str, default="data/splits")
    p.add_argument("--sources", type=str, default=None, help='Comma-separated task types, e.g. "livecodebench,spider"')
    p.add_argument("--run-index", type=int, default=0, help="Repeat index for multi-run scaling")
    p.add_argument("--legacy-model", type=str, default="qwen/qwen3.8-27b",
                   help="Model name to attribute to old log rows that have no assistant_model")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run_simulation(
        sharded_data_path=a.sharded_data,
        output_logs_path=a.output_logs,
        transcripts_path=a.transcripts,
        splits_dir=a.splits_dir,
        provider=a.provider,
        model_name=a.model,
        batch_size=a.batch_size,
        n_resamples=a.n_resamples,
        cool_down_seconds=a.cool_down,
        sources=a.sources,
        run_index=a.run_index,
        legacy_model=a.legacy_model,
    )