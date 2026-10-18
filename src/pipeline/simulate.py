import argparse
import json
import os
import time
from pathlib import Path
from tqdm import tqdm
from src.pipeline.backends import LLMBackend
from src.signals.feature_fusion import FeatureBuilder
from src.pipeline.dataset_builder import build_dataset


def get_completed_task_ids(output_logs_path: str) -> set:
    """Reads raw_logs.jsonl and gathers conversation_ids already generated."""
    completed = set()
    if os.path.exists(output_logs_path):
        with open(output_logs_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        if "conversation_id" in record:
                            completed.add(record["conversation_id"])
                    except json.JSONDecodeError:
                        continue
    return completed


def judge_eventual_success(task: dict, history: list[dict], backbone: LLMBackend) -> bool:
    """Evaluates whether the assistant's final response satisfied the task requirements."""
    assistant_responses = [
        m["content"]
        for m in history
        if m.get("role") == "assistant" and m.get("content")
    ]
    if not assistant_responses:
        return False

    task_type = task.get("task", task.get("source", "unknown"))
    final_response = assistant_responses[-1]

    reference_info = []
    for field in (
        "prompt",
        "question",
        "test",
        "reference_sql",
        "reference_answer",
        "answer",
        "public_test_cases",
        "starter_code",
    ):
        val = task.get(field)
        if val:
            reference_info.append(
                f"{field}: {json.dumps(val, ensure_ascii=False, default=str)[:1000]}"
            )
    ref_text = (
        "\n".join(reference_info)
        if reference_info
        else str(task.get("shards", []))[:1000]
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

    verdict_raw = (
        backbone.generate(judge_prompt, temperature=0.0, max_tokens=10)
        .strip()
        .lower()
    )
    return "true" in verdict_raw


def run_simulation(
    sharded_data_path: str = "data/raw/lost_in_conversation/data/sharded_instructions_600.json",
    output_logs_path: str = "data/processed/raw_logs.jsonl",
    splits_dir: str = "data/splits",
    provider: str = None,
    model_name: str = None,
    batch_size: int = 5,
    n_resamples: int = 2,
    cool_down_seconds: float = 0.2,
):
    os.makedirs(os.path.dirname(output_logs_path), exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)

    # 1. Identify tasks already logged in raw_logs.jsonl
    completed_ids = get_completed_task_ids(output_logs_path)

    with open(sharded_data_path, "r", encoding="utf-8") as f:
        all_tasks = json.load(f)

    # 2. Filter out already processed tasks
    remaining_tasks = [
        t for t in all_tasks if t.get("task_id", "conv_unk") not in completed_ids
    ]

    if not remaining_tasks:
        print("All tasks in the benchmark have been successfully processed!", flush=True)
        build_dataset(input_path=output_logs_path, output_dir=splits_dir)
        return

    print(f"Already processed: {len(completed_ids)} tasks.", flush=True)
    print(f"Remaining tasks: {len(remaining_tasks)}.", flush=True)
    actual_batch = min(batch_size, len(remaining_tasks)) if batch_size > 0 else len(remaining_tasks)
    print(f"Processing next batch of {actual_batch} tasks...\n", flush=True)

    # 3. Load backend and feature builder
    backbone = LLMBackend(provider=provider, model_name=model_name)
    print(f"Using backbone: provider={backbone.provider}, model={backbone.model_name}", flush=True)
    feature_builder = FeatureBuilder()

    tasks_to_run = remaining_tasks[:actual_batch]
    tasks_completed_in_batch = 0

    # 4. Run simulation turn-by-turn
    for task in tqdm(tasks_to_run, desc="Simulating Batch"):
        conv_id = task.get("task_id", "conv_unk")
        task_source = task.get("source", task.get("task", "general"))
        shards = task.get("shards", [])

        history = []
        feature_builder.reset()
        task_records = []

        for t_idx, shard_item in enumerate(shards, start=1):
            if isinstance(shard_item, dict):
                shard_text = (
                    shard_item.get("shard") or shard_item.get("shard_text") or ""
                )
                is_underspecified = shard_item.get("is_underspecified", False)
            elif isinstance(shard_item, str):
                shard_text = shard_item
                is_underspecified = False
            else:
                continue

            if not shard_text:
                continue

            history.append({"role": "user", "content": shard_text})
            assistant_response = backbone.generate(history, temperature=0.7, max_tokens=256)
            history.append({"role": "assistant", "content": assistant_response})

            features = feature_builder.compute_turn_features(
                conversation_id=conv_id,
                turn_index=t_idx,
                conversation_history=history,
                assistant_response=assistant_response,
                is_underspecified=is_underspecified,
                backbone=backbone,
                n_resamples=n_resamples,
            )

            record = features.to_dict()
            record["source"] = task_source
            task_records.append(record)
            print(
                f"  [{conv_id}] Turn {t_idx}/{len(shards)} computed "
                f"(var={features.self_consistency_variance:.2f}, contra={features.contradiction_score:.2f})",
                flush=True,
            )

        # 5. Judge task success
        success = judge_eventual_success(task, history, backbone)
        for record in task_records:
            record["eventual_success"] = success

        # 6. Append to log file and flush
        with open(output_logs_path, "a", encoding="utf-8") as out_f:
            for record in task_records:
                out_f.write(json.dumps(record) + "\n")
            out_f.flush()

        tasks_completed_in_batch += 1
        print(f"[{conv_id}] completed {len(task_records)} turns | Eventual success: {success}", flush=True)

        # Update train/val/test splits incrementally after each task
        build_dataset(input_path=output_logs_path, output_dir=splits_dir)

        # Thermal/rate pause
        if cool_down_seconds > 0:
            time.sleep(cool_down_seconds)

    print(f"\nDone! Successfully simulated and logged {tasks_completed_in_batch} tasks to {output_logs_path}.", flush=True)
    print(f"Remaining tasks: {len(remaining_tasks) - tasks_completed_in_batch}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Simulate multi-turn tasks and log drift features.")
    parser.add_argument("--provider", type=str, default=None, help="LLM provider: groq or ollama (auto-detects)")
    parser.add_argument("--model", type=str, default=None, help="LLM model name (defaults to qwen/qwen3.8-27b on Groq)")
    parser.add_argument("--batch-size", type=int, default=5, help="Number of tasks to simulate in this run")
    parser.add_argument("--n-resamples", type=int, default=3, help="Number of self-consistency resamples per turn")
    parser.add_argument("--cool-down", type=float, default=0.5, help="Cooldown delay between tasks in seconds")
    parser.add_argument("--sharded-data", type=str, default="data/raw/lost_in_conversation/data/sharded_instructions_600.json")
    parser.add_argument("--output-logs", type=str, default="data/processed/raw_logs.jsonl")
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_simulation(
        sharded_data_path=args.sharded_data,
        output_logs_path=args.output_logs,
        splits_dir=args.splits_dir,
        provider=args.provider,
        model_name=args.model,
        batch_size=args.batch_size,
        n_resamples=args.n_resamples,
        cool_down_seconds=args.cool_down,
    )