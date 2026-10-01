import json
import os
import time
from tqdm import tqdm
from src.pipeline.backends import LLMBackend
from src.signals.feature_fusion import FeatureBuilder

def get_completed_task_ids(output_logs_path: str) -> set:
    """Reads raw_logs.jsonl and gathers conversation_ids already generated."""
    completed = set()
    if os.path.exists(output_logs_path):
        with open(output_logs_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        if "conversation_id" in record:
                            completed.add(record["conversation_id"])
                    except json.JSONDecodeError:
                        continue
    return completed

def run_simulation(
    sharded_data_path: str = "data/raw/lost_in_conversation/data/sharded_instructions_600.json",
    output_logs_path: str = "data/processed/raw_logs.jsonl",
    model_name: str = "llama3.2:3b",
    batch_size: int = 2,           # Runs 2 tasks per execution
    cool_down_seconds: float = 3.0 # Pauses between tasks to keep Mac cool
):
    os.makedirs(os.path.dirname(output_logs_path), exist_ok=True)
    
    # 1. Identify tasks already logged in raw_logs.jsonl
    completed_ids = get_completed_task_ids(output_logs_path)
    
    with open(sharded_data_path, "r") as f:
        all_tasks = json.load(f)

    # 2. Filter out already processed tasks
    remaining_tasks = [
        t for t in all_tasks 
        if t.get("task_id", "conv_unk") not in completed_ids
    ]

    if not remaining_tasks:
        print("All tasks in the benchmark have been successfully processed!")
        return

    print(f"Already processed: {len(completed_ids)} tasks.")
    print(f"Remaining tasks: {len(remaining_tasks)}.")
    print(f"Processing next batch of {min(batch_size, len(remaining_tasks))} tasks...\n")

    # 3. Load models ONCE
    backbone = LLMBackend(provider="ollama", model_name=model_name)
    feature_builder = FeatureBuilder()

    # 4. Open log file in APPEND ("a") mode to concatenate at EOF
    with open(output_logs_path, "a") as out_f:
        tasks_to_run = remaining_tasks[:batch_size]

        for task in tqdm(tasks_to_run, desc="Simulating Batch"):
            conv_id = task.get("task_id", "conv_unk")
            task_source = task.get("source", "general")
            shards = task.get("shards", [])
            
            history = []
            feature_builder.reset()

            for t_idx, shard_item in enumerate(shards, start=1):
                if isinstance(shard_item, dict):
                    shard_text = shard_item.get("shard") or shard_item.get("shard_text") or ""
                    is_underspecified = shard_item.get("is_underspecified", False)
                elif isinstance(shard_item, str):
                    shard_text = shard_item
                    is_underspecified = False
                else:
                    continue

                if not shard_text:
                    continue

                history.append({"role": "user", "content": shard_text})
                assistant_response = backbone.generate(history, temperature=0.7)
                history.append({"role": "assistant", "content": assistant_response})

                features = feature_builder.compute_turn_features(
                    conversation_id=conv_id,
                    turn_index=t_idx,
                    conversation_history=history,
                    assistant_response=assistant_response,
                    is_underspecified=is_underspecified,
                    backbone=backbone
                )

                record = features.to_dict()
                record["source"] = task_source
                
                # Appends record to EOF and flushes buffer to disk immediately
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()

            # Thermal pause between tasks
            time.sleep(cool_down_seconds)

    print(f"\nDone! Batch appended to {output_logs_path}.")
    print(f"Remaining tasks left: {len(remaining_tasks) - len(tasks_to_run)}")

if __name__ == "__main__":
    run_simulation(batch_size=2)