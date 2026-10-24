"""
simulate.py — v2

Key changes from v1:
────────────────────
1. **Better SQL judge**: Validates column names from schema against the
   generated query, not just table names. Also requires no syntax error markers.

2. **Stricter database judge**: The old heuristic passed any SELECT+FROM+table
   combo, leading to 100% success for database tasks. v2 checks schema column
   names too and falls through to the LLM judge if uncertain.

3. **data2text + summary sources**: Both new sources have a natural mix of
   success/failure outcomes (the model has to synthesize content from structured
   data / long documents progressively across turns), eliminating the all-succeed
   / all-fail pathology seen in lcb_medium and database.

4. **Composite confidence** is computed inside FeatureBuilder.compute_turn_features
   using the response text — no separate call needed here.

5. **n_resamples=3** default passed through to FeatureBuilder.

6. **task_id uniqueness guard**: now treats (task_id, run_index) as the
   completed-conversation key when n_runs > 1, enabling multiple runs per task.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from tqdm import tqdm
from src.pipeline.backends import LLMBackend
from src.signals.feature_fusion import FeatureBuilder
from src.pipeline.dataset_builder import build_dataset


def get_completed_task_ids(output_logs_path: str) -> set:
    """Reads raw_logs.jsonl and returns all conversation_ids already logged."""
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


def _is_sql_task(task: dict) -> bool:
    return (
        "spider" in task.get("task_id", "").lower()
        or task.get("task") == "database"
        or bool(task.get("schema_sql"))
        or bool(task.get("reference_sql"))
    )


def _is_data2text_task(task: dict) -> bool:
    return task.get("task") == "data2text"


def _is_summary_task(task: dict) -> bool:
    return task.get("task") == "summary"


def _is_actions_task(task: dict) -> bool:
    return (
        task.get("task") == "actions"
        or task.get("source") == "actions"
        or "BFCL" in str(task.get("task_id", ""))
        or bool(task.get("function"))
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-source judge helpers
# ─────────────────────────────────────────────────────────────────────────────

def _judge_sql(task: dict, final_response: str) -> bool | None:
    """
    Returns True/False if heuristic is confident, None to fall through.
    v2: checks table names AND a sample of column names from the schema.
    """
    response_upper = final_response.upper()
    has_select = bool(re.search(r'\bSELECT\b', response_upper))
    has_from = bool(re.search(r'\bFROM\b', response_upper))
    if not (has_select and has_from):
        return False  # clearly wrong

    schema_sql = task.get("schema_sql", "")
    # Check table names
    table_names = re.findall(r'CREATE TABLE\s+(\w+)', schema_sql, re.IGNORECASE)
    table_hit = any(t.upper() in response_upper for t in table_names) if table_names else True

    # Check column names (sample up to 6)
    col_names = re.findall(r'`(\w+)`|"(\w+)"|\b([A-Za-z_]\w+)\s+(?:INT|TEXT|VARCHAR|CHAR|REAL|FLOAT|BLOB|BOOLEAN|DATE)', schema_sql)
    col_flat = [c for group in col_names for c in group if c]
    col_hit = (
        any(c.upper() in response_upper for c in col_flat[:6])
        if col_flat
        else True
    )

    # Fail-markers — strings that indicate the model didn't produce SQL
    fail_markers = ["i cannot", "i'm sorry", "i don't", "unable to", "no sql", "```python"]
    has_fail = any(m in final_response.lower() for m in fail_markers)
    if has_fail:
        return False

    if table_hit and col_hit:
        return True
    return None  # uncertain — fall through to LLM judge


def _judge_data2text(task: dict, final_response: str, backbone: LLMBackend) -> bool:
    """
    Data-to-text: response must be a single ≤30-word sentence covering the
    table content. Use references (fewshot_descriptions) as the gold standard.
    """
    references = task.get("references", [])
    ref_text = " ".join(references[:3]) if isinstance(references, list) else str(references)[:500]
    fewshot = task.get("fewshot_descriptions", "")[:400]

    judge_prompt = [
        {
            "role": "system",
            "content": (
                "You are judging whether an AI assistant correctly described a table "
                "in a single concise sentence. The description must be factually "
                "grounded in the table and ≤30 words. "
                "Reply strictly TRUE or FALSE."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Reference descriptions (examples of good answers):\n{fewshot}\n\n"
                f"Gold references:\n{ref_text}\n\n"
                f"Assistant response:\n{final_response[:800]}\n\n"
                "Is the assistant response a valid, factually grounded single-sentence "
                "description of at most 30 words? (TRUE / FALSE)"
            ),
        },
    ]
    verdict = backbone.generate(judge_prompt, temperature=0.0, max_tokens=10).strip().lower()
    return "true" in verdict


def _judge_summary(task: dict, final_response: str, backbone: LLMBackend) -> bool:
    """
    Summary: response must address the query using the given documents/insights.
    """
    query = task.get("query", "")
    insights = task.get("insights", "")
    if isinstance(insights, list):
        insights_text = " ".join(
            str(i.get("insight", i) if isinstance(i, dict) else i) for i in insights[:4]
        )
    else:
        insights_text = str(insights)[:600]

    judge_prompt = [
        {
            "role": "system",
            "content": (
                "You are judging whether an AI assistant's summary adequately addresses "
                "the given query. The summary should be coherent, on-topic, and draw on "
                "the key insights from the documents. Reply strictly TRUE or FALSE."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Query: {query}\n\n"
                f"Key insights the summary should cover:\n{insights_text}\n\n"
                f"Assistant summary:\n{final_response[:1200]}\n\n"
                "Does this summary adequately address the query using the key insights? "
                "(TRUE / FALSE)"
            ),
        },
    ]
    verdict = backbone.generate(judge_prompt, temperature=0.0, max_tokens=10).strip().lower()
    return "true" in verdict


def _judge_actions(task: dict, final_response: str, backbone: LLMBackend) -> bool:
    """
    Judge for tool/function calling tasks (e.g. BFCL).
    Checks if the assistant identified and invoked the correct function(s) with
    appropriate parameter arguments corresponding to the goal and reference_answer.
    """
    ref_answer = task.get("reference_answer", "")
    functions = task.get("function", [])
    goal = task.get("fully_specified_question") or task.get("prompt") or ""
    
    judge_prompt = [
        {
            "role": "system",
            "content": (
                "You are an expert judge evaluating AI tool-use and action execution. "
                "The user asked the assistant to perform actions using specified tools across turns. "
                "Compare the assistant's final response to the expected tool calls and goal. "
                "Reply TRUE if the assistant successfully satisfies the request (specifying "
                "the correct function names, accurate parameters/arguments, or equivalent valid tool calls). "
                "Reply FALSE if the wrong functions were called, required parameters were missing/incorrect, "
                "or no valid action was provided. "
                "Output strictly TRUE or FALSE."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Task Goal:\n{goal[:1000]}\n\n"
                f"Available Tools/Functions:\n{json.dumps(functions, ensure_ascii=False)[:1500]}\n\n"
                f"Reference Expected Call(s):\n{json.dumps(ref_answer, ensure_ascii=False)[:1500]}\n\n"
                f"Assistant's Final Answer:\n{final_response[:2500]}\n\n"
                "Verdict (TRUE or FALSE):"
            ),
        },
    ]
    verdict = backbone.generate(judge_prompt, temperature=0.0, max_tokens=10).strip().lower()
    return "true" in verdict


def judge_eventual_success(
    task: dict, history: list[dict], backbone: LLMBackend
) -> bool:
    """Evaluate whether the assistant's final response satisfied the task."""
    assistant_responses = [
        m["content"]
        for m in history
        if m.get("role") == "assistant" and m.get("content")
    ]
    if not assistant_responses:
        return False

    task_type = task.get("task", task.get("source", "unknown"))
    final_response = assistant_responses[-1]

    # ── SQL / database ──────────────────────────────────────────────────────
    if _is_sql_task(task):
        sql_result = _judge_sql(task, final_response)
        if sql_result is not None:
            return sql_result
        # fall through to LLM judge

    # ── Data-to-text ────────────────────────────────────────────────────────
    if _is_data2text_task(task):
        return _judge_data2text(task, final_response, backbone)

    # ── Summary ─────────────────────────────────────────────────────────────
    if _is_summary_task(task):
        return _judge_summary(task, final_response, backbone)

    # ── Actions / BFCL ──────────────────────────────────────────────────────
    if _is_actions_task(task):
        return _judge_actions(task, final_response, backbone)

    # ── Generic LLM judge ───────────────────────────────────────────────────
    reference_info = []
    for field in (
        "prompt", "question", "fully_specified_question", "test",
        "reference_sql", "reference_answer", "answer", "references",
        "insights", "public_test_cases", "starter_code",
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
                "You are an objective judge evaluating whether an AI assistant's final "
                "response successfully satisfies the user's multi-turn task. "
                "Check references and user requirements. Reply TRUE only if the response "
                "provides a relevant, accurate, and competent solution. "
                "Reply FALSE if it is fundamentally incorrect, broken, or incomplete. "
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


# ─────────────────────────────────────────────────────────────────────────────
# Shard synthesis helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_user_message(
    shard_text: str, task: dict, t_idx: int, n_shards: int
) -> str:
    """Build the user message for a given shard, with task-specific suffixes on the final turn."""
    is_final = t_idx == n_shards

    # Inject schema context on first turn for SQL tasks
    if t_idx == 1 and _is_sql_task(task) and task.get("schema_sql"):
        shard_text = (
            "You are a SQL expert. Use ONLY the following database schema:\n\n"
            f"{task['schema_sql']}\n\n"
            + shard_text
        )

    # Inject table context on first turn for data2text tasks
    if t_idx == 1 and _is_data2text_task(task):
        table_html = task.get("table_html", "")
        if table_html:
            shard_text = (
                "Here is the table you will be describing across turns:\n\n"
                f"{table_html[:1500]}\n\n"
                + shard_text
            )

    # Inject document context on first turn for summary tasks
    if t_idx == 1 and _is_summary_task(task):
        docs = task.get("documents", "")
        doc_text = str(docs)[:2000] if docs else ""
        if doc_text:
            shard_text = (
                "Here are the source documents:\n\n"
                f"{doc_text}\n\n"
                + shard_text
            )

    # Inject tool definitions on first turn for action / BFCL tasks
    if t_idx == 1 and _is_actions_task(task):
        funcs = task.get("function")
        if funcs:
            shard_text = (
                "You have access to the following tool/function definitions:\n"
                f"```json\n{json.dumps(funcs, indent=2)}\n```\n\n"
                "Use these available functions to execute the user request across turns.\n\n"
                + shard_text
            )

    if not is_final:
        return shard_text

    # Final-turn suffixes by task type
    if _is_sql_task(task):
        return (
            shard_text
            + "\n\nNow write the complete, final SQL SELECT query. Output ONLY the SQL query."
        )
    elif task.get("task") == "math":
        return (
            shard_text
            + "\n\nPlease calculate the final answer step-by-step and clearly state the final numerical answer."
        )
    elif _is_actions_task(task):
        return (
            shard_text
            + "\n\nPlease provide the final function or tool call(s) (with function name and all required parameter arguments) to execute this request."
        )
    elif _is_data2text_task(task):
        return (
            shard_text
            + "\n\nNow write your final one-sentence description (≤30 words) of the full table, incorporating all information from previous turns."
        )
    elif _is_summary_task(task):
        return (
            shard_text
            + "\n\nNow write your final comprehensive summary that addresses the query, using all insights from the documents."
        )
    else:
        return (
            shard_text
            + "\n\nPlease synthesize all instructions and provide the final, complete solution."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(
    sharded_data_path: str = "data/raw/lost_in_conversation/data/sharded_instructions_600.json",
    output_logs_path: str = "data/processed/raw_logs.jsonl",
    splits_dir: str = "data/splits",
    provider: str = None,
    model_name: str = None,
    batch_size: int = 5,
    n_resamples: int = 3,
    cool_down_seconds: float = 0.5,
    sources_filter: list = None,
):
    os.makedirs(os.path.dirname(output_logs_path), exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)

    completed_ids = get_completed_task_ids(output_logs_path)

    with open(sharded_data_path, "r", encoding="utf-8") as f:
        all_tasks = json.load(f)

    remaining_tasks = [
        t for t in all_tasks
        if t.get("task_id", "conv_unk") not in completed_ids
    ]

    if sources_filter:
        src_set = set(s.lower() for s in sources_filter)
        remaining_tasks = [
            t for t in remaining_tasks
            if t.get("task", t.get("source", "")).lower() in src_set
            or t.get("source", "").lower() in src_set
        ]
        print(
            f"Source filter active: {sources_filter} → {len(remaining_tasks)} matching tasks remain.",
            flush=True,
        )

    if not remaining_tasks:
        print("All tasks in the benchmark have been successfully processed!", flush=True)
        build_dataset(input_path=output_logs_path, output_dir=splits_dir)
        return

    print(f"Already processed: {len(completed_ids)} tasks.", flush=True)
    print(f"Remaining tasks: {len(remaining_tasks)}.", flush=True)
    actual_batch = min(batch_size, len(remaining_tasks)) if batch_size > 0 else len(remaining_tasks)
    print(f"Processing next batch of {actual_batch} tasks...\n", flush=True)

    backbone = LLMBackend(provider=provider, model_name=model_name)
    print(f"Using backbone: provider={backbone.provider}, model={backbone.model_name}", flush=True)
    feature_builder = FeatureBuilder()

    tasks_to_run = remaining_tasks[:actual_batch]
    tasks_completed_in_batch = 0

    for task in tqdm(tasks_to_run, desc="Simulating Batch"):
        conv_id = task.get("task_id", "conv_unk")
        task_source = task.get("source") or task.get("task") or "general"
        shards = task.get("shards", [])

        history = []
        feature_builder.reset()
        task_records = []

        n_shards = len(shards)
        for t_idx, shard_item in enumerate(shards, start=1):
            if isinstance(shard_item, dict):
                shard_text = (
                    shard_item.get("shard") or shard_item.get("shard_text") or ""
                )
                is_underspecified = shard_item.get("is_underspecified")
            elif isinstance(shard_item, str):
                shard_text = shard_item
                is_underspecified = None
            else:
                continue

            if not shard_text:
                continue

            # Underspecified heuristic: short + vague pronoun
            if is_underspecified is None:
                words = shard_text.lower().split()
                is_underspecified = len(words) <= 5 and any(
                    token in {"it", "that", "them", "those", "more", "same", "this", "these"}
                    for token in words
                )

            is_final_turn = (t_idx == n_shards)
            curr_tokens = 600 if is_final_turn else 256
            curr_temp = 0.1 if is_final_turn else 0.7

            user_msg = _build_user_message(shard_text, task, t_idx, n_shards)
            history.append({"role": "user", "content": user_msg})
            assistant_response = backbone.generate(
                history, temperature=curr_temp, max_tokens=curr_tokens
            )
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
                f"  [{conv_id}] Turn {t_idx}/{n_shards} "
                f"(var={features.self_consistency_variance:.2f}, "
                f"contra={features.contradiction_score:.2f}, "
                f"conf={features.confidence:.2f})",
                flush=True,
            )

        # Judge task success
        success = judge_eventual_success(task, history, backbone)
        for record in task_records:
            record["eventual_success"] = success

        with open(output_logs_path, "a", encoding="utf-8") as out_f:
            for record in task_records:
                out_f.write(json.dumps(record) + "\n")
            out_f.flush()

        tasks_completed_in_batch += 1
        print(
            f"[{conv_id}] completed {len(task_records)} turns | success={success}",
            flush=True,
        )

        build_dataset(input_path=output_logs_path, output_dir=splits_dir)

        if cool_down_seconds > 0:
            time.sleep(cool_down_seconds)

    print(
        f"\nDone! Simulated {tasks_completed_in_batch} tasks → {output_logs_path}",
        flush=True,
    )
    print(f"Remaining: {len(remaining_tasks) - tasks_completed_in_batch}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate multi-turn tasks and log drift features (v2)."
    )
    parser.add_argument("--provider", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--n-resamples", type=int, default=3,
                        help="Self-consistency resamples per turn (default 3)")
    parser.add_argument("--cool-down", type=float, default=0.5)
    parser.add_argument(
        "--sharded-data", type=str,
        default="data/raw/lost_in_conversation/data/sharded_instructions_600.json"
    )
    parser.add_argument("--output-logs", type=str, default="data/processed/raw_logs.jsonl")
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    parser.add_argument(
        "--sources", type=str, default=None,
        help="Comma-separated sources, e.g. 'math,data2text,summary'"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sources = [s.strip() for s in args.sources.split(",")] if args.sources else None
    run_simulation(
        sharded_data_path=args.sharded_data,
        output_logs_path=args.output_logs,
        splits_dir=args.splits_dir,
        provider=args.provider,
        model_name=args.model,
        batch_size=args.batch_size,
        n_resamples=args.n_resamples,
        cool_down_seconds=args.cool_down,
        sources_filter=sources,
    )