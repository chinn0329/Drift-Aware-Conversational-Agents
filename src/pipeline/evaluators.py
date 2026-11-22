"""Executable evaluators. LiveCodeBench only for now."""
import ast
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
import base64, zlib, pickle

HEADER = (
    "from typing import *\n"
    "import math, collections, heapq, bisect, itertools, functools, re, string, sys\n"
    "from collections import *\n"
    "from itertools import *\n"
    "from functools import *\n"
    "from heapq import *\n"
    "from bisect import *\n"
)
MARK = "@@RESULT@@"

# Appended after the model's code for 'functional' (LeetCode-style) tests.
FUNC_TAIL = '''
import json as _json, sys as _sys, inspect as _inspect
_fn = %r
_args = [_json.loads(l) for l in _sys.stdin.read().split("\\n") if l.strip()]
def _pick():
    if "Solution" in globals():
        _s = Solution()
        if hasattr(_s, _fn):
            return getattr(_s, _fn)
        _names = [n for n, v in vars(Solution).items()
                  if _inspect.isfunction(v) and not n.startswith("_")]
        if _names:
            return getattr(_s, _names[0])
    if _fn in globals():
        return globals()[_fn]
    _fs = [v for v in globals().values() if _inspect.isfunction(v)
           and v.__module__ == "__main__" and not v.__name__.startswith("_")]
    return _fs[-1] if _fs else None
_f = _pick()
if _f is None:
    _sys.exit(3)
print("''' + MARK + '''" + _json.dumps(_f(*_args)))
'''


def last_code_block(text):
    blocks = re.findall(r"```(?:[A-Za-z0-9_+-]*)\n(.*?)```", text, re.S)
    return blocks[-1].strip() if blocks else None


def _decode_private(raw):
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        pass
    try:  # pickle.loads runs code embedded in the data: only use on the trusted benchmark file
        data = pickle.loads(zlib.decompress(base64.b64decode(raw.encode("utf-8"))))
        return json.loads(data) if isinstance(data, (str, bytes)) else data
    except Exception:
        return None  # present but undecodable


def _load_tests(task, max_private=15):
    pub = task.get("public_test_cases")
    if isinstance(pub, str):
        try:
            pub = json.loads(pub)
        except Exception:
            pub = []
    priv = _decode_private(task.get("private_test_cases"))
    if priv is None:
        return None
    return (pub or []) + priv[:max_private]


def _func_name(task):
    meta = task.get("metadata")
    if isinstance(meta, str):
        try:
            meta = ast.literal_eval(meta)
        except Exception:
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
    return (meta or {}).get("func_name")


def _run(source, stdin, timeout=6):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "sol.py"
        p.write_text(source, encoding="utf-8")
        try:
            r = subprocess.run([sys.executable, str(p)], input=stdin, text=True,
                               capture_output=True, timeout=timeout, cwd=d)
        except subprocess.TimeoutExpired:
            return None
        return r.returncode, r.stdout, r.stderr


def _same_json(a, b):
    try:
        return json.loads(a) == json.loads(b)
    except Exception:
        return str(a).strip() == str(b).strip()


def eval_lcb(task, final_text):
    """Returns (success, label_valid, reason)."""
    if final_text.count("```") % 2 == 1:
        return False, False, "truncated_code_block"
    code = last_code_block(final_text)
    if code is None:
        if "def " in final_text:
            code = final_text
        else:
            return False, True, "no_code"

    tests = _load_tests(task)
    if tests is None:
        return False, False, "private_tests_undecodable"
    if not tests:
        return False, False, "no_tests"
    fn = _func_name(task)

    for i, t in enumerate(tests):
        if t.get("testtype") == "functional":
            if not fn:
                return False, False, "no_func_name"
            res = _run(HEADER + code + "\n" + FUNC_TAIL % fn, t["input"])
            if res is None:
                return False, True, f"timeout_t{i}"
            rc, out, _ = res
            line = next((l for l in out.splitlines() if l.startswith(MARK)), None)
            if rc != 0 or line is None:
                return False, True, f"runtime_error_t{i}"
            ok = _same_json(line[len(MARK):], t["output"])
        else:  # stdin / stdout
            res = _run(HEADER + code, t["input"])
            if res is None:
                return False, True, f"timeout_t{i}"
            rc, out, _ = res
            ok = rc == 0 and out.split() == t["output"].split()
        if not ok:
            return False, True, f"wrong_answer_t{i}"
    return True, True, "ok"

# ---------------- BFCL ("actions") ----------------
def _as_obj(v):
    if isinstance(v, str):
        try:
            return ast.literal_eval(v)
        except Exception:
            try:
                return json.loads(v)
            except Exception:
                return None
    return v


def _norm_val(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        return re.sub(r"\s+", " ", v.strip().strip("'\"").lower())
    if isinstance(v, (list, tuple)):
        return [_norm_val(x) for x in v]
    if isinstance(v, dict):
        return {str(k).lower(): _norm_val(x) for k, x in v.items()}
    return v

def _calls_from_node(node, order):
    nodes = node.elts if isinstance(node, (ast.List, ast.Tuple)) else [node]
    calls = []
    for n in nodes:
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            try:                                  # a call written as a quoted string
                n = ast.parse(n.value.strip(), mode="eval").body
            except SyntaxError:
                return None
        if not isinstance(n, ast.Call):
            return None
        name = ast.unparse(n.func)
        try:
            kw = {}
            names = order.get(name, [])
            for i, a in enumerate(n.args):
                if i >= len(names):
                    return None
                kw[names[i]] = ast.literal_eval(a)
            for k in n.keywords:
                if k.arg is None:
                    return None
                kw[k.arg] = ast.literal_eval(k.value)
        except Exception:
            return None
        calls.append((name, kw))
    return calls


def parse_calls(text, order=None):
    order = order or {}
    src = last_code_block(text) or text.strip()
    candidates = [src]
    if not src.lstrip().startswith("["):
        candidates.append("[" + ",".join(l.strip() for l in src.splitlines() if l.strip()) + "]")
    for c in candidates:
        try:
            node = ast.parse(c, mode="eval").body
        except SyntaxError:
            continue
        calls = _calls_from_node(node, order)
        if calls is not None:
            return calls
    try:                                              # JSON fallback
        data = json.loads(src)
        data = data if isinstance(data, list) else [data]
        calls = []
        for d in data:
            args = d.get("arguments", d.get("parameters", {}))
            if isinstance(args, str):
                args = json.loads(args)
            calls.append((d["name"], args))
        return calls
    except Exception:
        return None


def _call_matches(pname, pkw, gname, gparams):
    if pname.replace(".", "_") != gname.replace(".", "_"):
        return False
    if not set(pkw) <= set(gparams):                  # unexpected parameter
        return False
    for p, accepted in gparams.items():
        acc = accepted if isinstance(accepted, list) else [accepted]
        if p not in pkw:
            if "" in acc:                             # optional parameter
                continue
            return False
        v = _norm_val(pkw[p])
        if not any(v == _norm_val(a) for a in acc if a != ""):
            return False
    return True


def eval_bfcl(task, final_text):
    """Returns (success, label_valid, reason)."""
    if final_text.count("```") % 2 == 1:
        return False, False, "truncated_code_block"
    if last_code_block(final_text) is None and re.search(
        r"(can[’']t|cannot|unable to)\s+(help|assist|comply)|i[’']m sorry", final_text, re.I
    ):
        return False, True, "refusal"
    gt = _as_obj(task.get("reference_answer"))
    if not isinstance(gt, list) or not gt:
        return False, False, "no_reference"
    funcs = _as_obj(task.get("function")) or []
    order = {f["name"]: list((f.get("parameters") or {}).get("properties", {}).keys())
             for f in funcs if isinstance(f, dict) and "name" in f}
    pred = parse_calls(final_text, order)
    if pred is None:
        return False, True, "unparseable"
    if len(pred) != len(gt):
        return False, True, "wrong_call_count"
    used = set()
    for g in gt:
        (gname, gparams), = g.items()
        for i, (pn, pkw) in enumerate(pred):
            if i not in used and _call_matches(pn, pkw, gname, gparams):
                used.add(i)
                break
        else:
            return False, True, "no_match"
    return True, True, "ok"

def evaluate(task, final_text, finish_reason):
    """Returns (success, label_valid, reason). success=None -> caller uses LLM judge."""
    if finish_reason == "length":
        return False, False, "truncated"
    if str(task.get("source", "")).lower().startswith("lcb"):
        return eval_lcb(task, final_text)
    if task.get("task") == "actions":
        return eval_bfcl(task, final_text)
    return None, True, "no_executable_check"