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

def evaluate(task, final_text, finish_reason):
    """Returns (success, label_valid, reason). success=None -> caller uses LLM judge."""
    if finish_reason == "length":
        return False, False, "truncated"
    if str(task.get("source", "")).lower().startswith("lcb"):
        return eval_lcb(task, final_text)
    return None, True, "no_executable_check"