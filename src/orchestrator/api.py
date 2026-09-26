"""
Section 10.1 — Backend API (FastAPI)

Endpoints the split-screen frontend (Section 10.2) will call:
  POST /session            create a new paired baseline/drift-aware session
  POST /session/{id}/turn  submit the next (possibly ambiguity-injected) instruction
  GET  /session/{id}/log   full turn/feature/score/decision log for replay
"""

from fastapi import FastAPI

app = FastAPI(title="Drift-Aware Agents Demo API")


@app.post("/session")
def create_session():
    raise NotImplementedError


@app.post("/session/{session_id}/turn")
def submit_turn(session_id: str, instruction: str):
    raise NotImplementedError


@app.get("/session/{session_id}/log")
def get_log(session_id: str):
    raise NotImplementedError
