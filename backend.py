# backend.py
from __future__ import annotations

import json
import time
from typing import AsyncGenerator, Dict, List, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from state import State
from tool import Tool
from execute import Executor
from planner import planner

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}


def sse_event(event: str, data: Dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def execute_generator(question: str) -> AsyncGenerator[bytes, None]:
    p = planner()
    plan: List[Tuple[Tool, Tuple[str, ...]]] = p.astar(State(), question)

    plan_json = [
        {"step": i + 1, "tool": t.name, "args": list(args)}
        for i, (t, args) in enumerate(plan)
    ]

    yield sse_event("planner", {"question": question, "steps": plan_json}).encode("utf-8")

    exe = Executor(State())
    total = len(plan)

    for i, (tool, args) in enumerate(plan, start=1):
        exe.run_step(i, tool, args)
        snapshot = exe.history[-1]

        payload = {
            "index": i,
            "total": total,
            "tool": tool.name,
            "args": list(args),
            "state": snapshot,
            "progress": round(i / total, 4) if total else 1.0,
        }
        yield sse_event("step", payload).encode("utf-8")

        time.sleep(3)

    yield sse_event("done", {"total": total, "final_state": snapshot}).encode("utf-8")


@app.get("/execute_stream")
async def execute_stream(question: str):
    return StreamingResponse(execute_generator(question), media_type="text/event-stream")