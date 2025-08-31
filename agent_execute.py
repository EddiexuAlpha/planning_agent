from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Tuple, Optional, Protocol, runtime_checkable

from state import State
from tool import Tool
from tool_list import TOOLS
from planner_viz import Planner


def state_to_dict(s: State) -> Dict[str, Any]:
    """Convert your State dataclass to a JSON-serializable dict."""
    try:
        return asdict(s)  # works if State is a dataclass
    except Exception:
        # Fallback for custom classes
        return {
            k: getattr(s, k)
            for k in dir(s)
            if not k.startswith("_") and not callable(getattr(s, k))
        }

def dict_diff(prev: Dict[str, Any], curr: Dict[str, Any]) -> Dict[str, Any]:
    """Key-wise diff for observations."""
    diff = {}
    for k in sorted(set(prev.keys()) | set(curr.keys())):
        if prev.get(k) != curr.get(k):
            diff[k] = {"before": prev.get(k), "after": curr.get(k)}
    return diff


# =========================
# 2) LLM interface & impls
# =========================

@runtime_checkable
class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class DummyLLM:
    """
    Offline fallback:
    - 有 hint：回显 PLANNED_ARGS
    - 无 hint：从 AVAILABLE TOOLS 里挑第一个可用工具，给朴素默认 args
    """
    def complete(self, system: str, user: str) -> str:
        try:
            marker = "PLANNED_ARGS:"
            if marker in user:
                js = user.split(marker, 1)[1].strip()
                planned = json.loads(js.splitlines()[0])
                thought = "Using the planner hint. Args look consistent with current state."
                return json.dumps({"thought": thought, "args": planned}, ensure_ascii=False)
        except Exception:
            pass

        thought = "No hint mode: choose the first applicable tool with naive default args."
        tool_name = "STOP"
        args = []

        try:
            if "AVAILABLE TOOLS:" in user:
                tools_block = user.split("AVAILABLE TOOLS:", 1)[1]
                start = tools_block.find("[")
                end = tools_block.rfind("]") + 1
                tools = json.loads(tools_block[start:end])
            else:
                tools = []
        except Exception:
            tools = []

        for t in tools:
            tn = t.get("name")
            arg_names = t.get("arg_names", [])
            if tn == "set_origin":
                tool_name = tn; args = ["New York"][:len(arg_names)]; break
            if tn == "set_destination":
                tool_name = tn; args = ["Boston"][:len(arg_names)]; break
            if tn == "select_transport":
                tool_name = tn; args = ["flight"][:len(arg_names)]; break
            if tn == "confirm_booking":
                tool_name = tn; args = []
                break

        if tool_name == "STOP":
            return json.dumps({"thought": "No applicable tools.", "tool": "STOP", "args": []}, ensure_ascii=False)

        return json.dumps({"thought": thought, "tool": tool_name, "args": args}, ensure_ascii=False)


class OpenAIChatLLM:
    """
    Uses OpenAI Chat Completions (>=1.0 client style).
    Set env: OPENAI_API_KEY, optionally OPENAI_BASE_URL, OPENAI_MODEL.
    """
    def __init__(self, model: Optional[str] = None):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
        )
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def complete(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
        )
        txt = resp.choices[0].message.content.strip()
        return txt


# =========================
# 3) Execution Prompts (few-shot)
# =========================

AGENT_SYSTEM_PROMPT = """You are an execution agent.
You will be given:
- Either the overall PLAN (a sequence of tool calls) as a hint, OR no plan at all.
- The current step's tool schema and current state.
Your job: for EACH step, think briefly then return STRICT JSON:
  {"thought": "...", "args": ["...", "..."]}

Rules:
- If PLAN is provided, follow it unless it obviously conflicts with the current state.
- If PLAN is omitted, decide the next args yourself from the state + user request.
- Only output JSON, no backticks, no commentary outside JSON.
- Ensure args match the tool's arg schema exactly (both count and type).
- If you must adjust from PLAN, explain why in 'thought' then give final 'args'.

---

### Few-shot examples:

Example 1:
USER QUESTION: Book a train from New York to Boston
CURRENT STEP: set_origin (arg_names=["city"])
STATE: {"origin": null, "destination": null, "transport": null, "booking_confirmed": false}
→ {"thought": "Origin is not set, set it to New York", "args": ["New York"]}

Example 2:
USER QUESTION: Book a train from New York to Boston
CURRENT STEP: set_destination (arg_names=["city"])
STATE: {"origin": "New York", "destination": null, "transport": null, "booking_confirmed": false}
→ {"thought": "Destination missing, set it to Boston", "args": ["Boston"]}

Example 3:
USER QUESTION: Book a train from New York to Boston
CURRENT STEP: select_transport (arg_names=["mode"])
STATE: {"origin": "New York", "destination": "Boston", "transport": null, "booking_confirmed": false}
→ {"thought": "User explicitly asked for train, choose train", "args": ["train"]}

Example 4:
USER QUESTION: Book a train from New York to Boston
CURRENT STEP: confirm_booking (arg_names=[])
STATE: {"origin": "New York", "destination": "Boston", "transport": "train", "booking_confirmed": false}
→ {"thought": "All info filled, confirm the booking", "args": []}

---
Now perform the same reasoning for the actual task.
"""

STEP_USER_TEMPLATE = """USER QUESTION:
{user_question}

PLAN (HINT):
{plan_hint}

CURRENT STEP:
- Step index: {step_idx}/{total_steps}
- Tool: {tool_name}
- Tool description: {tool_desc}
- Arg names: {arg_names}

CURRENT STATE SNAPSHOT:
{state_json}

PLANNED_ARGS: {planned_args_json}

Now produce ONLY JSON: {{"thought": "...", "args": [...]}}"""

NOHINT_SYSTEM_PROMPT = """You are an execution agent.
You will be given the CURRENT STATE and a list of AVAILABLE TOOLS (schema only).
Your job: at EACH step, think briefly then return STRICT JSON:
{"thought": "...", "tool": "<tool_name>", "args": ["...", "..."]}

Rules:
- Pick exactly ONE tool from the provided list whose precondition likely holds.
- Ensure args match the tool's arg schema exactly (both count and type).
- If no tool is applicable, explain in 'thought' and pick "STOP" with empty args.

Output ONLY JSON. No backticks. No commentary outside JSON.

### Few-shot mini:

User: Book a train from New York to Boston
State: {"origin": null, "destination": null, "transport": null, "booking_confirmed": false}
Available: [{"name":"set_origin","arg_names":["city"]}, ...]
→ {"thought":"Set origin first","tool":"set_origin","args":["New York"]}

User: (next)
State: {"origin": "New York", "destination": null, "transport": null, "booking_confirmed": false}
Available: [{"name":"set_destination","arg_names":["city"]}, ...]
→ {"thought":"Now set destination","tool":"set_destination","args":["Boston"]}

User: (next)
State: {"origin":"New York","destination":"Boston","transport":null,"booking_confirmed":false}
Available: [{"name":"select_transport","arg_names":["mode"]}, ...]
→ {"thought":"User wants train","tool":"select_transport","args":["train"]}

User: (next)
State: {"origin":"New York","destination":"Boston","transport":"train","booking_confirmed":false}
Available: [{"name":"confirm_booking","arg_names":[]}, ...]
→ {"thought":"All set, confirm","tool":"confirm_booking","args":[]}
"""

NOHINT_USER_TEMPLATE = """USER QUESTION:
{user_question}

CURRENT STEP:
- Step index: {step_idx}/{max_steps}

CURRENT STATE SNAPSHOT:
{state_json}

AVAILABLE TOOLS:
{tools_json}

Now produce ONLY JSON: {{"thought": "...", "tool": "<tool_name or STOP>", "args": [...]}}"""


# =========================
# 4) Agent Executor
# =========================

class AgentExecutor:
    """
    Agent-driven executor: the LLM decides/justifies tool+args step-by-step.
    - use_plan_hint=True: 使用 Planner 产出的 plan，LLM 主要负责/修正 args
    - use_plan_hint=False: 不调用 Planner。LLM 自主选择 tool+args
    """
    def __init__(self, llm: LLM, use_plan_hint: bool = True, max_steps: int = 10):
        self.llm = llm
        self.use_plan_hint = use_plan_hint
        self.max_steps = max_steps
        self.trace: List[Dict[str, Any]] = []

    def run(self, user_question: str) -> List[Dict[str, Any]]:
        if self.use_plan_hint:
            p = Planner()
            plan = p.astar(State(), user_question)

            state = State()
            total_steps = len(plan)

            plan_hint_lines = [f"{i}. {tool.name}{args}" for i, (tool, args) in enumerate(plan, start=1)]
            plan_hint_text = "\n".join(plan_hint_lines)

            for step_idx, (tool, planned_args) in enumerate(plan, start=1):
                prev_state_dict = state_to_dict(state)

                user_block = STEP_USER_TEMPLATE.format(
                    user_question=user_question,
                    plan_hint=plan_hint_text,
                    step_idx=step_idx,
                    total_steps=total_steps,
                    tool_name=tool.name,
                    tool_desc=getattr(tool, "description", ""),
                    arg_names=getattr(tool, "arg_names", ()),
                    state_json=json.dumps(prev_state_dict, ensure_ascii=False, indent=2),
                    planned_args_json=json.dumps(list(planned_args), ensure_ascii=False),
                )

                raw = self.llm.complete(AGENT_SYSTEM_PROMPT, user_block)

                try:
                    data = json.loads(raw)
                    thought = str(data.get("thought", "")).strip()
                    args = data.get("args", [])
                    if not isinstance(args, list):
                        raise ValueError("args must be a JSON array")
                    args_tuple = tuple(str(x) for x in args)
                except Exception as e:
                    thought = f"Failed to parse JSON ({e}); falling back to planner args."
                    args_tuple = tuple(planned_args)

                ok = tool.precondition(state)
                error: Optional[str] = None
                if not ok:
                    error = "Precondition failed — tool skipped."
                else:
                    try:
                        tool(state, *args_tuple)
                    except Exception as e:
                        error = f"Tool raised: {e}"

                curr_state_dict = state_to_dict(state)
                obs = dict_diff(prev_state_dict, curr_state_dict)

                self.trace.append({
                    "step": step_idx,
                    "tool": tool.name,
                    "planned_args": list(planned_args),
                    "agent_thought": thought,
                    "agent_args": list(args_tuple),
                    "pre_ok": ok,
                    "error": error,
                    "observation": obs,
                    "state_after": curr_state_dict,
                })

            return self.trace

        state = State()
        step_idx = 0
        while step_idx < self.max_steps and not state.booking_confirmed:
            step_idx += 1
            prev_state_dict = state_to_dict(state)

            available = []
            for t in TOOLS:
                try:
                    if t.precondition(state):
                        available.append({
                            "name": t.name,
                            "description": getattr(t, "description", ""),
                            "arg_names": list(getattr(t, "arg_names", ())),
                        })
                except Exception:
                    continue

            if not available:
                self.trace.append({
                    "step": step_idx,
                    "tool": "STOP",
                    "planned_args": [],
                    "agent_thought": "No applicable tools remain.",
                    "agent_args": [],
                    "pre_ok": False,
                    "error": "No tool applicable",
                    "observation": {},
                    "state_after": prev_state_dict,
                })
                break

            user_block = NOHINT_USER_TEMPLATE.format(
                user_question=user_question,
                step_idx=step_idx,
                max_steps=self.max_steps,
                state_json=json.dumps(prev_state_dict, ensure_ascii=False, indent=2),
                tools_json=json.dumps(available, ensure_ascii=False, indent=2),
            )

            raw = self.llm.complete(NOHINT_SYSTEM_PROMPT, user_block)

            thought, chosen_tool_name, args_tuple, error = "", "STOP", tuple(), None
            try:
                data = json.loads(raw)
                thought = str(data.get("thought", "")).strip()
                chosen_tool_name = str(data.get("tool", "STOP"))
                args = data.get("args", [])
                if not isinstance(args, list):
                    raise ValueError("args must be a JSON array")
                args_tuple = tuple(str(x) for x in args)
            except Exception as e:
                error = f"Parse error: {e}"

            if chosen_tool_name.upper() == "STOP":
                self.trace.append({
                    "step": step_idx,
                    "tool": "STOP",
                    "planned_args": [],
                    "agent_thought": thought or "Agent chose to stop.",
                    "agent_args": [],
                    "pre_ok": False,
                    "error": error,
                    "observation": {},
                    "state_after": prev_state_dict,
                })
                break

            tool = next((t for t in TOOLS if t.name == chosen_tool_name), None)
            ok = bool(tool and tool.precondition(state))
            if not tool:
                error = f"Unknown tool: {chosen_tool_name}"
                ok = False
            elif not ok:
                error = "Precondition failed"

            if not error and tool:
                try:
                    tool(state, *args_tuple)
                except Exception as e:
                    error = f"Tool raised: {e}"

            curr_state_dict = state_to_dict(state)
            obs = dict_diff(prev_state_dict, curr_state_dict)

            self.trace.append({
                "step": step_idx,
                "tool": chosen_tool_name,
                "planned_args": [], 
                "agent_thought": thought,
                "agent_args": list(args_tuple),
                "pre_ok": ok,
                "error": error,
                "observation": obs,
                "state_after": curr_state_dict,
            })

            if state.booking_confirmed:
                break

        return self.trace


# =========================
# 5) Visual Logger (HTML)
# =========================

class VisualLogger:
    """
    Emit an interactive HTML that reveals steps progressively.
    Includes:
      - summary table
      - step-by-step panel: Thought / Action / Observation / State snapshot
      - Play/Pause button and slider
    """
    def __init__(self, title: str = "Agent Execution Trace"):
        self.title = title

    def to_html(self, steps: List[Dict[str, Any]]) -> str:
        table_rows = []
        for s in steps:
            table_rows.append(f"""
              <tr>
                <td>{s['step']}</td>
                <td>{s['tool']}</td>
                <td><code>{json.dumps(s.get('planned_args', []), ensure_ascii=False)}</code></td>
                <td><code>{json.dumps(s.get('agent_args', []), ensure_ascii=False)}</code></td>
                <td>{'✅' if s.get('pre_ok') and not s.get('error') else '⚠️'}</td>
              </tr>
            """)

        steps_json = json.dumps(steps, ensure_ascii=False)

        return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>{self.title}</title>
<style>
  :root {{
    --bg: #0b1020; --fg: #e7ecff; --muted:#9aa3b2; --card:#141a2f; --acc:#6ea8fe; --ok:#14b86a; --warn:#e5a50a;
  }}
  html,body {{ background:var(--bg); color:var(--fg); margin:0; font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; }}
  .wrap {{ max-width: 1080px; margin: 24px auto; padding: 0 16px; }}
  h1 {{ font-size: 24px; margin: 8px 0 16px; }}
  .card {{ background:var(--card); border-radius:16px; padding:16px; box-shadow: 0 8px 20px rgba(0,0,0,.3); }}
  table {{ width:100%; border-collapse: collapse; }}
  th, td {{ border-bottom: 1px solid #273251; padding: 8px 10px; text-align:left; font-size: 14px; }}
  th {{ color: var(--muted); font-weight:600; }}
  code {{ background:#0f1630; padding:2px 6px; border-radius:8px; }}
  .controls {{ display:flex; align-items:center; gap:12px; margin: 16px 0; }}
  .btn {{ background: var(--acc); color:#fff; border:none; padding:8px 14px; border-radius:12px; cursor:pointer; font-weight:600; }}
  .slider {{ width: 240px; }}
  .panel h3 {{ margin:6px 0 6px; font-size: 16px; color:#c9d4ff; }}
  .kv {{ display:grid; grid-template-columns: 160px 1fr; gap:10px; margin: 8px 0; }}
  .kv div:first-child {{ color: var(--muted); }}
  .statepre {{ white-space: pre-wrap; background:#0f1630; padding:12px; border-radius:12px; font-size:13px; }}
  .badge-ok {{ color: var(--ok); font-weight:700; }}
  .badge-warn {{ color: var(--warn); font-weight:700; }}
  .muted {{ color: var(--muted); }}
</style>
</head>
<body>
<div class="wrap">
  <h1>🔎 {self.title}</h1>

  <div class="card">
    <div class="controls">
      <button id="play" class="btn">▶ Play</button>
      <input id="scrub" type="range" min="1" max="{len(steps)}" step="1" value="1" class="slider" />
      <div id="label" class="muted">Step 1 / {len(steps)}</div>
    </div>

    <div style="overflow:auto;">
      <table>
        <thead>
          <tr><th>Step</th><th>Tool</th><th>Planned args</th><th>Agent args</th><th>Status</th></tr>
        </thead>
        <tbody>
          {''.join(table_rows)}
        </tbody>
      </table>
    </div>
  </div>

  <div class="card" style="margin-top:16px;">
    <div id="panel" class="panel"></div>
  </div>
</div>

<script>
const steps = {steps_json};

const panel = document.getElementById('panel');
const playBtn = document.getElementById('play');
const scrub = document.getElementById('scrub');
const label = document.getElementById('label');

let playing = false;
let timer = null;

function render(n) {{
  const s = steps[n-1];
  label.textContent = `Step ${{n}} / ${{steps.length}}`;

  const status = (!s.error && s.pre_ok)
    ? '<span class="badge-ok">OK</span>'
    : `<span class="badge-warn">WARN</span>`;

  panel.innerHTML = `
    <h3>Step ${{s.step}}: ${{s.tool}} ${'{'}{{status}}{'}'}</h3>
    <div class="kv"><div>Planned args</div><div><code>${'{'}JSON.stringify(s.planned_args){'}'}</code></div></div>
    <div class="kv"><div>Agent thought</div><div>${'{'}escapeHtml(s.agent_thought || ''){'}'}</div></div>
    <div class="kv"><div>Agent args</div><div><code>${'{'}JSON.stringify(s.agent_args){'}'}</code></div></div>
    <div class="kv"><div>Error</div><div>${'{'}escapeHtml(s.error || ''){'}'}</div></div>
    <div class="kv"><div>Observation</div><div><code>${'{'}JSON.stringify(s.observation){'}'}</code></div></div>
    <div class="kv"><div>State after</div><div class="statepre">${'{'}escapeHtml(JSON.stringify(s.state_after, null, 2)){'}'}</div></div>
  `;
}}

function escapeHtml(str) {{
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}}

function next() {{
  const v = parseInt(scrub.value, 10);
  if (v < steps.length) {{
    scrub.value = String(v + 1);
    render(v + 1);
  }} else {{
    playing = false;
    playBtn.textContent = "▶ Play";
    clearInterval(timer);
  }}
}}

playBtn.addEventListener('click', () => {{
  playing = !playing;
  playBtn.textContent = playing ? "⏸ Pause" : "▶ Play";
  if (playing) {{
    timer = setInterval(next, 1000);
  }} else {{
    clearInterval(timer);
  }}
}});

scrub.addEventListener('input', () => {{
  render(parseInt(scrub.value, 10));
}});

render(1);
</script>

</body>
</html>"""

    def save(self, steps: List[Dict[str, Any]], outfile: str = "agent_run.html") -> str:
        html = self.to_html(steps)
        with open(outfile, "w", encoding="utf-8") as f:
            f.write(html)
        return outfile


# =========================
# 6) High-level helper
# =========================

def run_agent_on_question(
    user_question: str,
    use_openai: bool = False,
    use_plan_hint: bool = True,
    out_html: str = "agent_run.html",
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Returns (trace, html_path)
    - use_openai=False -> DummyLLM (no API required)
    - use_plan_hint: whether planner's plan is given as a hint to the LLM
    """
    llm: LLM = OpenAIChatLLM() if use_openai else DummyLLM()
    agent = AgentExecutor(llm=llm, use_plan_hint=use_plan_hint)
    trace = agent.run(user_question)

    viz = VisualLogger("Agent Execution Trace")
    html_path = viz.save(trace, out_html)
    return trace, html_path


if __name__ == "__main__":
    q = "Book a flight from New York to someplace cold in Europe"
    trace, path = run_agent_on_question(q, use_openai=True, use_plan_hint=False)
    print(f"Steps: {len(trace)}; HTML written to: {path}")
