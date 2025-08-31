# SBPL Codingt — LLM‑based Multi‑API Planning (Travel/Tool Agent)

This repo contains a lightweight, hackable planning agent with an interactive HTML visualizer. 
It’s designed for experiments comparing **with/without prompt hint** settings, A*‑style planning over tool calls, 
and traceable execution logs.

> Files referenced by the project (as provided):
>
> - `agent_execute.py` — main runner that orchestrates planning + tool execution and saves an HTML trace.
> - `execute.py` — convenience CLI to run a single question with default settings.
> - `planner.py` — search/planning logic (CBS‑style/A* style hooks).
> - `planner_viz.py` — wrapper around the visualizer/logger utilities.
> - `visualizer.py` — builds interactive HTML (plan steps, tool calls, outcomes).
> - `backend.py` — tool/back‑end call abstractions (HTTP/LLM/etc.).
> - `tool.py` — tool interface & base definitions.
> - `tool_list.py` — registry of concrete tools available to the agent.
> - `state.py` — typed state container for the planner/executor.
> - `prompt.py` — prompt templates and few‑shot examples (hint/no‑hint variants).
> - `eval_prompt_hint.py` — batch evaluator to compare **prompt_hint** vs **no_hint** runs.
> - `plans_export.csv` / `plans_export.jsonl` — saved plans/traces for analysis.
> - `plan_report.html` — prebuilt (or generated) report of a run.
>
> Some filenames may be optional in your local copy — this README covers how they are intended to work together.

---

## Quick start

### 1) Environment

- Python **3.10+** recommended
- Install deps:
  ```bash
  python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
  pip install -r requirements.txt
  ```

### 2) Configure API keys (if you use the LLM tools)

If your tools call models via OpenAI (or other providers), set environment variables (e.g., in a `.env` file):
```
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=
OPENAI_MODEL=gpt-4o-mini
```
You can omit or customize these based on `backend.py`/`tool_list.py` in your copy.

### 3) Run a single question

```bash
# Minimal runner
python execute.py   --question "I want to escape the summer heat from New York and head somewhere cool in Northern Europe, preferably by plane."   --out out/trace.html
```

This should produce an interactive HTML trace (e.g., `out/trace.html`) that you can open in a browser.

### 4) Use the planner visualizer

```bash
python planner_viz.py --logdir out/
```
This collects recorded steps and renders a richer dashboard using `visualizer.py`.

### 5) Batch evaluation: prompt_hint vs no_hint

`eval_prompt_hint.py` supports running your benchmark twice (with and without hint prompts) and summarizing metrics:

```bash
python eval_prompt_hint.py   --bench data/bench.txt   --with_hint   --without_hint   --out out/eval_report.html
```

Typical metrics you may track (and that the scripts can log):
- **Destination Accuracy** — Did the final destination meet the user’s location constraint?  
- **Instruction Compliance** — Were transport/seasonality/time constraints respected?  
- **Plan Coherence** — Tool sequence sanity (e.g., `set_origin → set_destination → select_transport → confirm_booking`) and absence of extraneous calls.

> Tip: If you don’t have a bench file, start from a Python list in `eval_prompt_hint.py` and iterate.

---

## Project structure (intended)

```
.
├── agent_execute.py        # Orchestrates plan + tool execution; writes HTML trace
├── backend.py              # Back‑end integrations (LLM calls, HTTP tool calls, etc.)
├── execute.py              # Thin CLI wrapper around agent_execute
├── planner.py              # Planning/search logic (A* hooks, heuristics)
├── planner_viz.py          # Visualizer entrypoint
├── prompt.py               # Prompt templates (hint/no‑hint) + few‑shots
├── state.py                # Shared, typed state for planner/executor
├── tool.py                 # Tool interface and base classes
├── tool_list.py            # Tool registry
├── visualizer.py           # HTML generation for traces/step logs
├── eval_prompt_hint.py     # Batch evaluator (with vs without hint)
├── plans_export.csv        # (Optional) saved plans/traces
├── plans_export.jsonl      # (Optional) saved plans/traces in JSONL
└── plan_report.html        # (Optional) generated HTML report
```

---

## Common commands

```bash
# 1) Lint/type‑check (optional)
pip install ruff mypy
ruff check .
mypy . --ignore-missing-imports

# 2) Run a demo question
python execute.py --question "From Seattle to Vancouver by bus?" --out out/seattle_van.html

# 3) Open a generated report
python -m webbrowser out/seattle_van.html
```

---

## Troubleshooting

- **Missing module errors**  
  If you hit `ModuleNotFoundError: openai` (or similar), confirm you’ve installed `requirements.txt` in the active venv.
- **HTML template variables**  
  If you see `NameError: name 'status' is not defined` or `NameError: name 'toolsHtml' is not defined` in HTML building code, verify the f‑string/template variables in `visualizer.py` or related functions (`planner_viz.py`, `agent_execute.py`) are properly defined before formatting.
- **API limits/keys**  
  Ensure `OPENAI_API_KEY` is set; adjust model name/base URL in `backend.py`.

---

## Contributing

- Keep tools small and well‑typed.
- Prefer explicit logs visible in the HTML trace for each step (inputs/outputs/status).
- When adding a new tool:
  1. Implement it in `tool.py` (interface) and register in `tool_list.py`.
  2. Add a short example to `prompt.py` few‑shots if relevant.
  3. Update `README.md` if new env vars are required.

---

## License

MIT
