# visualizer.py
import json
from typing import List, Tuple


class VisualLogger:
    """Interactive planner report with per-step *multiple tools* visualization."""

    def __init__(self):
        self.steps = []

    def begin_step(self, step: int):
        self.steps.append({
            "step": step,
            "tools": [],
            "chosen": None,
            "executed": False,
        })

    def log_tool(self, step: int, tool_name: str, tool_prior: float,
                 arg_candidates: List[Tuple[str, ...]], probs: List[float]):
        s = self.steps[step]
        s["tools"].append({
            "tool": tool_name,
            "prior": float(tool_prior),
            "candidates": [
                {"args": list(c), "p": float(p)}
                for c, p in zip(arg_candidates, probs)
            ]
        })

    def set_choice(self, step: int, tool_name: str, chosen_args: Tuple[str, ...],
                   pred_prob: float, combined_p: float):
        s = self.steps[step]
        s["chosen"] = {
            "tool": tool_name,
            "args": list(chosen_args),
            "pred_prob": float(pred_prob),
            "combined_p": float(combined_p),
        }

    def mark_executed(self, step: int):
        self.steps[step]["executed"] = True

    def generate_html(self, path: str = "plan_report.html"):
        row_html_list = []
        for s in self.steps:
            considered = ", ".join([t["tool"] for t in s["tools"]]) or "-"
            chosen_tool = s["chosen"]["tool"] if s["chosen"] else "-"
            chosen_args = s["chosen"]["args"] if s["chosen"] else []
            pred_prob   = f'{s["chosen"]["pred_prob"]:.2f}' if s["chosen"] else "-"
            combined_p  = f'{s["chosen"]["combined_p"]:.2f}' if s["chosen"] else "-"
            executed_badge = "✅" if s["executed"] else ""
            row_html_list.append(
                f"<tr style=\"display:none\">\n"
                f"  <td>{s['step']}</td>\n"
                f"  <td>{considered}</td>\n"
                f"  <td><b>{chosen_tool}</b> {tuple(chosen_args)}</td>\n"
                f"  <td>{pred_prob}</td>\n"
                f"  <td>{combined_p}</td>\n"
                f"  <td>{executed_badge}</td>\n"
                f"</tr>"
            )
        rows_html = "\n".join(row_html_list)

        steps_json = json.dumps(self.steps, ensure_ascii=False)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Planner Execution Report</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto; margin: 24px; color: #0b1020; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #b7c1d6; padding: 8px 10px; text-align: left; font-size: 14px; }}
    th {{ background: #eef3ff; }}
    tr:nth-child(even) td {{ background: #fafcff; }}
    .controls {{ margin-bottom: 12px; display:flex; gap:12px; align-items:center; }}
    input[type=range] {{ width: 220px; }}
    .panel h3 {{ margin:6px 0; }}
    .tool-card {{ border:1px solid #dbe3f3; border-radius:12px; padding:10px; margin:8px 0; background:#f7faff; }}
    .muted {{ color:#6b7485; }}
    .pill {{ display:inline-block; padding:2px 8px; border-radius:999px; background:#eef3ff; border:1px solid #dbe3f3; margin-left:6px; font-size:12px; }}
    code {{ background:#f3f6ff; padding:1px 6px; border-radius:8px; }}
  </style>
</head>
<body>
  <h2>A* Tool-Chain (per-step candidates → chosen)</h2>

  <div class="controls">
    <button id="playBtn">▶ Play</button>
    <label>Step: <span id="stepLabel">0/{len(self.steps)-1}</span></label>
    <input type="range" id="scrubber" min="0" max="{len(self.steps)-1}" value="0">
  </div>

  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>Considered tools</th>
        <th>Chosen (tool & args)</th>
        <th>Pred p</th>
        <th>Combined p</th>
        <th>Exec</th>
      </tr>
    </thead>
    <tbody id="tbody">
      {rows_html}
    </tbody>
  </table>

  <div class="panel" id="panel"></div>

<script>
  const rows = Array.from(document.querySelectorAll('#tbody tr'));
  const stepLabel = document.getElementById('stepLabel');
  const scrubber  = document.getElementById('scrubber');
  const panel = document.getElementById('panel');
  const steps = {steps_json};

  let current = -1;

  function renderPanel(n) {{
    const s = steps[n];
    stepLabel.textContent = n + '/' + (steps.length - 1);

    // Build "considered tools" section
    let toolsHtml = '';
    for (const t of s.tools) {{
      const candList = t.candidates.map(c => `
        <li><code>${{JSON.stringify(c.args)}}</code> — p=${{c.p.toFixed(2)}}</li>
      `).join('');
      toolsHtml += `
        <div class="tool-card">
          <div><b>${{t.tool}}</b><span class="pill">prior=${{t.prior.toFixed(2)}}</span></div>
          <div class="muted" style="margin-top:4px;">Candidates:</div>
          <ul style="margin:6px 0 0 18px;">${{candList || '<li class="muted">None</li>'}}</ul>
        </div>
      `;
    }}

    const chosen = s.chosen ? `
      <div class="tool-card" style="border-color:#b5e0c6; background:#f2fbf6;">
        <div><b>Chosen:</b> ${{s.chosen.tool}} <code>${{JSON.stringify(s.chosen.args)}}</code></div>
        <div class="muted" style="margin-top:4px;">
          pred_p=${{Number(s.chosen.pred_prob).toFixed(2)}}, combined_p=${{Number(s.chosen.combined_p).toFixed(2)}}
        </div>
      </div>
    ` : '<div class="muted">No choice recorded.</div>';

    panel.innerHTML = `
      <h3>Step ${{s.step}}</h3>
      <div class="muted">All tools considered this step, with their priors and candidate args:</div>
      ${{toolsHtml}}
      ${{chosen}}
    `;
  }}

  function showStep(n) {{
    rows.forEach((tr, idx) => {{ tr.style.display = idx <= n ? '' : 'none'; }});
    current = n;
    scrubber.value = n;
    renderPanel(n);
  }}

  scrubber.addEventListener('input', (e) => {{
    showStep(parseInt(e.target.value));
  }});

  document.getElementById('playBtn').addEventListener('click', () => {{
    const btn = event.target;
    if (btn.dataset.playing === 'true') {{
      btn.dataset.playing = 'false';
      btn.textContent = '▶ Play';
      return;
    }}
    btn.dataset.playing = 'true';
    btn.textContent = '⏸ Pause';
    const interval = setInterval(() => {{
      if (btn.dataset.playing !== 'true') {{ clearInterval(interval); return; }}
      if (current >= rows.length - 1) {{
        btn.dataset.playing = 'false';
        btn.textContent = '▶ Play';
        clearInterval(interval);
        return;
      }}
      showStep(current + 1);
    }}, 1000);
  }});

  // init
  showStep(0);
</script>
</body>
</html>"""
        with open(path, "w", encoding="utf-8") as fp:
            fp.write(html)
        print(f"[VisualLogger] Interactive report saved to {path}")
