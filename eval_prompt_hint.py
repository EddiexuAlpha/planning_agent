import json
import csv
import os
import statistics as stats
from typing import Any, Dict, List, Tuple

from agent_execute import run_agent_on_question  

BENCH = [
    "I want to escape the summer heat from New York and head somewhere cool in Northern Europe, preferably by plane.",
    "Can you help me plan a relaxing train journey starting in Chicago, ending somewhere scenic on the West Coast?",
    "From San Francisco, I’d like to visit a nearby international city that’s famous for food. Which transport should I take?",
    "I’m in Boston and want to reach a cultural capital in Europe this winter; flights are okay.",
    "Could you suggest a short weekend trip from Seattle to a Canadian city, maybe by bus?",
    "I’d like to leave Miami and end up somewhere colder in the U.S., ideally by train.",
    "Starting in Los Angeles, I want to explore a historic East Coast city. Please plan transport.",
    "I want to go from Dallas to a coastal European destination, but I’m unsure whether to fly or take a train once I land.",
    "From Toronto, I want to travel to a European city that’s known for architecture.",
    "Help me book a trip from Denver to a major cultural festival somewhere in Europe.",
    "I want to leave Washington D.C. for a mountainous European city, but I’m not sure which transport mode is best.",
    "Plan me a journey from Paris to a Mediterranean destination where I can swim in summer.",
    "I’d like to go from London to a Northern European capital, maybe not the most expensive option.",
    "From Berlin, I want to reach a historic city in Southern Europe by train or flight.",
    "I’m in Madrid and I’d like to plan a getaway to a small coastal city, not too far away.",
    "Help me travel from Rome to a central European country that’s famous for castles.",
    "Starting in Amsterdam, I’d like to book a trip to a colder, less crowded destination.",
    "From Zurich, plan a route to a scenic city by train where I can enjoy the landscape.",
    "I want to leave Vienna for a culturally rich Eastern European city.",
    "I’d like to go from Prague to a romantic destination, maybe Paris or Venice.",
    "From Athens, I’d like to travel to another European city that is cooler in summer.",
    "Help me plan a winter journey from Oslo to a warmer European city.",
    "From Stockholm, I’d like to travel south to a European city by train, if possible.",
    "Starting in Helsinki, plan me a route to a historic city in Central Europe.",
    "I’m in Copenhagen and want to reach a famous art city in Europe by flight.",
    "From Istanbul, I’d like to travel to a European capital where I can experience history.",
    "Help me leave Warsaw for a Western European cultural hub.",
    "From Budapest, I’d like to plan a journey to a nearby capital city.",
    "I want to leave Brussels and reach a sunny European destination.",
    "From Lisbon, plan me a trip to a colder Northern European city.",
    "I’d like to travel from Dublin to a scenic European city with castles.",
    "Starting in Edinburgh, I want to end up in a romantic city on the continent.",
    "From Montreal, I’d like to visit a famous U.S. city by train if possible.",
    "I want to go from Vancouver to a large American cultural city.",
    "Starting in Mexico City, plan me a route to a European capital.",
    "From Buenos Aires, I’d like to fly to a cultural city in Europe.",
    "I want to leave São Paulo for a well-known European art city.",
    "Plan me a journey from Cape Town to a European destination with colder weather.",
    "I’d like to leave Cairo and head to a major European cultural city.",
    "From Dubai, I want to travel to a European destination by plane.",
    "I’m in Tokyo and want to plan a trip to Europe for sightseeing.",
    "From Seoul, I’d like to reach a famous European capital by plane.",
    "Starting in Beijing, help me travel to a historic European city.",
    "From Shanghai, I’d like to book a trip to a romantic European capital.",
    "I want to leave Hong Kong for a scenic European city in the mountains.",
    "From Singapore, plan me a trip to a colder European capital.",
    "I’d like to go from Sydney to a European city for culture and history.",
    "Starting in Melbourne, I want to reach a romantic European destination.",
    "I want to leave Auckland and plan a journey to Europe.",
    "From Delhi, plan me a route to a European capital that’s not too hot in summer."
]

def compute_metrics(trace: List[Dict[str, Any]]) -> Dict[str, float]:
    steps = len(trace)
    arg_match = sum(1 for s in trace if s["agent_args"] == s["planned_args"])
    state_change = sum(1 for s in trace if s["observation"])
    final_success = 1.0 if trace and trace[-1]["state_after"].get("booking_confirmed") else 0.0
    pre_viol = sum(1 for s in trace if not s["pre_ok"])
    errors = sum(1 for s in trace if s["error"])

    try:
        from tool_list import TOOL_DICT
        def step_cost(s): return TOOL_DICT[s["tool"]].cost
        total_cost = sum(step_cost(s) for s in trace)
        wasted_cost = sum(step_cost(s) for s in trace if not s["observation"])
        wcr = (wasted_cost / total_cost) if total_cost > 0 else 0.0
    except Exception:
        wcr = 0.0

    return {
        "TSR": final_success,
        "AAR": arg_match / steps if steps else 0.0,
        "SCSR": state_change / steps if steps else 0.0,
        "PVR": pre_viol / steps if steps else 0.0,
        "EER": errors / steps if steps else 0.0,
        "WCR": wcr,
    }

def extract_plan(trace: List[Dict[str, Any]]) -> List[Tuple[str, List[str]]]:
    """返回 [(tool_name, planned_args_list), ...]"""
    return [(s["tool"], list(s["planned_args"])) for s in trace]

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def variant_dir(use_hint: bool) -> str:
    return "hint" if use_hint else "nohint"

def variant_paths(use_hint: bool) -> Tuple[str, str, str]:
    base = variant_dir(use_hint)
    ensure_dir(base)
    return (
        os.path.join(base, "plans_export.jsonl"),
        os.path.join(base, "plans_export.csv"),
        base,
    )

def append_jsonl(path: str, obj: Dict[str, Any]):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def write_csv_header(path: str):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["question","variant","step","tool","planned_args","agent_args","pre_ok","error"])

def append_csv_rows(path: str, question: str, variant: str, trace: List[Dict[str, Any]]):
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for s in trace:
            w.writerow([
                question,
                variant,
                s["step"],
                s["tool"],
                json.dumps(s["planned_args"], ensure_ascii=False),
                json.dumps(s["agent_args"], ensure_ascii=False),
                int(bool(s["pre_ok"])),
                s["error"] or ""
            ])

def run_variant(question: str, use_hint: bool, use_openai: bool=True):
    variant = "hint" if use_hint else "nohint"
    jsonl_path, csv_path, out_dir = variant_paths(use_hint)

    safe_slug = "".join(c for c in question[:50] if c.isalnum() or c in (" ", "_", "-")).strip().replace(" ", "_")
    html_name = f"agent_run_{variant}_{safe_slug}.html"
    html_path_full = os.path.join(out_dir, html_name)

    trace, html_path = run_agent_on_question(
        question,
        use_openai=use_openai,
        use_plan_hint=use_hint,
        out_html=html_path_full
    )
    return trace, html_path_full, variant, jsonl_path, csv_path

def main():
    for flag in (True, False):
        jsonl_path, csv_path, _ = variant_paths(flag)
        if os.path.exists(jsonl_path):
            os.remove(jsonl_path)
        write_csv_header(csv_path)

    summary = []
    for q in BENCH:
        # A: with hint
        trace_A, html_A, var_A, jsonl_A, csv_A = run_variant(q, use_hint=True)
        met_A = compute_metrics(trace_A)
        plan_A = extract_plan(trace_A)
        append_jsonl(jsonl_A, {
            "question": q,
            "variant": var_A,
            "plan": plan_A,
            "metrics": met_A,
            "html": html_A,
            "steps": trace_A  
        })
        append_csv_rows(csv_A, q, var_A, trace_A)

        trace_B, html_B, var_B, jsonl_B, csv_B = run_variant(q, use_hint=False)
        met_B = compute_metrics(trace_B)
        plan_B = extract_plan(trace_B)
        append_jsonl(jsonl_B, {
            "question": q,
            "variant": var_B,
            "plan": plan_B,
            "metrics": met_B,
            "html": html_B,
            "steps": trace_B
        })
        append_csv_rows(csv_B, q, var_B, trace_B)

        delta = {k: met_A[k] - met_B[k] for k in met_A}
        summary.append({"q": q, "Δ": delta})
        print(f"[OK] {q[:48]}...  ΔTSR={delta['TSR']:+.2f} ΔAAR={delta['AAR']:+.2f} ΔSCSR={delta['SCSR']:+.2f}")

    def mean_key(key): return stats.mean(r["Δ"][key] for r in summary)
    print("\n=== Hint Gain (mean deltas) ===")
    for k in ["TSR","AAR","SCSR","PVR","EER","WCR"]:
        print(f"Δ{k}: {mean_key(k):+.3f}")

    print("\nOutputs by variant:")
    print(" - hint/:   plans_export.jsonl, plans_export.csv, per-run HTMLs")
    print(" - nohint/: plans_export.jsonl, plans_export.csv, per-run HTMLs")

if __name__ == "__main__":
    main()
