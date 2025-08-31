import heapq
import json
import os
import re
from dataclasses import dataclass, field, asdict
from typing import List, Tuple
import random
import openai

from visualizer import VisualLogger
from state import State
from tool import Tool
from tool_list import TOOLS
from prompt import SYSTEM_PROMPT, ARG_PROMPT


@dataclass(order=True)
class Node:
    f: float
    g: float = field(compare=False)
    state: State = field(compare=False)
    plan: List[Tuple[Tool, Tuple[str, ...]]] = field(compare=False, default_factory=list)


class Planner:
    def __init__(self):
        self.logger = VisualLogger()

    # ---------------- GPT helper -----------------
    def call_gpt(self, prompt: str, user_content: str, max_tokens: int = 32, default="error"):
        try:
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.0,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print("[WARN] OpenAI call failed:", e)
            return default

    # --------------- Core helpers ---------------
    def predict_success(self, state: State, tool: Tool, args: Tuple[str, ...]) -> float:
        user_prompt = (
            f"Current state: {asdict(state)}\n"
            f"Tool: {tool.name}{args}\n"
            f"Description: {tool.description}\n"
            "Return a single float between 0 and 1."
        )
        out = self.call_gpt(SYSTEM_PROMPT, user_prompt, max_tokens=4, default=str(random.uniform(0.4, 0.9)))
        try:
            return float(out)
        except ValueError:
            return random.uniform(0.4, 0.9)

    CITY_RE = re.compile(r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)")
    TRANSPORT_MODES = ["train", "flight", "bus"]

    def propose_args(self, tool: Tool, state: State, user_question: str) -> List[Tuple[str, ...]]:
        user_prompt = (
            f"Current state: {asdict(state)}\n"
            f"Tool: {tool.name}\n"
            f"Description: {tool.description}\n"
            f"Tool input: {tool.arg_names}\n"
            f"User request: {user_question}\n"
            "Return only JSON list of tuples with up to 3 candidates.\n"
        )
        raw = self.call_gpt(ARG_PROMPT, user_prompt, max_tokens=64, default="error")
        try:
            candidates = eval(raw) if raw != "error" else []
            if not isinstance(candidates, list):
                raise ValueError
            candidates = [tuple(c) for c in candidates]
        except Exception:
            candidates = []

        if not candidates:
            cities = self.CITY_RE.findall(user_question)
            seen = set(); cities = [c for c in cities if not (c in seen or seen.add(c))]
            if tool.name == "set_origin":
                candidates = [(c,) for c in cities[:3]] or [("New York",)]
            elif tool.name == "set_destination":
                candidates = [(c,) for c in cities[1:4]] or [("Boston",)]
            elif tool.name == "select_transport":
                candidates = [(m,) for m in self.TRANSPORT_MODES]
            else:
                candidates = [tuple()]
        return candidates[:3]

    def propose_tools(self, state: State, user_question: str, top_k: int = 3):
        candidates = [t for t in TOOLS if t.precondition(state)]
        if not candidates:
            return []
        tools_brief = [
            {"name": t.name, "cost": t.cost, "description": t.description}
            for t in candidates
        ]
        user_prompt = (
            "You are ranking tools for the *next single step*.\n"
            "Return STRICT JSON: a list of objects {\"name\": str, \"p\": float}.\n\n"
            f"Current state:\n{asdict(state)}\n\nUser request:\n{user_question}\n\nAvailable tools:\n{tools_brief}\n"
        )
        raw = self.call_gpt(
            prompt="You return only JSON. No extra words.",
            user_content=user_prompt,
            max_tokens=256,
            default="error",
        )
        ranked = []
        try:
            data = json.loads(raw) if raw != "error" else []
            name2tool = {t.name: t for t in candidates}
            for item in data:
                name = item.get("name")
                p = float(item.get("p", 0.5))
                if name in name2tool:
                    ranked.append((name2tool[name], max(0.0, min(1.0, p))))
        except Exception:
            ranked = []
        if not ranked:
            # fallback heuristic ranking
            def heuristic_rank(t: Tool):
                score = 0.5
                if t.name == "confirm_booking" and state.transport and state.origin and state.destination:
                    score = 0.95
                elif t.name == "select_transport" and state.origin and state.destination and not state.transport:
                    score = 0.85
                elif t.name == "set_destination" and state.origin and not state.destination:
                    score = 0.75
                elif t.name == "set_origin" and not state.origin:
                    score = 0.70
                score -= 0.01 * t.cost
                return score
            ranked = sorted([(t, heuristic_rank(t)) for t in candidates], key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def heuristic(self, state: State) -> float:
        if state.booking_confirmed:
            return 0.0
        remaining_costs = [t.cost for t in TOOLS if t.precondition(state)]
        return min(remaining_costs) if remaining_costs else 0.0

    # --------------- Search ---------------------
    def astar(self, initial_state: State, user_question: str) -> List[Tuple[Tool, Tuple[str, ...]]]:
        open_set: List['Node'] = []
        closed = set()
        root = Node(f=self.heuristic(initial_state), g=0.0, state=initial_state, plan=[])
        heapq.heappush(open_set, root)

        while open_set:
            node = heapq.heappop(open_set)
            if node.state in closed:
                continue
            closed.add(node.state)
            if node.state.is_goal():
                return node.plan

            # consider ALL ranked tools for this step
            tool_candidates = self.propose_tools(node.state, user_question, top_k=3)
            if not tool_candidates:
                continue

            step_idx = len(node.plan)
            self.logger.begin_step(step_idx)

            # Track global best over all (tool, args)
            # best = (f_new, g_new, sim_state, new_plan, tool, args, prob, combined_p)
            best = None

            for tool, tool_prior in tool_candidates:
                if not tool.precondition(node.state):
                    continue

                # generate args & probs for this tool
                arg_candidates = self.propose_args(tool, node.state, user_question)
                probs = [self.predict_success(node.state, tool, a) for a in arg_candidates]

                # log: this tool's candidates for visualization
                self.logger.log_tool(step_idx, tool.name, tool_prior, arg_candidates, probs)

                # evaluate each (tool, args)
                for args, prob in zip(arg_candidates, probs):
                    sim_state = State(**asdict(node.state))
                    try:
                        tool(sim_state, *args)
                    except Exception:
                        continue

                    combined_p = max(1e-9, prob * tool_prior)  # guard against 0
                    g_new = node.g + tool.cost
                    h_new = self.heuristic(sim_state) / combined_p
                    f_new = g_new + h_new

                    if (best is None) or (f_new < best[0]):
                        new_plan = node.plan + [(tool, args)]
                        best = (f_new, g_new, sim_state, new_plan, tool, args, prob, combined_p)

            if best is None:
                continue

            f_new, g_new, new_state, new_plan, best_tool, best_args, best_prob, best_combined_p = best

            # visualize final choice for this step
            self.logger.set_choice(step_idx, best_tool.name, best_args, best_prob, best_combined_p)

            print(
                f"[Planner] Step {step_idx}: choose {best_tool.name}{best_args} | "
                f"prob={best_prob:.3f}, combined_p={best_combined_p:.3f}, "
                f"g={g_new:.3f}, h={(f_new-g_new):.3f}, f={f_new:.3f}"
            )

            heapq.heappush(open_set, Node(f=f_new, g=g_new, state=new_state, plan=new_plan))

        raise RuntimeError("No plan found")

    # --------------- Entry ----------------------
    def main(self):
        user_question = "Book a ticket from New York to some place cold in Europe"
        print("User question:", user_question)
        init_state = State()
        plan = self.astar(init_state, user_question)

        # Execute the plan
        exec_state = State()
        for i, (tool, args) in enumerate(plan):
            tool(exec_state, *args)
            self.logger.mark_executed(i)
            print(f"Executed {tool.name}{args} -> {exec_state}")
        self.logger.generate_html()
        print("Final state:", exec_state)


if __name__ == "__main__":
    Planner().main()
