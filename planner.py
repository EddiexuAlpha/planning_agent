import heapq
import json
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Callable, List, Tuple
import random
import openai
from state import State
from tool import Tool
from tool_list import TOOLS, TOOL_DICT
from prompt import RANK_SYSTEM_PREFIX, SYSTEM_PROMPT, ARG_PROMPT

@dataclass(order=True)
class Node:
    f: float
    g: float = field(compare=False)
    state: State = field(compare=False)
    plan: List[Tuple[Tool, Tuple[str, ...]]] = field(compare=False, default_factory=list)

class planner:

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
        """Return a list of candidate argument tuples for the tool."""
        # 1) Try GPT
        user_prompt = (
            f"Current state: {asdict(state)}\n"
            f"Tool: {tool.name}\n"
            f"Description: {tool.description}\n"
            f"Tool input: {tool.arg_names}\n"
            f"Tool precondition: {tool.precondition}\n"
            f"Tool effect: {tool.effect}\n"
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
            RANK_SYSTEM_PREFIX
            + "CURRENT STATE:\n"
            + f"{asdict(state)}\n\n"
            + "USER REQUEST:\n"
            + f"{user_question}\n\n"
            + "AVAILABLE TOOLS:\n"
            + f"{tools_brief}\n"
            + "OUTPUT:\n"
        )

        raw = self.call_gpt(
            prompt="You return only JSON. No extra words.",
            user_content=user_prompt,
            max_tokens=256,
            default="error"
        )

        ranked = []

        print("the raw is", raw)
        try:
            if raw != "error":
                print("have data")
            else:
                print("no data")
            data = json.loads(raw) if raw != "error" else []
            if isinstance(data, list):
                name2tool = {t.name: t for t in candidates}
                for item in data:
                    name = item.get("name")
                    p = float(item.get("p", 0.5))
                    if name in name2tool:
                        ranked.append((name2tool[name], max(0.0, min(1.0, p))))
        except Exception:
            ranked = []

        return ranked[:top_k]


    def heuristic(self, state: State) -> float:
        if state.booking_confirmed:
            return 0.0
        # optimistic: assume next cheapest tool will succeed
        remaining_costs = [t.cost for t in TOOLS if t.precondition(state)]
        return min(remaining_costs) if remaining_costs else 0.0


    def astar(self, initial_state: State, user_question: str) -> List[Tuple[Tool, Tuple[str, ...]]]:
        open_set: List[Node] = []
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
            
            tool_candidates = self.propose_tools(node.state, user_question, top_k=3) # tool should use
            print("[Planner] Step:", len(node.plan), "Tool Candidates:", tool_candidates)
            print("Tool Choosed is:", tool_candidates[0])
            tool, tool_prior =  tool_candidates[0]
            if not tool.precondition(node.state):
                continue

            best = None  # (f_new, g_new, sim_state, new_plan, args, prob, combined_p)

            for args in self.propose_args(tool, node.state, user_question):
                sim_state = State(**asdict(node.state))
                try:
                    success = tool(sim_state, *args)
                except Exception:
                    continue

                if success is False:
                    continue

                prob = self.predict_success(node.state, tool, args) # argment should use
                combined_p =  prob * tool_prior

                g_new = node.g + tool.cost
                h_new = self.heuristic(sim_state) / combined_p # TODO: using probability 
                f_new = g_new + h_new

                if (best is None) or (f_new < best[0]):
                    new_plan = node.plan + [(tool, args)]
                    best = (f_new, g_new, sim_state, new_plan, args, prob, combined_p)

            if best is not None:
                f_new, g_new, new_state, new_plan, args, prob, combined_p = best
                print(
                    f"[Choose Args] {tool.name}{args} | success=True, "
                    f"prob={prob:.3f}, prior={tool_prior:.3f}, combined_p={combined_p:.3f}, "
                    f"g={g_new:.3f}, h={f_new-g_new:.3f}, f={f_new:.3f} -> {new_state}"
                )
                heapq.heappush(open_set, Node(f=f_new, g=g_new, state=new_state, plan=new_plan))

        raise RuntimeError("No plan found")


    def main(self):
        # get the input from the user
        # user_question = input("Enter your question (e.g. 'Book a flight from New York to Boston'): ").strip()
        # user_question = re.sub(r'\s+', ' ', user_question)  # normalize whitespace
        user_question = "Book a ticket from New York to some place cold in Europe"
        if not user_question:
            print("No input provided. Exiting.")
            return
        print("User question:", user_question)

        init_state = State()
        plan = self.astar(init_state, user_question)

        exec_state = State()
        print("\nPlanned sequence (with GPT‑proposed args):")
        tool_sequence = {}
        for i, (tool, args) in enumerate(plan, 1):
            if tool.name not in tool_sequence:
                tool_sequence[tool.name] = args
                success = tool(exec_state, *args)
                print(f"{i}. {tool.name}{args} -> success={success} -> {exec_state}")

        print("\nFinal state:", exec_state)
        print("\nTool Seqence:", tool_sequence)


if __name__ == "__main__":
    agent = planner()
    agent.main()

