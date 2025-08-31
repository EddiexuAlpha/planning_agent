from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import pandas as pd
from state import State
from tool import Tool


class Executor:
    """Iteratively applies Tools to a State while recording the full trace."""

    def __init__(self, initial_state: State | None = None):
        self.state: State = initial_state or State()
        # Each snapshot is a dict holding the state after the step.
        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Core execution helpers
    # ------------------------------------------------------------------
    def run_step(self, step_idx: int, tool: Tool, args: Tuple[str, ...]):
        """Execute *one* tool call and record the resulting state."""
        # Execute (will raise if precondition fails).
        tool(self.state, *args)

        # Add a snapshot row for visualisation.
        self.history.append(
            {
                "Step": step_idx,
                "Tool": tool.name,
                "Args": str(args),
                "Origin": self.state.origin,
                "Destination": self.state.destination,
                "Transport": self.state.transport,
                "Confirmed": self.state.booking_confirmed,
            }
        )

    def run_plan(self, plan: List[Tuple[Tool, Tuple[str, ...]]]):
        """Run the whole plan (a list of (Tool, args) tuples)."""
        for idx, (tool, args) in enumerate(plan, start=1):
            self.run_step(idx, tool, args)
        return self.history

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:  # noqa: D401 (simple verb)
        """Return the execution trace as a *pandas* DataFrame."""
        return pd.DataFrame(self.history)

    def display_dataframe(self, title: str = "Execution Trace"):
        """Render the DataFrame inside ChatGPT (if possible)."""
        try:
            from ace_tools import display_dataframe_to_user  # type: ignore

            display_dataframe_to_user(title, self.to_dataframe())
        except ImportError:
            # Fallback for environments without the ChatGPT helper.
            print(self.to_dataframe())

    def plot_timeline(self):
        import matplotlib.pyplot as plt

        df = self.to_dataframe()
        plt.figure()
        plt.scatter(df["Step"], df["Tool"])
        plt.yticks(range(len(df["Tool"].unique())), df["Tool"].unique())
        plt.xlabel("Step")
        plt.title("Executor Timeline")
        plt.tight_layout()
        plt.show()


def run_executor_on_question(user_question: str) -> pd.DataFrame:
    """Plan + execute in one call and return a DataFrame trace."""
    from planner import planner

    p = planner()
    plan = p.astar(State(), user_question)

    exe = Executor(State())
    exe.run_plan(plan)
    return exe.to_dataframe()


if __name__ == "__main__":
    df = run_executor_on_question("Book a flight from New York to Boston")
    print(df)
