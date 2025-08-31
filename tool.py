from typing import Callable, List, Tuple
from dataclasses import dataclass, field, asdict
from state import State

@dataclass
class Tool:
    name: str
    description: str
    cost: float
    precondition: Callable[[State], bool]
    effect: Callable[[State, Tuple[str, ...]], None]
    arg_names: Tuple[str, ...] = field(default_factory=tuple)

    def __call__(self, state: State, *args):
        if not self.precondition(state):
            raise ValueError(f"Precondition failed for {self.name}")
        self.effect(state, args)
        return True


