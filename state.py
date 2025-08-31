from dataclasses import dataclass, field, asdict

@dataclass
class State:
    origin: str = None
    destination: str = None
    transport: str = None
    booking_confirmed: bool = False

    def is_goal(self) -> bool:
        return self.booking_confirmed

    def __hash__(self):
        return hash((self.origin, self.destination, self.transport, self.booking_confirmed))

    def __repr__(self):
        return (
            f"State(origin={self.origin}, destination={self.destination}, "
            f"transport={self.transport}, confirmed={self.booking_confirmed})"
        )