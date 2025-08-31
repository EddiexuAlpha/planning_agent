from tool import Tool


TOOLS = [
    Tool(
        name="set_origin",
        description="Set the origin city for travel",
        cost=1.0,
        precondition=lambda s: s.origin is None,
        effect=lambda s, args: setattr(s, "origin", args[0]),
        arg_names=("city",)
    ),

    Tool(
        name="set_destination",
        description="Set the destination city",
        cost=1.0,
        precondition=lambda s: s.origin is not None and s.destination is None,
        effect=lambda s, args: setattr(s, "destination", args[0]),
        arg_names=("city",)
    ),

    Tool(
        name="select_transport",
        description="Choose a transport mode (e.g. train, flight)",
        cost=1.2,
        precondition=lambda s: s.destination is not None and s.transport is None,
        effect=lambda s, args: setattr(s, "transport", args[0]),
        arg_names=("mode",)
    ),

    Tool(
        name="confirm_booking",
        description="Finalize the booking and mark as confirmed",
        cost=2.0,
        precondition=lambda s: s.transport is not None and not s.booking_confirmed,
        effect=lambda s, args: setattr(s, "booking_confirmed", True),
        arg_names=()
    ),
]

TOOL_DICT = {t.name: t for t in TOOLS}
