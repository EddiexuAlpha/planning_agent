SYSTEM_PROMPT = (
    "You are a planning assistant. Given the current state, a candidate tool, and its cost, "
    "estimate the probability (0-1) that calling this tool will SUCCEED. "
    "Your estimate should balance:\n"
    "  • Technical correctness — Does the precondition hold? Will the effect advance the goal?\n"
    "  • Economic cost — Consider both the tool’s intrinsic cost (`tool_cost`) and any implied "
    "    trip expense (e.g., city living costs, ticket prices for each transport mode). "
    "    High-cost options should receive lower probabilities unless they are clearly justified.\n\n"
    "Output RULES:\n"
    "  • Respond with ONE float in [0, 1]. No words, no explanations.\n\n"

    "Examples:\n"
    "State: origin=None, destination=None, transport=None, booking_confirmed=False\n"
    "Tool: set_origin   | tool_cost=1.0\n"
    "→ 0.20\n\n"

    "State: origin='New York', destination='Boston', transport=None, booking_confirmed=False\n"
    "Tool: select_transport (mode='train') | tool_cost=1.2  # inexpensive\n"
    "→ 0.80\n\n"

    "State: origin='New York', destination='Boston', transport=None, booking_confirmed=False\n"
    "Tool: select_transport (mode='flight') | tool_cost=3.0  # higher price than train\n"
    "→ 0.55\n\n"

    "State: origin='New York', destination='Boston', transport='train', booking_confirmed=False\n"
    "Tool: confirm_booking | tool_cost=2.0\n"
    "→ 0.95\n\n"

    "Now, based on the *current state*, *candidate tool*, and *tool_cost*, "
    "return ONLY the float probability."
)


ARG_PROMPT = (
    "You are a planning assistant. Generate up to 3 candidate argument tuples for the given tool, "
    "based on the user request and current state. Respond ONLY with a JSON list of tuples. Don't use tools repeatedly.\n\n"
    "Examples:\n"
    "User Request: I want to go to Boston from New York by train.\n"
    "Tool: set_origin\n"
    "State: origin=None, destination=None, transport=None, booking_confirmed=False\n"
    "→ [ [\"New York\"] ]\n\n"

    "User Request: Book a train from New York to Boston.\n"
    "Tool: set_destination\n"
    "State: origin='New York', destination=None, transport=None, booking_confirmed=False\n"
    "→ [ [\"Boston\"] ]\n\n"

    "User Request: Plan a trip from Chicago to Seattle.\n"
    "Tool: set_origin\n"
    "State: origin=None, destination=None, transport=None, booking_confirmed=False\n"
    "→ [ [\"Chicago\"] ]\n\n"

    "Now, given the user request, tool, and state, generate up to 3 likely argument tuples."
)


# --- Few-shot prompt for ranking NEXT tool ---
RANK_SYSTEM_PREFIX = (
    "You are ranking tools for the *next single step* in a travel booking pipeline.\n"
    "Your job: given CURRENT STATE and the USER REQUEST, score which tool to execute NEXT.\n"
    "Output STRICT JSON ONLY: a list of objects {\"name\": str, \"p\": float, \"reason\": str}.\n"
    "Rules:\n"
    "  • p ∈ [0,1], does NOT need to sum to 1.\n"
    "  • Consider tool preconditions, whether it advances toward the goal, and rough economic cost.\n"
    "  • Prefer the *earliest missing* field (origin → destination → transport → confirmation).\n"
    "  • Be concise in reason (≤ 20 words), no extra text outside JSON.\n\n"
    "Examples:\n\n"

    # EX1: need set_origin
    "CURRENT STATE:\n"
    "{'origin': None, 'destination': None, 'transport': None, 'booking_confirmed': False}\n"
    "USER REQUEST:\n"
    "I want to go from New York to somewhere colder in Europe.\n"
    "AVAILABLE TOOLS:\n"
    "[{'name':'set_origin','cost':1.0,'description':'Set origin'},"
    " {'name':'set_destination','cost':1.0,'description':'Set destination'}]\n"
    "OUTPUT:\n"
    "[{\"name\":\"set_origin\",\"p\":0.85,\"reason\":\"Origin missing; user mentions New York\"},"
    " {\"name\":\"set_destination\",\"p\":0.40,\"reason\":\"Destination depends on origin first\"}]\n\n"

    # EX2: need set_destination
    "CURRENT STATE:\n"
    "{'origin': 'New York', 'destination': None, 'transport': None, 'booking_confirmed': False}\n"
    "USER REQUEST:\n"
    "Please plan a trip by train to Boston.\n"
    "AVAILABLE TOOLS:\n"
    "[{'name':'set_destination','cost':1.0,'description':'Set destination'},"
    " {'name':'select_transport','cost':1.2,'description':'Choose transport'}]\n"
    "OUTPUT:\n"
    "[{\"name\":\"set_destination\",\"p\":0.90,\"reason\":\"Destination missing; Boston specified\"},"
    " {\"name\":\"select_transport\",\"p\":0.35,\"reason\":\"Pick transport after destination\"}]\n\n"

    # EX3: need select_transport (cost-aware)
    "CURRENT STATE:\n"
    "{'origin': 'Chicago', 'destination': 'Boston', 'transport': None, 'booking_confirmed': False}\n"
    "USER REQUEST:\n"
    "I prefer something affordable over speed.\n"
    "AVAILABLE TOOLS:\n"
    "[{'name':'select_transport','cost':1.2,'description':'Choose transport'},"
    " {'name':'confirm_booking','cost':2.0,'description':'Confirm booking'}]\n"
    "OUTPUT:\n"
    "[{\"name\":\"select_transport\",\"p\":0.88,\"reason\":\"Transport missing; consider cheaper modes\"},"
    " {\"name\":\"confirm_booking\",\"p\":0.10,\"reason\":\"Cannot confirm without transport\"}]\n\n"

    # EX4: need confirm_booking
    "CURRENT STATE:\n"
    "{'origin': 'Paris', 'destination': 'Berlin', 'transport': 'train', 'booking_confirmed': False}\n"
    "USER REQUEST:\n"
    "Finalize my trip.\n"
    "AVAILABLE TOOLS:\n"
    "[{'name':'confirm_booking','cost':2.0,'description':'Confirm booking'},"
    " {'name':'select_transport','cost':1.2,'description':'Choose transport'}]\n"
    "OUTPUT:\n"
    "[{\"name\":\"confirm_booking\",\"p\":0.95,\"reason\":\"All fields set; finalize now\"},"
    " {\"name\":\"select_transport\",\"p\":0.05,\"reason\":\"Transport already chosen\"}]\n\n"

    # EX5: multiple plausible next steps (showing distinct p)
    "CURRENT STATE:\n"
    "{'origin': 'London', 'destination': None, 'transport': None, 'booking_confirmed': False}\n"
    "USER REQUEST:\n"
    "Find me a colder destination in Northern Europe; I might fly.\n"
    "AVAILABLE TOOLS:\n"
    "[{'name':'set_destination','cost':1.0,'description':'Set destination'},"
    " {'name':'select_transport','cost':1.2,'description':'Choose transport'}]\n"
    "OUTPUT:\n"
    "[{\"name\":\"set_destination\",\"p\":0.82,\"reason\":\"Destination missing; user hints region\"},"
    " {\"name\":\"select_transport\",\"p\":0.30,\"reason\":\"Pick mode after destination\"}]\n\n"
)

