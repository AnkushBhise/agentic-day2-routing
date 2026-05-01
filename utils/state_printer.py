"""
Generic State Printer — prints any LangGraph state dict.

No field names are hardcoded. Automatically detects:
  - BaseMessage lists  → prints label + content preview
  - Sequence of str    → prints numbered list
  - Booleans           → prints True/False
  - Dicts              → prints key/value pairs
  - Any other scalar   → prints as-is

Usage:
    from state_printer import print_state

    print_state(state)                      # full state
    print_state(state, section="reasoning") # single key
    print_state(state, title="After Node")  # custom title
"""

from typing import Optional, Any


# ---------------------------------------------------------------------------
# Detect if something is a BaseMessage without hard importing langchain
# ---------------------------------------------------------------------------

def _is_base_message(obj: Any) -> bool:
    """Works even if langchain is not installed."""
    return hasattr(obj, "content") and hasattr(obj, "type")


def _msg_label(msg: Any) -> str:
    type_label = getattr(msg, "type", "message").title()
    name       = getattr(msg, "name", None)
    return f"{type_label}({name})" if name else type_label


def _divider(char: str = "─", width: int = 56) -> str:
    return char * width


# ---------------------------------------------------------------------------
# Renderers — one per value type
# ---------------------------------------------------------------------------

def _render_message_list(messages: list) -> None:
    if not messages:
        print("    (empty)")
        return
    for i, msg in enumerate(messages, 1):
        label   = _msg_label(msg)
        content = str(msg.content)
        preview = content[:120] + ("..." if len(content) > 120 else "")
        print(f"  [{i}] {label}")
        print(f"       {preview}")


def _render_str_sequence(items: list) -> None:
    if not items:
        print("    (empty)")
        return
    for i, item in enumerate(items, 1):
        # "agent_name : reasoning text" → aligned two-line display
        if isinstance(item, str) and " : " in item:
            agent, text = item.split(" : ", 1)
            print(f"  [{i}] {agent.strip()}")
            print(f"       → {text.strip()}")
        else:
            print(f"  [{i}] {item}")


def _render_dict(d: dict, indent: int = 2) -> None:
    pad = " " * indent
    for k, v in d.items():
        print(f"  {pad}{k}: {v}")


def _render_scalar(value: Any) -> None:
    print(f"  {value}")


# ---------------------------------------------------------------------------
# Core dispatcher — figures out what a value is and picks a renderer
# ---------------------------------------------------------------------------

def _render_value(key: str, value: Any) -> None:
    # Empty / None
    if value is None:
        print("  —")
        return

    # List — check what's inside
    if isinstance(value, (list, tuple)):
        if not value:
            print("  (empty)")
        elif _is_base_message(value[0]):
            _render_message_list(list(value))
        elif isinstance(value[0], str):
            _render_str_sequence(list(value))
        else:
            # Generic list — just enumerate items
            for i, item in enumerate(value, 1):
                print(f"  [{i}] {item}")
        return

    # Dict
    if isinstance(value, dict):
        _render_dict(value)
        return

    # Scalar (str, bool, int, float…)
    _render_scalar(value)


# ---------------------------------------------------------------------------
# Section type label — shown next to key name
# ---------------------------------------------------------------------------

def _type_label(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        if not value:
            return "(list, empty)"
        if _is_base_message(value[0]):
            return f"({len(value)} messages)"
        return f"({len(value)} entries)"
    if isinstance(value, dict):
        return f"({len(value)} keys)"
    if isinstance(value, bool):
        return "(bool)"
    return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def print_state(
    state   : dict,
    section : Optional[str] = None,
    title   : str = "GRAPH STATE",
) -> None:
    """
    Pretty print any LangGraph state dict.

    Args:
        state   : Any TypedDict / dict used as graph state.
        section : If given, prints only that key.
        title   : Header label (default: "GRAPH STATE").
    """

    # ── Single section ──────────────────────────────────────────────────────
    if section:
        if section not in state:
            print(f"\n  Key '{section}' not found in state.")
            print(f"  Available keys: {list(state.keys())}\n")
            return

        value = state[section]
        print(f"\n  STATE['{section}'] {_type_label(value)}")
        print(f"  {_divider()}")
        _render_value(section, value)
        print()
        return

    # ── Full state ──────────────────────────────────────────────────────────
    print(f"\n{'═' * 56}")
    print(f"  {title}")
    print(f"{'═' * 56}")

    # Split keys into scalars vs collections for cleaner layout
    scalars     = {}
    collections = {}

    for key, value in state.items():
        if key.startswith("_"):               # skip internal meta keys
            continue
        if isinstance(value, (list, tuple, dict)) and value != "":
            collections[key] = value
        else:
            scalars[key] = value

    # Scalars block
    if scalars:
        print("\n  SCALARS")
        print(f"  {_divider()}")
        for key, value in scalars.items():
            display = "—" if value in (None, "", False) and value != 0 else value
            print(f"  {key:<22} {display}")

    # Collection blocks — one section per key
    for key, value in collections.items():
        label = _type_label(value)
        print(f"\n  {key.upper()}  {label}")
        print(f"  {_divider()}")
        _render_value(key, value)

    print(f"\n{'═' * 56}\n")


# ---------------------------------------------------------------------------
# Quick test — two different state shapes
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Simulate BaseMessage without langchain dependency for the test
    class FakeHuman:
        type = "human"; name = None
        def __init__(self, text): self.content = text

    class FakeAI:
        type = "ai"
        def __init__(self, text, name=None): self.content = text; self.name = name

    # ── SupportState ────────────────────────────────────────────────────────
    support_state = {
        "messages": [
            FakeHuman("I was charged twice this month"),
            FakeAI('{"reasoning":"billing issue","issue_type":"billing"}', "determine_issue_type"),
        ],
        "reasoning": [
            "determine_issue_type : User mentioned being charged twice — billing issue",
            "billing_agent        : Within 30-day window, eligible for refund",
        ],
        "should_escalate" : False,
        "issue_type"      : "billing",
        "user_tier"       : "standard",
        "user_path"       : "standard_path",
        "priority"        : "normal",
        "user_id"         : "usr_001",
        "ticket_id"       : "tkt_001",
        "final_resolution": "",
    }

    print_state(support_state, title="SUPPORT STATE")
    print_state(support_state, section="reasoning")
    print_state(support_state, section="messages")

    # ── A totally different state shape ─────────────────────────────────────
    research_state = {
        "query"     : "latest AI papers",
        "sources"   : ["arxiv.org", "papers.nips.cc"],
        "summaries" : [
            "search_agent : Found 12 relevant papers on transformer scaling",
            "rank_agent   : Top 3 filtered by citation count",
        ],
        "final_answer" : "",
        "iterations"   : 3,
        "done"         : False,
    }

    print_state(research_state, title="RESEARCH STATE")