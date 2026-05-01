from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
import operator
from typing import TypedDict, Annotated, Sequence
from operator import add
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph import graph
from langgraph.graph import StateGraph, END
import json

from matplotlib import category
from mypy.state import state
from pydantic.v1 import compiled
from utils.prompt_manager import PromptManager
from utils.state_printer import print_state
from langchain.agents import create_agent
from langgraph.graph import StateGraph, END


load_dotenv()

llm_openai = ChatOpenAI(model="gpt-4.1-nano")



# ===== Graph State =====
class SupportState(TypedDict):
	# ── Append fields (operator.add — each node appends, never overwrites) ─
	messages: Annotated[list[BaseMessage], operator.add]
	reasoning: Annotated[Sequence[str], operator.add]
	
    # ── Scalar fields (last-write-wins) ───────────────────────────────────
	should_escalate: bool
	issue_type: str # 'billing' | 'shipping' | 'technical' | 'general'
	user_tier: str  # "vip" or "standard"
	priority: str  # "low", "normal", "high"
	user_id: str
	ticket_id: str
	final_resolution: str


# ===== TOOLS =====
@tool("get_customer_tier")
def get_customer_tier(customer_id: str) -> str:
    """
    Look up a customer's tier (vip or standard) by their customer ID.

    Args:
        customer_id: Customer ID string

    Returns:
        'vip' or 'standard'
    """
    vip_customers = {"ID-001", "ID-003", "ID-007"}
    return "vip" if customer_id in vip_customers else "standard"


@tool("create_ticket")
def create_ticket(customer_id: str, issue: str, priority: str) -> str:
    """
    Create a support ticket.

    Args:
        customer_id: Customer ID
        issue: Issue description
        priority: 'low', 'normal', or 'high'

    Returns:
        Ticket ID
    """
    import random
    ticket_id = f"TKT-{random.randint(10000, 99999)}"
    print(f"[DB] Created ticket {ticket_id} | Customer: {customer_id} | Priority: {priority}")
    return ticket_id


# ===== NODES =====
def extract_customer_id(state: SupportState) -> dict:
    """Extract customer ID from the latest message."""
    print("Step: EXTRACT CUSTOMER ID")
    # print_state(state, title="EXTRACT CUSTOMER ID - INPUT STATE")
    last_message = state["messages"][-1].content
    # Simple extraction — in production use regex or NLP
    import re
    match = re.search(r'ID-\d+', last_message.upper())
    user_id = match.group(0) if match else "ID-UNKNOWN"
    print(f"  [extract] user_id = {user_id}")
    return {"user_id": user_id}


def determine_issue_type(state: SupportState) -> str:
    """
	Determine issue type based on user messages.
	Only one of 'billing', 'shipping', 'technical', or 'general' should be returned.
	"""
    print("Step: DETERMINE ISSUE TYPE")
    print_state(state, title="DETERMINE ISSUE TYPE - INPUT STATE")
    agent = "determine_issue_type"
    last_message = state["messages"][-1].content
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    manager = PromptManager(prompts_dir="prompts")
    compiled = manager.compile_messages(agent, last_message)

    response = llm.invoke([
        SystemMessage(content=compiled["system"]),
        HumanMessage(content=compiled["user"]),
		]
	)
    output = json.loads(response.content.strip().lower())
    print(f"[LLM] determined issue type: {output['issue_type']}")
    output = {
      "reasoning": [output["reasoning"]],
      "issue_type": output["issue_type"]
    }
    return output

def check_user_tier_node(state: SupportState):
	"""
	Decide if user is VIP or standard
	"""
	print("Step: CHECK USER TIER")
	print_state(state, title="CHECK USER TIER - INPUT STATE")
	agent = "check_user_tier"
	last_message = state["messages"][-1].content

	# Enrich user message with state context so LLM knows priority + issue_type
	enriched_message: str = f"""
	Customer ID : {state['user_id']}
	User Message: {last_message}
	""".strip()
	manager = PromptManager(prompts_dir="prompts")
	compiled = manager.compile_messages(agent, enriched_message)

	agent_llm = create_agent(
    model=ChatOpenAI(model="gpt-4.1-nano"),
    tools=[get_customer_tier],
	system_prompt=compiled["system"],
    )

	result = agent_llm.invoke({"messages":[
        {"role": "user", "content": compiled["user"]},
    ]})

	for msg in reversed(result["messages"]):
		if isinstance(msg, AIMessage) and msg.content:
			ai_message = json.loads(msg.content.strip().lower())

	output = {
		"reasoning": [ai_message["reasoning"]],
		"user_tier": ai_message["user_tier"],
		"priority": ai_message["priority"],
		"should_escalate": ai_message["should_escalate"]
	}
	print(f"[LLM] determined user tier: {output['user_tier']}")
	return output


def route_by_tier(state: SupportState) -> str:
	"""Route based on user tier."""
	if state.get("user_tier") == "vip":
		return "vip_path"
	return "standard_path"


def vip_agent_node(state: SupportState):
	print_state(state, title="VIP AGENT - INPUT STATE")
	agent = "vip_agent"
	last_message = state["messages"][-1].content

	# Enrich user message with state context so LLM knows priority + issue_type
	enriched_message: str = f"""
	Customer ID : {state['user_id']}
	Priority   : {state['priority']}
	Issue Type : {state['issue_type']}
	Should Escalate : {state['should_escalate']}
	User Message: {last_message}
	""".strip()
	manager = PromptManager(prompts_dir="prompts")
	compiled = manager.compile_messages(agent, enriched_message)

	agent_llm = create_agent(
    model=ChatOpenAI(model="gpt-4.1-nano"),
    tools=[create_ticket],
	system_prompt=compiled["system"],
    )

	result = agent_llm.invoke({"messages":[
        {"role": "user", "content": compiled["user"]},
    ]})

	for msg in reversed(result["messages"]):
		if isinstance(msg, AIMessage) and msg.content:
			ai_message = json.loads(msg.content.strip().lower())
			break

	output = {
		"reasoning": [ai_message["reasoning"]],
		"final_resolution": ai_message["final_resolution"],
		"ticket_id": ai_message["ticket_id"]
	}
	print(f"[LLM] determined final resolution: {output['final_resolution']}")
	return output


def standard_agent_node(state: SupportState) -> dict:
	print_state(state, title="STANDARD AGENT - INPUT STATE")
	agent = "standard_agent"
	last_message = state["messages"][-1].content

	# Enrich user message with state context so LLM knows priority + issue_type
	enriched_message: str = f"""
	Customer ID : {state['user_id']}
	Priority   : {state['priority']}
	Issue Type : {state['issue_type']}
	Should Escalate : {state['should_escalate']}
	User Message: {last_message}
	""".strip()
	manager = PromptManager(prompts_dir="prompts")
	compiled = manager.compile_messages(agent, enriched_message)

	agent_llm = create_agent(
    model=ChatOpenAI(model="gpt-4.1-nano"),
    tools=[create_ticket],
	system_prompt=compiled["system"],
    )

	result = agent_llm.invoke({"messages":[
        {"role": "user", "content": compiled["user"]},
    ]})

	for msg in reversed(result["messages"]):
		if isinstance(msg, AIMessage) and msg.content:
			ai_message = json.loads(msg.content.strip().lower())
			break

	output = {
		"reasoning": [ai_message["reasoning"]],
		"final_resolution": ai_message["final_resolution"],
		"ticket_id": ai_message["ticket_id"]
	}
	print(f"[LLM] determined final resolution: {output['final_resolution']}")
	return output



def build_graph():

	# add nodes and edges to the graph
	workflow = StateGraph(SupportState)
	workflow.add_node("extract_customer_id", extract_customer_id)
	workflow.add_node("check_tier", check_user_tier_node)
	workflow.add_node("determine_issue_type", determine_issue_type)
	workflow.add_node("vip_agent", vip_agent_node)
	workflow.add_node("standard_agent", standard_agent_node)

	# define edges
	workflow.set_entry_point("extract_customer_id")
	workflow.add_edge("extract_customer_id", "determine_issue_type")
	workflow.add_edge("determine_issue_type", "check_tier")
	workflow.add_conditional_edges(
		"check_tier",
		route_by_tier,
		{
			"vip_path": "vip_agent",
			"standard_path": "standard_agent",
		},
	)
	workflow.add_edge("vip_agent", END)
	workflow.add_edge("standard_agent", END)
	return workflow.compile()


from langchain_core.messages import HumanMessage


def main() -> None:
	graph = build_graph()

    # ── VIP customer invoke ────────────────────────────────────────────────
	vip_result = graph.invoke({
        "messages"        : [HumanMessage(content="I'm furious! I've been charged 3 times and nobody is helping. I'll take legal action!, My user is ID-001")],
        "reasoning"       : [],
        "should_escalate" : False,
        "issue_type"      : "",
        "user_tier"       : "",
        "user_path"       : "",
        "priority"        : "",
        "user_id"         : "",
        "ticket_id"       : "",
        "final_resolution": "",
    })
	print_state(vip_result, title="VIP AGENT - OUTPUT STATE")

    # ── Standard customer invoke ───────────────────────────────────────────
	standard_result = graph.invoke({
        "messages"        : [HumanMessage(content="Hi, I have a question about my last invoice. My user is ID-009, and I think I'm in the wrong tier")],
        "reasoning"       : [],
        "should_escalate" : False,
        "issue_type"      : "",
        "user_tier"       : "",
        "user_path"       : "",
        "priority"        : "",
        "user_id"         : "",
        "ticket_id"       : "",
        "final_resolution": "",
    })
	print_state(standard_result, title="STANDARD AGENT - OUTPUT STATE")


if __name__ == "__main__":
	main()