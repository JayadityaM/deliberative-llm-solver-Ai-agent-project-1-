from langchain_aws import ChatBedrock
import boto3
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

# -----------------------------------------------------
# Bedrock Session
# -----------------------------------------------------

session = boto3.Session(profile_name="sandbox", region_name="us-east-1")
bedrock_client = session.client("bedrock-runtime")

llm = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    client=bedrock_client,
    temperature=0,
    streaming=True
)

# -----------------------------------------------------
# State Definition
# -----------------------------------------------------

class AgentState(TypedDict):
    problem: str
    depth: str
    plan: List[str]
    final_answer: str


# -----------------------------------------------------
# Planner Node (STREAMING PLAN)
# -----------------------------------------------------

def planner_node(state: AgentState):
    depth = state["depth"]

    if depth == "quick":
        constraint = "Use 3–4 concise reasoning steps."
    elif depth == "deep":
        constraint = "Use 8–12 detailed reasoning steps."
    else:
        constraint = "Use around 5–7 reasoning steps."

    prompt = f"""
You are a senior engineer.
ONLY output a numbered reasoning plan.
{constraint}

Problem:
{state["problem"]}
"""

    print("\n===== GENERATED PLAN (Streaming) =====\n")

    full_output = ""

    for chunk in llm.stream(prompt):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            full_output += chunk.content

    print("\n\n===== END PLAN =====\n")

    plan_lines = [
        line.strip()
        for line in full_output.split("\n")
        if line.strip()
    ]

    return {"plan": plan_lines}


# -----------------------------------------------------
# Solver Node (FINAL ANSWER ONLY)
# -----------------------------------------------------

def solver_node(state: AgentState):
    plan_text = "\n".join(state["plan"])

    prompt = f"""
Follow this plan internally to solve the problem.

Problem:
{state["problem"]}

Plan:
{plan_text}

IMPORTANT:
- Do NOT show reasoning.
- Only output the final answer clearly.
"""

    print("===== FINAL ANSWER (Streaming) =====\n")

    full_output = ""

    for chunk in llm.stream(prompt):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            full_output += chunk.content

    print("\n\n===== END FINAL =====\n")

    return {"final_answer": full_output.strip()}


# -----------------------------------------------------
# Build LangGraph
# -----------------------------------------------------

workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("solver", solver_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "solver")
workflow.add_edge("solver", END)

graph = workflow.compile()

# -----------------------------------------------------
# Run Graph
# -----------------------------------------------------

graph.invoke({
    "problem": "Can you explain the paper attention is all you need?",
    "depth": "deep"
})