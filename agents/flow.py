"""
flow.py

This module defines a LangGraph pipeline for analyzing log data using two AI agents:
1. Ingest Agent - Predicts anomalies using a trained ML model.
2. Decision Agent - Suggests actions based on the predictions.

The flow is managed using LangGraph's StateGraph and can be executed via `run_log_analysis_flow`.
"""

import os
import json
import pandas as pd
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List

from agents.ingest_agent import predict_tool

# Define input/output state schema (simple key-value string map)
from typing import TypedDict

import agents.keys as keys
import agents.ai_config as ai_config

# Set API keys for OpenAI and Google Gemini
os.environ["OPENAI_API_KEY"] = keys.API_KEY_GPT_MELI
os.environ["GOOGLE_API_KEY"] = keys.API_KEY_GEMINI_MELI

class FlowState(TypedDict):
    """
    Shared state passed between LangGraph nodes.
    """
    logs: str  
    predictions: str  
    decisions: str  

class DecisionOutput(BaseModel):
    """
    Structure for the output of a single decision per log entry.
    """    
    id: int
    prediction: str
    action: str

class DecisionOutputList(BaseModel):
    """
    Structure for the overall decision output containing multiple decisions per log.
    """
    whole_analysis: str = Field(description = "A string containing the whole analysis of the decision output")
    analysis: List[DecisionOutput]
    

# PROMPTS
# Base prompt for both agents
agent_base_prompt = """
# IDENTITY AND ROLE
You are an AI agent that is part of a cybersecurity area, you are part of decision chain in which log data is analyzed to determine if it contains anomalies from attacks, and finally it gives actions to react to the attack detected.

"""

# Prompt for the ingest agent
ingest_agent_prompt = agent_base_prompt +"""
youre the first agent in the chain, you will receive logs and predict anomalies using a trained model available in your tools.
for this task you will recive the data, and you will have tools to process the data and get a result.
the result will be then passed on to another agent that will determine actions

# TOOLS
You have access to the following tools:
{tools}

use the tools to predict anomalies in the logs provided.

# information to analyze
{log_records}
"""

# Prompt for the decision agent
decision_agent_prompt = agent_base_prompt + """
youre the second agent in the chain, you will receive the results of the first agent and you will determine what actions to take based on the results.
results will be in the form of a string, and you will return a string with the actions to take.

please return the best course of action based on the results provided by the first agent.
some of the attachs you may encounter are:

- Generic: Broad category of attacks that do not fit into a specific pattern or exploit. Often used to test general system weaknesses.
- Exploits: Attacks that take advantage of specific software vulnerabilities to gain unauthorized access or control.
- Fuzzers: Tools or methods that send random or malformed data to programs to find vulnerabilities like crashes or unexpected behavior.
- DoS (Denial of Service): Attempts to make a system or service unavailable to its intended users by overwhelming it with traffic or requests.
- Reconnaissance: Pre-attack activities aimed at gathering information about a system, such as scanning for open ports or services.
- Analysis: Involves examining a system's behavior or configurations to uncover potential security weaknesses.
- Backdoor: Hidden methods for bypassing normal authentication to gain remote access to a system.
- Shellcode: Malicious code that gives an attacker control of a compromised machine, often used as a payload in exploits.
- Worms: Self-replicating malware that spreads across networks without user intervention, often consuming bandwidth or resources.

"""

# Define the base LLMs for both agents
base_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)
base_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

def format_decision_output(output: DecisionOutputList) -> str:
    """
    Convert structured decision output to a human-readable string.
    """  
    lines = [f"🧠 Overall analysis: {output.whole_analysis}", ""]
    
    for item in output.analysis:
        lines.append(
            f"• Log ID {item.id}: Prediction → '{item.prediction.upper()}', Suggested action → '{item.action.upper()}'"
        )
    
    return "\n".join(lines)

def format_tools_description(tools: list[BaseTool]) -> str:
    """
    Format tool descriptions for inclusion in prompts.
    """
    return "\n\n".join([f"- {tool.name}: {tool.description}\n Input arguments: {tool.args}" for tool in tools])

def ingest_node(state: FlowState):
    """
    Ingests logs and predicts anomalies using the provided tool.

    updates the state with predictions.
    Args:
        state (FlowState): The shared state containing logs and other data.
    """
    logs = state["logs"]
    # Asociate tools with the LLM
    tools = [predict_tool]
    ingest_llm = base_llm.bind_tools(tools)
    
    # Complete the prompt for the ingest agent
    prompt = ingest_agent_prompt.format(
        tools=format_tools_description([predict_tool]),
        log_records=logs
    )
    # Invoke the LLM with the prompt
    result = ingest_llm.invoke(prompt, config={"verbose": True})

    # Check if the result contains tool calls and execute them
    for tool_call in result.tool_calls:
        selected_tool = {"predict_anomaly_model": predict_tool}[tool_call["name"].lower()]
        # tool_msg = selected_tool.invoke(json.dumps(tool_call["args"]))
        tool_msg = selected_tool.invoke(tool_call["args"])
        
    # Parse the tool message as JSON
    preds = json.loads(tool_msg)

    if isinstance(result, str):
        raise ValueError(f"Error in prediction tool: {result}")
    
    print(f"Predictions: {preds}")
    return {"predictions": preds}


def decision_node(state: FlowState) -> FlowState:
    """
    Analyzes predictions from the ingest node and makes decisions based on them. for each log entry.
    and also returns a string with the whole analysis of the decision output.

    returns:
        FlowState: Updated state with decisions made by the agent.
    """
    decision_llm = base_llm.with_structured_output(DecisionOutputList)
    decision_prompt = decision_agent_prompt + f"Predictions from the first agent: {state['predictions']}"
    result = decision_llm.invoke(decision_prompt)

    parsed = format_decision_output(result)

    return {"decisions": parsed}

# Build the LangGraph
builder = StateGraph(FlowState)

builder.add_node("predict", ingest_node)
builder.add_node("decide", decision_node)

builder.set_entry_point("predict")
builder.add_edge("predict", "decide")
builder.add_edge("decide", END)

graph = builder.compile()

# Entry point function to run the flow
def run_log_analysis_flow(logs_json: str):
    """
    Runs the full LangGraph flow given a batch of logs in JSON string format.
    """
    result = graph.invoke({"logs": logs_json})
    return result["decisions"]
