import os
import json
import pandas as pdJ
from typing import List, Dict
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field
import agents.tools.ai_tools as ai_tools
import agents.keys as keys

ingest_agent_prompt = """
# IDENTITY AND ROLE

You are an AI agent that is part of a cybersecurity area, you are part of decision chan in which log data is analyzed to determine if it contains anomalies from attacks.

for this task you will recive the data as a json rows, and you will have tools to process the data and get a result.
the result will be then passed on to another agent that will determine actions

# TOOLS
You have access to the following tools:
{tools}

# information to analyze
{log_records}
"""

class IngestAgentInput(BaseModel):
    "Input class for Ingest Agent"
    log_records: str = Field(description="The log records to be analyzed, provided as a JSON string.")

predictor = ai_tools.ModelPredictor()

@tool("predict_anomaly_model")
def predict_tool(inputs: str) -> str:
    """
    Uses a trained model to predict whether access logs contain anomalies.
    """
    try:
        logs = json.loads(inputs)
        # logs = json.loads(inputs["inputs"])
        predictions = predictor.predict_logs(logs)
        predictions = json.dumps(predictions, indent=2)
        return predictions
    except Exception as e:
        return json.dumps({"error": str(e)})

    