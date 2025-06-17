"""
ingest_agent.py

define elements/tools for the Ingest Agent

"""

import json
from langchain_core.tools import tool
import agents.tools.ai_tools as ai_tools

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

    