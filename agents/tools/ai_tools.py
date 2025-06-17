"""
ai_tools.py

Define the logic for llm tools

"""
import os
import sys
import pandas as pd
from typing import List, Dict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)
from models.loader import load_model
from models.config import MODEL_BASE_NAME, MODEL_EXPORT_DIR

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Adjust this path to your actual project root if needed


class ModelPredictor():
    """A class to load a trained model and make predictions on log data."""

    def __init__(self, model_version: int = 0):
        model_path = os.path.join(MODEL_EXPORT_DIR, MODEL_BASE_NAME.format(version=model_version))
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.model = load_model(version=model_version)
    def predict_logs(self, logs: List[Dict]) -> List[Dict]:
        df = pd.DataFrame(logs)
        predictions = self.model.predict(df)
        return [
            {"id": log.get("id", idx), "prediction": pred}
            for idx, (log, pred) in enumerate(zip(logs, predictions))
        ]