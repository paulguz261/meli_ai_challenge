import agents.keys as keys
import agents.ai_config as ai_config
from agents.flow import run_log_analysis_flow
import pandas as pd
import os
import json

def read_test_data() -> str:
    """
    Reads test data from a file and returns it as a JSON string.
    """
    test_data_path = ai_config.TEST_DATA_FILEPATH
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data file not found at {test_data_path}")
    
    df = pd.read_csv(test_data_path).head(5)
    json_data = df.to_dict(orient="records")

    # Serialize
    json_string = json.dumps(json_data, indent=2)

    
    return json_string

if __name__ == "__main__":
    # Read test data
    logs_json = read_test_data()
    print(logs_json)
    print("END OF LOGS")
    # Run the flow
    decisions = run_log_analysis_flow(logs_json)
   
    # Print the decisions made by the second agent
    print("Decisions made by the second agent:")
    print(decisions)