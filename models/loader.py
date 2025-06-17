import os
import joblib
from models import config as cnf
import models.functions  # Needed so joblib can deserialize

def load_model(version):
    """Load a trained model from the specified version.
    Args:
        version (int): The version of the model to load.
    Returns:
        The loaded model.
    """
    model_path = os.path.join(cnf.MODEL_EXPORT_DIR, cnf.MODEL_BASE_NAME.format(version=str(version)))
    return joblib.load(model_path)