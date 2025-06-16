import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data location
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
TRAIN_DATA_NAME = 'UNSW_NB15_training-set.csv'
TEST_DATA_NAME = 'UNSW_NB15_testing-set.csv'
TRAIN_DIR = os.path.join(DATA_DIR, TRAIN_DATA_NAME)
TEST_DIR = os.path.join(DATA_DIR, TEST_DATA_NAME)

# Model location
MODEL_BASE_NAME = "cybersecurity_model_v{version}.joblib"
MODEL_DIR = os.path.join(PROJECT_DIR, 'models')
MODEL_EXPORT_DIR = os.path.join(MODEL_DIR, 'exports')
