import pandas as pd
import numpy as np
import sys
import os
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report, confusion_matrix

import models.config as cnf
import models.functions as func

def create_pipeline(df):
    """
    Create a machine learning pipeline.
    This function constructs a pipeline that includes preprocessing steps for both numeric and categorical features,
    followed by a Gradient Boosting Classifier.
    """
    
    df_clean = func.BasicPreprocessor().fit_transform(df)

    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns

    num_pipeline = Pipeline([
        ("safe_log", func.SafeLogTransformer(columns=numeric_cols)),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("cardinality_reducer", func.CardinalityReducer(top_n=5, placeholder='otros')),
        ("encoder", OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num",num_pipeline,numeric_cols),
        ("cat",cat_pipeline,categorical_cols)
    ]
    )

    model_pipeline = make_pipeline(
        func.BasicPreprocessor(),
        preprocessor,
        GradientBoostingClassifier()
    )
    
    return model_pipeline


def fit_pipeline(X_train, y_train):
    """Fit the pipeline to the training data."""
    df = X_train.copy()
   
    model_pipeline = create_pipeline(df)
    model_pipeline.fit(df, y_train)
    
    return model_pipeline

def save_pipeline(pipeline, filename):
    """Save the fitted pipeline to a file."""
    model_path = os.path.join(cnf.MODEL_EXPORT_DIR, filename)
    joblib.dump(pipeline, model_path)
    print(f"Pipeline saved to {model_path}")

def model_metrics(pipeline, X, y):
    """
    Evaluate the model's performance using various metrics.
    This function predicts the labels for the given features and calculates accuracy, precision, recall, and F1-score.
    It also prints a classification report and confusion matrix.
    Args:
        pipeline (Pipeline): The fitted machine learning pipeline.
        X (pd.DataFrame): The features to predict.
        y (pd.Series): The true labels.
    """
    y_pred = pipeline.predict(X)

    # Calculate accuracy
    accuracy = accuracy_score(y, y_pred)
    # dict_accuracy_models["reg_log_ajustado"] = accuracy
    print(f'Accuracy: {accuracy:.4f}')

    #calculo Precision usado=1
    prec = precision_score(y, y_pred, average='macro')
    
    #calculo Recall nuevo = 1
    rec = recall_score(y, y_pred, average='macro')
    
    #calculo F1-Score
    f1 = f1_score(y, y_pred, average='macro')
    
    model_performance = pd.DataFrame.from_dict({
    'Accuracy': [accuracy],
    'Recall': [rec],
    'Precision': [prec],
    'F1-Score': [f1]
    })

    print(model_performance)
    print(classification_report(y, y_pred))

def train_model():
    """
    Train a machine learning model using the training data.
    This function reads the training data, splits it into training and testing sets,
    fits a machine learning pipeline to the training data, evaluates the model's performance,
    and saves the fitted pipeline to a file.
    """
    df = func.read_train_data(cnf.TRAIN_DIR)
    X = df.drop(columns=["attack_cat"])
    y = df["attack_cat"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)


    pipe_fit = fit_pipeline(X_train, y_train)
    model_metrics(pipe_fit, X_test, y_test)

    save_pipeline(pipe_fit, cnf.MODEL_BASE_NAME.format(version=str(0)))

if __name__ == "__main__":
    train_model()