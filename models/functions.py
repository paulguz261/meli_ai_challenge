import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

import models.config as cnf

def read_train_data(file_path):
    """Read data from a CSV file."""
    df = pd.read_csv(file_path)
    return df



class SafeLogTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that applies a logarithmic transformation to numeric columns in a DataFrame,
    replacing zeros with a small epsilon value to avoid log(0) issues.
    """
    def __init__(self, columns=None, epsilon=1e-9):
        self.columns = columns
        self.epsilon = epsilon

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.select_dtypes(include=[np.number]).columns.tolist()
        return self

    def transform(self, X):
        X = X.copy().astype(float)
        X = np.log(X.replace(0, self.epsilon))
        return X


class CardinalityReducer(BaseEstimator, TransformerMixin):
    """
    A transformer that reduces the cardinality of categorical columns in a DataFrame
    by replacing less frequent categories with a placeholder value.
    This is useful for handling high-cardinality categorical features in machine learning models.
    """
    def __init__(self, top_n=5, placeholder='otros'):
        self.top_n = top_n
        self.placeholder = placeholder
        self.top_categories_ = {}

    def fit(self, X, y=None):
        X = X.copy()
        for col in X.select_dtypes(include='object'):
            self.top_categories_[col] = X[col].value_counts().nlargest(self.top_n).index.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        for col, top_vals in self.top_categories_.items():
            X[col] = X[col].where(X[col].isin(top_vals), self.placeholder)
        return X
    
class BasicPreprocessor(BaseEstimator, TransformerMixin):
    """ A basic preprocessor that drops rows with any missing values and removes specified columns."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X = X.dropna(axis=0, how='any')
        X = X.drop(columns=['id', 'label'], errors='ignore')  # safe drop
        return X
    