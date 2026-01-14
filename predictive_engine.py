# modules/predictive_engine.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

class PredictiveEngine:
    """
    Moteur prédictif LIGHT – Hackathon Safe
    (sans XGBoost)
    """

    def __init__(self):
        self.model = None
        self.model_performance = {}

    def prepare_training_data(self, df: pd.DataFrame, target_col: str):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if target_col not in df.columns:
            raise ValueError("Variable cible absente")

        if target_col in numeric_cols:
            numeric_cols.remove(target_col)

        X = df[numeric_cols].fillna(0)
        y = df[target_col]

        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self, X_train, y_train, model_type="random_forest"):
        if model_type == "logistic":
            model = LogisticRegression(max_iter=1000)
        else:
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )

        model.fit(X_train, y_train)
        self.model = model
        return model

    def predict(self, X):
        if self.model is None:
            raise ValueError("Modèle non entraîné")

        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)

        self.model_performance = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
        }

        return self.model_performance

    def save_model(self, path: str):
        if self.model:
            joblib.dump(self.model, path)

    def load_model(self, path: str):
        self.model = joblib.load(path)
