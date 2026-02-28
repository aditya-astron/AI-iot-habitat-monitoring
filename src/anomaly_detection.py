import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
from pathlib import Path

class AnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(contamination=0.08, random_state=42)
        self.model_path = Path("models/isolation_forest.pkl")

    def train(self, df: pd.DataFrame):
        features = df[["temperature", "humidity", "soil_moisture", "co2", "light", "motion"]]
        self.model.fit(features)
        self.model_path.parent.mkdir(exist_ok=True)
        joblib.dump(self.model, self.model_path)
        return "Model trained & saved"

    def load_model(self):
        if self.model_path.exists():
            self.model = joblib.load(self.model_path)

    def detect(self, df: pd.DataFrame):
        features = df[["temperature", "humidity", "soil_moisture", "co2", "light", "motion"]]
        scores = self.model.predict(features)
        df = df.copy()
        df["anomaly"] = scores
        df["risk_score"] = -self.model.decision_function(features)
        return df
