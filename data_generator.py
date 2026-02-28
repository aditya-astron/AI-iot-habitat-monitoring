import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sensor_data(n_samples=1000, seed=42):
    np.random.seed(seed)
    dates = [datetime.now() - timedelta(minutes=i*15) for i in range(n_samples)]
    return pd.DataFrame({
        "timestamp": dates,
        "temperature": np.random.normal(28, 4, n_samples),
        "humidity": np.random.normal(65, 12, n_samples),
        "soil_moisture": np.random.normal(45, 15, n_samples),
        "co2": np.random.normal(420, 80, n_samples),
        "light": np.random.normal(800, 300, n_samples),
        "motion": np.random.poisson(2, n_samples)
    })
