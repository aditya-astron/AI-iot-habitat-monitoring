import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sensor_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generates synthetic IoT sensor data representing space habitat interior/exterior conditions.

    Args:
        n_samples (int): Number of sensor data points to generate.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: A DataFrame containing time-series environmental measurements.
    """
    np.random.seed(seed)
    dates = [datetime.now() - timedelta(minutes=i*15) for i in range(n_samples)]
    return pd.DataFrame({
        "timestamp": dates,
        "temperature": np.random.normal(28.0, 4.0, n_samples),
        "humidity": np.random.normal(65.0, 12.0, n_samples),
        "soil_moisture": np.random.normal(45.0, 15.0, n_samples),
        "co2": np.random.normal(420.0, 80.0, n_samples),
        "light": np.random.normal(800.0, 300.0, n_samples),
        "motion": np.random.poisson(2, n_samples)
    })
