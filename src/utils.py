import os
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest

def load_model(model_path: str = "models/isolation_forest.pkl") -> IsolationForest:
    """
    Loads the trained Isolation Forest anomaly detection model. If it doesn't exist,
    initializes a new model.

    Args:
        model_path (str): Path to the saved pickle model file.

    Returns:
        IsolationForest: The loaded or newly initialized model instance.
    """
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception:
            pass
    return IsolationForest(contamination=0.08, random_state=42)

def plot_sensor_data(df: pd.DataFrame) -> go.Figure:
    """
    Creates an interactive multi-line chart of IoT environmental metrics using Plotly.

    Args:
        df (pd.DataFrame): Dataframe containing the sensor logs.

    Returns:
        go.Figure: The constructed Plotly figure object.
    """
    fig = go.Figure()
    
    # Sort by timestamp to ensure proper line plotting
    df_sorted = df.sort_values(by="timestamp")
    
    # Add temperature
    fig.add_trace(go.Scatter(
        x=df_sorted["timestamp"], 
        y=df_sorted["temperature"],
        mode="lines",
        name="Temperature (°C)",
        line=dict(color="#FF4B4B", width=2)
    ))
    
    # Add humidity
    fig.add_trace(go.Scatter(
        x=df_sorted["timestamp"], 
        y=df_sorted["humidity"],
        mode="lines",
        name="Humidity (%)",
        line=dict(color="#00D2B4", width=2)
    ))
    
    # Add CO2 (rescaled for visualization)
    fig.add_trace(go.Scatter(
        x=df_sorted["timestamp"], 
        y=df_sorted["co2"] / 10.0,
        mode="lines",
        name="CO2 (ppm / 10)",
        line=dict(color="#FFAA00", width=2)
    ))
    
    fig.update_layout(
        title="🌿 Environmental Telemetry (Live)",
        xaxis_title="Timeline",
        yaxis_title="Metric Value",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F0F2F6"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig
