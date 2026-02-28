import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

class HabitatReport(BaseModel):
    executive_summary: str
    health_score: float
    key_insights: list[str]
    recommended_actions: list[str]
    confidence: float

def generate_habitat_report(df, anomalies, optimization_plan):
    if os.getenv("GENAI_MODE", "mock") == "mock":
        # Professional mock for portfolio/demo (no API cost)
        return {
            "executive_summary": "🚨 Critical anomaly detected in Zone B3 (soil moisture -28%). AI recommends immediate drone patrol. Habitat health: 87%.",
            "health_score": 87.4,
            "key_insights": ["Soil moisture drop indicates drought stress", "CO2 spike correlates with anomaly"],
            "recommended_actions": ["Deploy 3 patrol units to Zone B3", "Increase sensor sampling rate 2x"],
            "confidence": 0.93
        }

    client = OpenAI()
    prompt = f"""You are a senior ecologist with 20+ years experience.
    Analyze the habitat data below and produce a structured report.

    Sensor Data Summary: {df.describe().to_string()}
    Detected Anomalies: {anomalies[anomalies['anomaly']==-1].to_string()}
    Optimized Deployment Plan: {optimization_plan}

    Use chain-of-thought reasoning. Return ONLY valid JSON matching this Pydantic model:
    {HabitatReport.model_json_schema()}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.1
    )
    return HabitatReport.model_validate_json(response.choices[0].message.content)
