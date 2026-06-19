import streamlit as st
import pandas as pd
import numpy as np
import ollama

# Page configuration
st.set_page_config(
    page_title="Climate Control Dashboard",
    layout="wide"
)

st.title("Climate Control Dashboard")

# -------------------------
# Placeholder Graph
# -------------------------

data = pd.DataFrame({
    "Time": range(20),
    "Value": np.random.randint(20, 100, 20)
})

st.line_chart(data.set_index("Time"))

st.divider()

# -------------------------
# Sensor Information
# -------------------------

temperature = 24
humidity = 58
weather = "Cloudy"
power = 1.8

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="🌡️ Temperature",
        value=f"{temperature}°C"
    )

with col2:
    st.metric(
        label="💧 Humidity",
        value=f"{humidity}%"
    )

with col3:
    st.metric(
        label="☁️ Weather",
        value=weather
    )

with col4:
    st.metric(
        label="⚡ Power Consumption",
        value=f"{power} kW"
    )

st.divider()

# -------------------------
# Qwen 3.5 Analysis
# -------------------------

st.subheader("🤖 AI Climate Analysis")

prompt = f"""
Current room conditions:

Temperature: {temperature}°C
Humidity: {humidity}%
Weather: {weather}
Power Consumption: {power} kW

Provide a short analysis of the room conditions and
recommend any actions if needed.
"""

try:
    with st.spinner("Generating analysis..."):
        response = ollama.chat(
            model="qwen3.5:9b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

    ai_response = response["message"]["content"]

    st.info(ai_response)

except Exception as e:
    st.error(f"Could not connect to Ollama: {e}")
