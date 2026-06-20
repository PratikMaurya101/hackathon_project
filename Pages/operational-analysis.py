import streamlit as st
import pandas as pd
import numpy as np

# ----------------------------------
# Page Config
# ----------------------------------

st.set_page_config(
    page_title="Operational Analysis",
    layout="wide"
)

st.title("📊 Operational Analysis")

# ==================================
# Dummy Data
# ==================================

days = np.arange(1, 31)

df = pd.DataFrame(
    {
        "Day": days,

        "Baseline_CO2":
            np.random.normal(120, 5, 30),

        "Optimized_CO2":
            np.random.normal(95, 4, 30),

        "Baseline_kWh":
            np.random.normal(550, 20, 30),

        "Optimized_kWh":
            np.random.normal(430, 15, 30),

        "Baseline_Cost":
            np.random.normal(170, 8, 30),

        "Optimized_Cost":
            np.random.normal(135, 6, 30)
    }
)

# ==================================
# KPI Calculations
# ==================================

co2_reduction = (
    (df["Baseline_CO2"].sum()
     - df["Optimized_CO2"].sum())
    / df["Baseline_CO2"].sum()
) * 100

energy_reduction = (
    (df["Baseline_kWh"].sum()
     - df["Optimized_kWh"].sum())
    / df["Baseline_kWh"].sum()
) * 100

cost_reduction = (
    (df["Baseline_Cost"].sum()
     - df["Optimized_Cost"].sum())
    / df["Baseline_Cost"].sum()
) * 100

# ==================================
# KPI Cards
# ==================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "🌍 CO₂ Reduction",
        f"{co2_reduction:.1f}%"
    )

with col2:
    st.metric(
        "⚡ Energy Reduction",
        f"{energy_reduction:.1f}%"
    )

with col3:
    st.metric(
        "💶 Cost Reduction",
        f"{cost_reduction:.1f}%"
    )

st.divider()

# ==================================
# Navigation State
# ==================================

if "analysis_type" not in st.session_state:
    st.session_state.analysis_type = "CO₂"

# ==================================
# Centered Navigation Buttons
# ==================================

left, center, right = st.columns([2, 3, 2])

with center:

    b1, b2, b3 = st.columns(3)

    with b1:
        if st.button(
            "🌍 CO₂",
            use_container_width=True
        ):
            st.session_state.analysis_type = "CO₂"

    with b2:
        if st.button(
            "⚡ Energy",
            use_container_width=True
        ):
            st.session_state.analysis_type = "Energy"

    with b3:
        if st.button(
            "💶 Cost",
            use_container_width=True
        ):
            st.session_state.analysis_type = "Cost"

analysis_type = st.session_state.analysis_type

st.write("")

# ==================================
# CO₂ Analysis
# ==================================

if analysis_type == "CO₂":

    st.subheader("🌍 Carbon Emissions Analysis")

    chart_df = df[
        [
            "Day",
            "Baseline_CO2",
            "Optimized_CO2"
        ]
    ].set_index("Day")

    st.line_chart(
        chart_df,
        height=500
    )

    st.metric(
        "CO₂ Reduction",
        f"{co2_reduction:.1f}%"
    )

# ==================================
# Energy Analysis
# ==================================

elif analysis_type == "Energy":

    st.subheader("⚡ Energy Consumption Analysis")

    chart_df = df[
        [
            "Day",
            "Baseline_kWh",
            "Optimized_kWh"
        ]
    ].set_index("Day")

    st.line_chart(
        chart_df,
        height=500
    )

    st.metric(
        "Energy Reduction",
        f"{energy_reduction:.1f}%"
    )

# ==================================
# Cost Analysis
# ==================================

elif analysis_type == "Cost":

    st.subheader("💶 Cost Analysis")

    chart_df = df[
        [
            "Day",
            "Baseline_Cost",
            "Optimized_Cost"
        ]
    ].set_index("Day")

    st.line_chart(
        chart_df,
        height=500
    )

    st.metric(
        "Cost Reduction",
        f"{cost_reduction:.1f}%"
    )