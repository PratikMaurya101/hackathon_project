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
# Centered Selector
# ==================================

left, center, right = st.columns([1, 2, 1])

with center:

    analysis_type = st.segmented_control(
        "Analysis Type",
        [
            "CO₂",
            "Energy",
            "Cost"
        ],
        default="CO₂"
    )

st.write("")

# ==================================
# CO₂ Analysis
# ==================================

if analysis_type == "CO₂":

    st.subheader("🌍 Carbon Emissions")

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
        "Reduction",
        f"{co2_reduction:.1f}%"
    )

# ==================================
# Energy Analysis
# ==================================

elif analysis_type == "Energy":

    st.subheader("⚡ Energy Consumption")

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
        "Reduction",
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
        "Reduction",
        f"{cost_reduction:.1f}%"
    )