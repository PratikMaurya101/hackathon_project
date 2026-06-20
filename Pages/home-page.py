import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

sys.path.append(str(ROOT_DIR))

import streamlit as st
from datetime import datetime

from dbms.database import ClimateDatabase


# ----------------------------------
# Page Configuration
# ----------------------------------

st.set_page_config(
    page_title="Climate Control Dashboard",
    layout="wide"
)

db = ClimateDatabase()

st.title("🏢 Climate Control Dashboard")

# ==================================
# Date Range Selection
# ==================================

st.subheader("📈 Indoor Temperature Trend")

col1, col2 = st.columns(2)

with col1:

    start_date = st.date_input(
        "Start Date"
    )

    start_time = st.time_input(
        "Start Time"
    )

with col2:

    end_date = st.date_input(
        "End Date"
    )

    end_time = st.time_input(
        "End Time"
    )

# ==================================
# Convert To Datetime Strings
# ==================================

start_datetime = datetime.combine(
    start_date,
    start_time
).strftime("%Y-%m-%d %H:%M:%S")

end_datetime = datetime.combine(
    end_date,
    end_time
).strftime("%Y-%m-%d %H:%M:%S")

# ==================================
# Fetch Data
# ==================================

df = db.get_data_between(
    start_datetime,
    end_datetime
)

# ==================================
# Graph
# ==================================

if not df.empty:

    graph_df = df[
        [
            "last_seen_at",
            "status_temperature_in_celsius"
        ]
    ].copy()

    graph_df = graph_df.set_index(
        "last_seen_at"
    )

    st.line_chart(
        graph_df
    )

else:

    st.warning(
        "No data found for selected time range."
    )

st.divider()

# ==================================
# Metrics
# ==================================

if not df.empty:

    latest = df.iloc[-1]

    inside_temp = latest[
        "status_temperature_in_celsius"
    ]

    outside_temp = latest[
        "status_temperature_outside_in_celsius"
    ]

    humidity = latest[
        "status_humidity_in_percent"
    ]

    co2_ppm = latest[
        "status_carbon_dioxide_in_ppm"
    ]

    target_temp = latest[
        "status_target_temperature_in_celsius"
    ]

    # ----------------------------------
    # Infer Mode
    # ----------------------------------

    if target_temp > inside_temp:
        operation_mode = "Heating"
    else:
        operation_mode = "Cooling"

    st.subheader(
        "🏢 Current Building State"
    )

    row1_col1, row1_col2, row1_col3 = st.columns(3)

    with row1_col1:
        st.metric(
            "🌡️ Inside Temperature",
            f"{inside_temp:.1f} °C"
        )

    with row1_col2:
        st.metric(
            "🌤️ Outside Temperature",
            f"{outside_temp:.1f} °C"
        )

    with row1_col3:
        st.metric(
            "🫁 CO₂ Level",
            f"{int(co2_ppm)} ppm"
        )

    row2_col1, row2_col2, row2_col3 = st.columns(3)

    with row2_col1:
        st.metric(
            "💧 Humidity",
            f"{humidity:.1f}%"
        )

    with row2_col2:
        st.metric(
            "🎯 Target Temperature",
            f"{target_temp:.1f} °C"
        )

    with row2_col3:
        st.metric(
            "❄️ Operation Mode",
            operation_mode
        )

    st.divider()

    st.success(
        "HVAC system operating normally."
    )

else:

    st.info(
        "Select a time range containing data."
    )