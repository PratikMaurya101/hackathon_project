import pandas as pd

from database import ClimateDatabase


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MARCH_FILE = "data/heat_pump_snapshots_march.csv"
MAY_FILE = "data/heat_pump_snapshots_may.csv"


# --------------------------------------------------
# Helper
# --------------------------------------------------

def prepare_dataframe(df):

    if "op_mode" not in df.columns:
        df["op_mode"] = None

    df["last_seen_at"] = pd.to_datetime(
        df["last_seen_at"]
    )

    df["last_seen_at"] = (
        df["last_seen_at"]
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )

    # Add power consumption columns
    if "baseline_power_con" not in df.columns:
        df["baseline_power_con"] = None

    if "model_power_con" not in df.columns:
        df["model_power_con"] = None

    return df


# --------------------------------------------------
# Load CSV Files
# --------------------------------------------------

march_df = pd.read_csv(MARCH_FILE)

may_df = pd.read_csv(MAY_FILE)


# --------------------------------------------------
# Clean Data
# --------------------------------------------------

march_df = prepare_dataframe(march_df)

may_df = prepare_dataframe(may_df)


# --------------------------------------------------
# Merge Data
# --------------------------------------------------

combined_df = pd.concat(
    [march_df, may_df],
    ignore_index=True
)


# --------------------------------------------------
# Keep Only Schema Columns
# --------------------------------------------------

schema_columns = [
    "id",
    "last_seen_at",

    "status_temperature_in_celsius",
    "status_humidity_in_percent",

    "status_target_temperature_in_celsius",
    "status_temperature_outside_in_celsius",

    "status_carbon_dioxide_in_ppm",

    "status_air_flow_supply_in_percent",
    "status_air_flow_return_in_percent",

    "status_is_heating_required",
    "status_is_cooling_required",

    "baseline_power_con",
    "model_power_con",

    "op_mode"
]

combined_df = combined_df[schema_columns]


# --------------------------------------------------
# Insert Into SQLite
# --------------------------------------------------

db = ClimateDatabase()

# Debug message
print(combined_df.columns.tolist())
print(combined_df.head())

db.insert_dataframe(combined_df)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print(f"Imported {len(combined_df)} rows successfully.")

print(
    f"Database now contains "
    f"{db.get_row_count()} rows."
)