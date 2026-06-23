import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Yet to connect with db.

# ==========================================================
# READ EXCEL
# ==========================================================

hp_heat_interval_df = pd.read_excel(
    r"..."
)

# ==========================================================
# TIME COLUMN
# ==========================================================

hp_heat_interval_df['interval_start_time'] = pd.to_datetime(
    hp_heat_interval_df['interval_start_time']
)

hp_heat_interval_df.sort_values(
    ['device_id', 'interval_start_time'],
    inplace=True
)

# ==========================================================
# PRESSURE DIFFERENCE
# ==========================================================

hp_heat_interval_df['pressure_difference'] = (
    hp_heat_interval_df['high_pressure_highest_in_bar']
    -
    hp_heat_interval_df['low_pressure_highest_in_bar']
)

# ==========================================================
# PARAMETERS
# ==========================================================

parameters = {

    'temp': 'median_temperature_in_celsius',

    'return_temp': 'median_return_temperature_in_celsius',

    'humidity': 'median_humidity_in_percent',

    'CO2': 'median_carbon_dioxide_in_ppm',

    'pressure_difference': 'pressure_difference'
}

# ==========================================================
# WEIGHTS
# ==========================================================

weights = {

    'temp': 0.2,

    'return_temp': 0.2,

    'humidity': 0.1,

    'CO2': 0.1,

    'pressure_difference': 0.4
}

# ==========================================================
# ROLLING FEATURES (1 hour = 4 samples)
# ==========================================================

window_size = 4

for parameter, column in parameters.items():

    hp_heat_interval_df[f'{parameter}_mean_1h'] = (
        hp_heat_interval_df
        .groupby('device_id')[column]
        .transform(lambda x: x.rolling(window_size, min_periods=1).mean())
    )

    hp_heat_interval_df[f'{parameter}_std_1h'] = (
        hp_heat_interval_df
        .groupby('device_id')[column]
        .transform(lambda x: x.rolling(window_size, min_periods=1).std())
    )

    hp_heat_interval_df[f'{parameter}_change'] = (
        hp_heat_interval_df
        .groupby('device_id')[column]
        .diff()
    )

hp_heat_interval_df.fillna(0, inplace=True)

# ==========================================================
# DEVICE STATISTICS
# ==========================================================

device_statistics = {}

for device in hp_heat_interval_df['device_id'].unique():

    subset = hp_heat_interval_df[
        hp_heat_interval_df['device_id'] == device
    ]

    device_statistics[device] = {}

    for parameter, column in parameters.items():

        device_statistics[device][parameter] = {

            'mean': subset[column].mean(),

            'std': subset[column].std(),

            'min': subset[column].min(),

            'max': subset[column].max()

        }

# ==========================================================
# HEALTH INDEX
# ==========================================================

def calculate_health_index(device, reading):

    HI = 0

    for parameter, value in reading.items():

        mean = device_statistics[device][parameter]['mean']
        std = device_statistics[device][parameter]['std']

        if std == 0 or np.isnan(std):

            z_score = 0

        else:

            z_score = abs((value - mean) / std)

        HI += weights[parameter] * z_score

    return HI

# ==========================================================
# CALCULATE HEALTH INDEX FOR ALL ROWS
# ==========================================================

health_indices = []

for _, row in hp_heat_interval_df.iterrows():

    device = row['device_id']

    reading = {

        'temp': row['median_temperature_in_celsius'],

        'return_temp': row['median_return_temperature_in_celsius'],

        'humidity': row['median_humidity_in_percent'],

        'CO2': row['median_carbon_dioxide_in_ppm'],

        'pressure_difference': row['pressure_difference']

    }

    HI = calculate_health_index(device, reading)

    health_indices.append(HI)

hp_heat_interval_df['health_index'] = health_indices

# ==========================================================
# HEALTH STATUS
# ==========================================================

def classify_health(HI):

    if HI < 1:
        return "Healthy"

    elif HI < 2:
        return "Watch"

    elif HI < 3:
        return "Warning"

    else:
        return "Critical"

hp_heat_interval_df['health_status'] = (
    hp_heat_interval_df['health_index']
    .apply(classify_health)
)

# ==========================================================
# FEATURES
# ==========================================================

X = hp_heat_interval_df[[
    'median_temperature_in_celsius',
    'median_return_temperature_in_celsius',
    'median_humidity_in_percent',
    'median_carbon_dioxide_in_ppm',
    'pressure_difference',

    'temp_mean_1h',
    'return_temp_mean_1h',
    'humidity_mean_1h',
    'CO2_mean_1h',
    'pressure_difference_mean_1h',

    'temp_change',
    'return_temp_change',
    'humidity_change',
    'CO2_change',
    'pressure_difference_change'
]]

y = hp_heat_interval_df['health_index']

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================================
# MODEL
# ==========================================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predictions
)

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

print("\nFeature Importance\n")
print(
    importance.sort_values(
        ascending=False
    )
)

# ==========================================================
# OVERALL STATS
# ==========================================================

print("\nMean Absolute Error =", round(mae,4))

print(
    "Average Health Index =",
    round(
        hp_heat_interval_df['health_index'].mean(),
        3
    )
)

print(
    "Maximum Health Index =",
    round(
        hp_heat_interval_df['health_index'].max(),
        3
    )
)

# ==========================================================
# PREDICTION FUNCTION
# ==========================================================

def predict_machine_status(model,
                           device,
                           temp,
                           return_temp,
                           humidity,
                           CO2,
                           pressure_difference):

    reading = {

        'temp': temp,

        'return_temp': return_temp,

        'humidity': humidity,

        'CO2': CO2,

        'pressure_difference': pressure_difference

    }

    rule_HI = calculate_health_index(device, reading)

    new_data = pd.DataFrame([{

        'median_temperature_in_celsius': temp,

        'median_return_temperature_in_celsius': return_temp,

        'median_humidity_in_percent': humidity,

        'median_carbon_dioxide_in_ppm': CO2,

        'pressure_difference': pressure_difference,

        'temp_mean_1h': temp,
        'return_temp_mean_1h': return_temp,
        'humidity_mean_1h': humidity,
        'CO2_mean_1h': CO2,
        'pressure_difference_mean_1h': pressure_difference,

        'temp_change': 0,
        'return_temp_change': 0,
        'humidity_change': 0,
        'CO2_change': 0,
        'pressure_difference_change': 0

    }])

    ml_HI = model.predict(new_data)[0]

    print("\n====================")
    print("Device:", device)
    print("====================")

    print("Rule HI =", round(rule_HI,3))

    print("ML HI =", round(ml_HI,3))

    print(
        "Machine Status =",
        classify_health(ml_HI)
    )

# ==========================================================
# USER SELECTS MACHINE
# ==========================================================

devices = hp_heat_interval_df['device_id'].unique()

print("\nAvailable Machines:\n")

for i, device in enumerate(devices):

    print(f"{i+1}. {device}")

choice = int(input("\nSelect machine number: "))

selected_device = devices[choice-1]

# ==========================================================
# USER INPUT
# ==========================================================

temp = float(input("Temperature: "))
return_temp = float(input("Return temperature: "))
humidity = float(input("Humidity: "))
CO2 = float(input("CO2: "))
pressure_difference = float(input("Pressure difference: "))

# ==========================================================
# PREDICT
# ==========================================================

predict_machine_status(
    model,
    selected_device,
    temp,
    return_temp,
    humidity,
    CO2,
    pressure_difference
)