"""Shared settings + feature code. Used by BOTH train_models.py and app.py
so training and prediction always build features the exact same way."""
import numpy as np
import pandas as pd

CITIES = {
    "Delhi": (28.61, 77.21), "Mumbai": (19.08, 72.88), "Kolkata": (22.57, 88.36),
    "Chennai": (13.08, 80.27), "Bengaluru": (12.97, 77.59), "Hyderabad": (17.39, 78.49),
    "Pune": (18.52, 73.86), "Ahmedabad": (23.02, 72.57), "Jaipur": (26.91, 75.79),
    "Lucknow": (26.85, 80.95), "Chandigarh": (30.73, 76.78), "Bhopal": (23.26, 77.41),
    "Patna": (25.59, 85.14), "Guwahati": (26.14, 91.74),
}

MAX_HORIZON = 14
DAILY_VARS = "temperature_2m_mean,relative_humidity_2m_mean,surface_pressure_mean"
HOURLY_VAR = "temperature_2m"   # used to pick out the 12 PM temperature
NOON_HOUR = 12                  # local time; the model now predicts the temperature at 12:00

FEATURES = [
    "doy_sin", "doy_cos", "temp_now",
    "temp_lag1", "temp_lag2", "temp_lag3", "temp_lag7", "temp_roll7",
    "humidity", "pressure", "pressure_change", "lat", "lon",
    "noon_now", "noon_lag1", "noon_lag2", "noon_roll7", "noon_minus_mean",
]

def daily_frame_from_api(js):
    """Open-Meteo JSON (with both "daily" and "hourly") -> one row per day: the daily
    columns plus temp_noon = the temperature at 12:00 local time that day."""
    df = pd.DataFrame(js["daily"])
    df["time"] = pd.to_datetime(df["time"])
    h = pd.DataFrame({"t": pd.to_datetime(js["hourly"]["time"]), "v": js["hourly"][HOURLY_VAR]})
    h = h[h["t"].dt.hour == NOON_HOUR]
    noon = pd.Series(h["v"].values, index=h["t"].dt.normalize())
    df["temp_noon"] = df["time"].map(noon)
    return df

def add_features(df, lat, lon):
    """df needs columns: time (datetime), temperature_2m_mean,
    relative_humidity_2m_mean, surface_pressure_mean, temp_noon."""
    df = df.copy()
    doy = df["time"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    t = df["temperature_2m_mean"]
    df["temp_now"] = t
    for lag in [1, 2, 3, 7]:
        df[f"temp_lag{lag}"] = t.shift(lag)
    df["temp_roll7"] = t.rolling(7).mean()
    df["humidity"] = df["relative_humidity_2m_mean"]
    df["pressure"] = df["surface_pressure_mean"]
    df["pressure_change"] = df["surface_pressure_mean"].diff(1)
    n = df["temp_noon"]
    df["noon_now"] = n
    df["noon_lag1"] = n.shift(1)
    df["noon_lag2"] = n.shift(2)
    df["noon_roll7"] = n.rolling(7).mean()
    df["noon_minus_mean"] = n - t
    df["lat"] = lat
    df["lon"] = lon
    return df
