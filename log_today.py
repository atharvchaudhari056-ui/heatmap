"""Predict TODAY's 12 PM temperature for every city with the trained LightGBM
model and store it in temperature_log.txt, one line per date.

Run it once a day:   python log_today.py

Put this file in the same folder as common.py and the models/ folder.
Running it again on the same day replaces that day's line (no duplicates).
"""
import os
import time
import joblib
import requests
import pandas as pd
from common import CITIES, FEATURES, DAILY_VARS, HOURLY_VAR, add_features, daily_frame_from_api

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "temperature_log.txt")

# The models predict the change from the last OBSERVED day (yesterday).
# So the 1-day-ahead model (h=1) gives today's 12 PM temperature.
model = joblib.load(os.path.join(BASE_DIR, "models", "model_h1.pkl"))


def predict_today(lat, lon):
    """Returns (today's date string, predicted 12 PM temp in C)."""
    r = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": lat, "longitude": lon, "daily": DAILY_VARS, "hourly": HOURLY_VAR,
        "past_days": 14, "forecast_days": 1, "timezone": "auto",
    }, timeout=30)
    r.raise_for_status()

    df = daily_frame_from_api(r.json())
    today = df["time"].iloc[-1]
    df = df.iloc[:-1]                      # drop today's row: use observed days only
    df = add_features(df, lat, lon)

    row = df[FEATURES].iloc[[-1]]
    if row.isna().values.any():
        raise ValueError("recent weather data incomplete")

    base_temp = float(df["temp_noon"].iloc[-1])
    delta = float(model.predict(row)[0])
    return today.strftime("%Y-%m-%d"), round(base_temp + delta, 1)


def main():
    today_str = None
    parts = []
    for name, (lat, lon) in CITIES.items():
        try:
            date_str, temp = predict_today(lat, lon)
            today_str = today_str or date_str
            parts.append(f"{name}: {temp}")
            print(f"{name}: {temp} C")
        except Exception as e:
            parts.append(f"{name}: N/A")
            print(f"{name}: failed ({e})")
        time.sleep(0.3)                    # be gentle with the free API

    if today_str is None:
        raise SystemExit("No predictions could be made - nothing saved. Check your internet and try again.")

    new_line = f"{today_str} | " + " | ".join(parts)

    # keep older days, replace today's line if it already exists, keep sorted by date
    lines = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    lines = [ln for ln in lines if not ln.startswith(today_str)]
    lines.append(new_line)
    lines.sort()

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nSaved to {LOG_FILE}")


if __name__ == "__main__":
    main()
