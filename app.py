"""Local prediction server. Run with:  python app.py
Loads the saved LightGBM models, and for each request pulls that city's
recent OBSERVED weather, builds features, and runs the models itself."""
import json, os
import joblib, requests
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from common import CITIES, FEATURES, MAX_HORIZON, DAILY_VARS, HOURLY_VAR, add_features, daily_frame_from_api

app = Flask(__name__)
CORS(app)

models = {h: joblib.load(f"models/model_h{h}.pkl") for h in range(1, MAX_HORIZON + 1)}
with open("models/metrics.json") as f:
    metrics = json.load(f)

# Second model (Random Forest) used for the comparison. Optional: if the files aren't
# there yet (train_models.py not re-run), the server still works with LightGBM only.
rf_models = {}
if all(os.path.exists(f"models/rf_h{h}.pkl") for h in range(1, MAX_HORIZON + 1)):
    rf_models = {h: joblib.load(f"models/rf_h{h}.pkl") for h in range(1, MAX_HORIZON + 1)}
else:
    print("Random Forest models not found - re-run train_models.py to enable the comparison.")
print(f"Loaded {len(models)} LightGBM + {len(rf_models)} Random Forest models. Server ready.")

@app.route("/")
def home():
    return "ML forecast server is running."

@app.route("/cities")
def cities():
    return jsonify(CITIES)   # {name: [lat, lon]}

@app.route("/predict")
def predict():
    city = request.args.get("city", "")
    if city not in CITIES:
        return jsonify(error=f"Unknown city: {city}"), 400
    lat, lon = CITIES[city]

    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": lat, "longitude": lon, "daily": DAILY_VARS, "hourly": HOURLY_VAR,
            "past_days": 14, "forecast_days": 1, "timezone": "auto",
        }, timeout=30)
        r.raise_for_status()
    except Exception as e:
        return jsonify(error=f"Could not fetch recent weather: {e}"), 502

    df = daily_frame_from_api(r.json())
    today = df["time"].iloc[-1]
    df = df.iloc[:-1]            # drop today's row: it's partly a forecast, we only use observed days
    df = add_features(df, lat, lon)

    row = df[FEATURES].iloc[[-1]]
    if row.isna().values.any():
        return jsonify(error="Recent weather data incomplete, try again later."), 502

    base_temp = float(df["temp_noon"].iloc[-1])   # last observed 12 PM temperature
    base_date = df["time"].iloc[-1]

    forecast = []
    for h in range(1, MAX_HORIZON + 1):
        delta = float(models[h].predict(row)[0])
        item = {
            "date": (base_date + pd.Timedelta(days=h)).strftime("%Y-%m-%d"),
            "temp": round(base_temp + delta, 1),
            "typical_error": metrics[str(h)]["model_mae"],
            "persistence_error": metrics[str(h)]["persistence_mae"],
        }
        if rf_models and "rf_mae" in metrics[str(h)]:
            rf_delta = float(rf_models[h].predict(row)[0])
            item["rf_temp"] = round(base_temp + rf_delta, 1)
            item["rf_error"] = metrics[str(h)]["rf_mae"]
        forecast.append(item)

    return jsonify(city=city, based_on=base_date.strftime("%Y-%m-%d"),
                   today=today.strftime("%Y-%m-%d"), forecast=forecast)

if __name__ == "__main__":
    app.run(port=5000)
