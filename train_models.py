"""Run ONCE (re-run any time you want to retrain).
Downloads ~11 years of history for every city, trains one LightGBM model per
forecast day (1..14) on ALL cities together, saves them to ./models/"""
import json, os, time
import joblib, requests
import pandas as pd
from datetime import date, timedelta
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from common import CITIES, FEATURES, MAX_HORIZON, DAILY_VARS, HOURLY_VAR, add_features, daily_frame_from_api

os.makedirs("models", exist_ok=True)

# The archive API only accepts dates up to a few days ago, so stop 7 days short of today.
END_DATE = (date.today() - timedelta(days=7)).isoformat()

os.makedirs("cache", exist_ok=True)

def download_city(name, lat, lon):
    """Downloads one city's history. Saves it to ./cache so reruns skip finished cities,
    and waits + retries automatically if Open-Meteo's per-minute limit is hit."""
    path = f"cache/{name}_noon.csv"   # new cache name: these files also hold the 12 PM temperature
    if os.path.exists(path):
        print(f"{name}: using saved copy")
        d = pd.read_csv(path)
        d["time"] = pd.to_datetime(d["time"])
        return d
    for attempt in range(6):
        print(f"Downloading {name}...")
        r = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
            "latitude": lat, "longitude": lon,
            "start_date": "2015-01-01", "end_date": END_DATE,
            "daily": DAILY_VARS, "hourly": HOURLY_VAR, "timezone": "auto",
        }, timeout=120)
        if r.status_code == 429:
            print("  Rate limit hit - waiting 65 seconds, then retrying...")
            time.sleep(65)
            continue
        if not r.ok:
            raise SystemExit(f"API error {r.status_code}: {r.text}")
        d = daily_frame_from_api(r.json())
        d.to_csv(path, index=False)
        time.sleep(10)   # be gentle with the free API
        return d
    raise SystemExit("Still rate limited after several tries. Wait ~10 minutes and run again.")

frames = []
for name, (lat, lon) in CITIES.items():
    d = download_city(name, lat, lon)
    d = add_features(d, lat, lon)
    t = d["temp_noon"]                        # we now predict the 12 PM temperature
    for h in range(1, MAX_HORIZON + 1):
        d[f"delta_h{h}"] = t.shift(-h) - t      # change in 12 PM temp from today, h days ahead
    frames.append(d)

data = pd.concat(frames, ignore_index=True)
print(f"\nTotal rows: {len(data)} across {len(CITIES)} cities\n")

SPLIT = "2024-01-01"
params = dict(n_estimators=300, max_depth=6, learning_rate=0.05,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)

# Second model for the comparison: Random Forest, same features, same train/test split.
# (Kept modest in size so the 14 saved files stay small enough to load quickly.)
rf_params = dict(n_estimators=100, max_depth=10, min_samples_leaf=30,
                 n_jobs=-1, random_state=42)
metrics = {}

print(f"{'Day':<5}{'LightGBM MAE':<14}{'RandomForest MAE':<18}{'Persistence MAE':<17}")
for h in range(1, MAX_HORIZON + 1):
    col = f"delta_h{h}"
    d = data.dropna(subset=FEATURES + [col])
    train, test = d[d["time"] < SPLIT], d[d["time"] >= SPLIT]

    # 1) honest test: train on the past, score on 2024+ data neither model has seen
    m = LGBMRegressor(**params).fit(train[FEATURES], train[col])
    model_mae = mean_absolute_error(test[col], m.predict(test[FEATURES]))
    rf = RandomForestRegressor(**rf_params).fit(train[FEATURES], train[col])
    rf_mae = mean_absolute_error(test[col], rf.predict(test[FEATURES]))
    persistence_mae = test[col].abs().mean()   # guess "same 12 PM temp as today"
    metrics[str(h)] = {"model_mae": round(float(model_mae), 3),
                       "rf_mae": round(float(rf_mae), 3),
                       "persistence_mae": round(float(persistence_mae), 3)}
    print(f"{h:<5}{model_mae:<14.2f}{rf_mae:<18.2f}{persistence_mae:<17.2f}")

    # 2) final models for the website: retrain on everything
    final = LGBMRegressor(**params).fit(d[FEATURES], d[col])
    joblib.dump(final, f"models/model_h{h}.pkl")
    final_rf = RandomForestRegressor(**rf_params).fit(d[FEATURES], d[col])
    joblib.dump(final_rf, f"models/rf_h{h}.pkl", compress=3)

with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

avg = lambda k: sum(m[k] for m in metrics.values()) / len(metrics)
lgb_wins = sum(m["model_mae"] < m["rf_mae"] for m in metrics.values())
print(f"\nAverage MAE over 14 days -> LightGBM: {avg('model_mae'):.3f} | "
      f"RandomForest: {avg('rf_mae'):.3f} | Persistence: {avg('persistence_mae'):.3f}")
print(f"LightGBM beats Random Forest on {lgb_wins} of {len(metrics)} forecast days.")
print("\nDone. Models saved in ./models/ — now run: python app.py")
