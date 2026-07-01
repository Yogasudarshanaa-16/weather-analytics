"""
Weather Analytics Backend
--------------------------
Flask API that reads weatherAUS.csv, exposes analytics endpoints,
and serves a simple Random Forest 'will it rain tomorrow' prediction.

Run:
    pip install -r requirements.txt
    python app.py
Then it serves on http://localhost:5000
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

app = Flask(__name__)
CORS(app)  # allow the frontend (different origin/port) to call this API

CSV_PATH = os.path.join(os.path.dirname(__file__), "weatherAUS.csv")

# ------------------------------------------------------------------
# Load + clean data once at startup
# ------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

NUMERIC_COLS = [
    "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
    "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "Humidity9am",
    "Humidity3pm", "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm",
    "Temp9am", "Temp3pm"
]

# ------------------------------------------------------------------
# Train a small model once at startup (kept in memory)
# ------------------------------------------------------------------
model_df = df.dropna(subset=["RainTomorrow"]).copy()
for col in NUMERIC_COLS:
    model_df[col] = model_df[col].fillna(model_df[col].median())

encoders = {}
for col in ["Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday"]:
    model_df[col] = model_df[col].fillna("Unknown")
    le = LabelEncoder()
    model_df[col] = le.fit_transform(model_df[col])
    encoders[col] = le

target_le = LabelEncoder()
model_df["RainTomorrow"] = target_le.fit_transform(model_df["RainTomorrow"])

FEATURES = NUMERIC_COLS + ["Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday"]
X = model_df[FEATURES]
y = model_df["RainTomorrow"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)
MODEL_ACCURACY = accuracy_score(y_test, clf.predict(X_test))


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "rows": len(df), "model_accuracy": round(MODEL_ACCURACY, 4)})


@app.route("/api/locations")
def locations():
    return jsonify(sorted(df["Location"].dropna().unique().tolist()))


@app.route("/api/summary")
def summary():
    """Overall dataset summary stats."""
    return jsonify({
        "total_records": int(len(df)),
        "date_range": {
            "start": df["Date"].min().strftime("%Y-%m-%d"),
            "end": df["Date"].max().strftime("%Y-%m-%d"),
        },
        "locations": int(df["Location"].nunique()),
        "rain_days_pct": round(float((df["RainToday"] == "Yes").mean() * 100), 2),
        "avg_max_temp": round(float(df["MaxTemp"].mean()), 2),
        "avg_min_temp": round(float(df["MinTemp"].mean()), 2),
        "avg_rainfall_mm": round(float(df["Rainfall"].mean()), 2),
        "avg_humidity_3pm": round(float(df["Humidity3pm"].mean()), 2),
    })


@app.route("/api/monthly-rainfall")
def monthly_rainfall():
    """Average rainfall per month, optionally filtered by location."""
    location = request.args.get("location")
    data = df if not location else df[df["Location"] == location]
    grouped = data.groupby("Month")["Rainfall"].mean().reindex(range(1, 13))
    return jsonify({
        "labels": ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
        "values": [None if pd.isna(v) else round(float(v), 2) for v in grouped.values],
    })


@app.route("/api/temp-trend")
def temp_trend():
    """Yearly average max/min temperature trend."""
    location = request.args.get("location")
    data = df if not location else df[df["Location"] == location]
    grouped = data.groupby("Year")[["MaxTemp", "MinTemp"]].mean().dropna()
    return jsonify({
        "labels": [int(y) for y in grouped.index],
        "max_temp": [round(float(v), 2) for v in grouped["MaxTemp"]],
        "min_temp": [round(float(v), 2) for v in grouped["MinTemp"]],
    })


@app.route("/api/rain-by-location")
def rain_by_location():
    """% of rainy days per location (top 15 by record count)."""
    top_locations = df["Location"].value_counts().head(15).index
    data = df[df["Location"].isin(top_locations)]
    grouped = data.groupby("Location")["RainToday"].apply(lambda s: (s == "Yes").mean() * 100)
    grouped = grouped.sort_values(ascending=False)
    return jsonify({
        "labels": grouped.index.tolist(),
        "values": [round(float(v), 2) for v in grouped.values],
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Predict RainTomorrow from user-supplied feature values.
    Expects JSON body with keys matching FEATURES (raw, human-readable values
    for categorical fields e.g. Location: "Sydney").
    """
    payload = request.get_json(force=True)
    row = {}
    try:
        for col in NUMERIC_COLS:
            row[col] = float(payload.get(col, model_df[col].median()))
        for col in ["Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday"]:
            raw_val = payload.get(col, "Unknown")
            le = encoders[col]
            row[col] = int(le.transform([raw_val])[0]) if raw_val in le.classes_ else 0
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    X_input = pd.DataFrame([row])[FEATURES]
    pred = clf.predict(X_input)[0]
    proba = clf.predict_proba(X_input)[0]
    label = target_le.inverse_transform([pred])[0]

    return jsonify({
        "prediction": label,
        "probability_rain": round(float(proba[list(target_le.classes_).index("Yes")]), 4)
                              if "Yes" in target_le.classes_ else None,
        "model_accuracy": round(MODEL_ACCURACY, 4),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
