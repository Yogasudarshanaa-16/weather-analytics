"""
Weather Analytics Backend (Render-compatible, no pandas)
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
import csv
import os
import statistics
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)
CORS(app)

CSV_PATH = os.path.join(os.path.dirname(__file__), "weatherAUS.csv")

def load_data():
    rows = []
    with open(CSV_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def safe_float(val):
    try:
        return float(val)
    except:
        return None

print("Loading data...")
ALL_ROWS = load_data()
print(f"Loaded {len(ALL_ROWS)} rows")

NUMERIC_COLS = ["MinTemp","MaxTemp","Rainfall","Evaporation","Sunshine",
                "WindGustSpeed","WindSpeed9am","WindSpeed3pm","Humidity9am",
                "Humidity3pm","Pressure9am","Pressure3pm","Cloud9am","Cloud3pm",
                "Temp9am","Temp3pm"]
CAT_COLS = ["Location","WindGustDir","WindDir9am","WindDir3pm","RainToday"]

# Compute medians for imputation
medians = {}
for col in NUMERIC_COLS:
    vals = [safe_float(r[col]) for r in ALL_ROWS if safe_float(r[col]) is not None]
    medians[col] = statistics.median(vals) if vals else 0.0

# Build ML dataset
encoders = {}
label_classes = {}
for col in CAT_COLS:
    unique = sorted(set(r.get(col,'Unknown') or 'Unknown' for r in ALL_ROWS))
    encoders[col] = {v:i for i,v in enumerate(unique)}
    label_classes[col] = unique

rain_enc = {"Yes":1,"No":0}

X, y = [], []
for r in ALL_ROWS:
    if not r.get("RainTomorrow") in ("Yes","No"):
        continue
    row_x = []
    for col in NUMERIC_COLS:
        v = safe_float(r.get(col))
        row_x.append(v if v is not None else medians[col])
    for col in CAT_COLS:
        val = r.get(col,'Unknown') or 'Unknown'
        row_x.append(encoders[col].get(val, 0))
    X.append(row_x)
    y.append(rain_enc[r["RainTomorrow"]])

print("Training model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)
MODEL_ACCURACY = accuracy_score(y_test, clf.predict(X_test))
print(f"Model ready. Accuracy: {MODEL_ACCURACY:.4f}")

@app.route("/api/health")
def health():
    return jsonify({"status":"ok","rows":len(ALL_ROWS),"model_accuracy":round(MODEL_ACCURACY,4)})

@app.route("/api/locations")
def locations():
    locs = sorted(set(r["Location"] for r in ALL_ROWS if r.get("Location")))
    return jsonify(locs)

@app.route("/api/summary")
def summary():
    dates = [r["Date"] for r in ALL_ROWS if r.get("Date")]
    rain_days = sum(1 for r in ALL_ROWS if r.get("RainToday")=="Yes")
    max_temps = [safe_float(r["MaxTemp"]) for r in ALL_ROWS if safe_float(r.get("MaxTemp")) is not None]
    rainfalls = [safe_float(r["Rainfall"]) for r in ALL_ROWS if safe_float(r.get("Rainfall")) is not None]
    humidity = [safe_float(r["Humidity3pm"]) for r in ALL_ROWS if safe_float(r.get("Humidity3pm")) is not None]
    return jsonify({
        "total_records": len(ALL_ROWS),
        "date_range": {"start": min(dates), "end": max(dates)},
        "locations": len(set(r["Location"] for r in ALL_ROWS if r.get("Location"))),
        "rain_days_pct": round(rain_days/len(ALL_ROWS)*100, 2),
        "avg_max_temp": round(sum(max_temps)/len(max_temps), 2),
        "avg_rainfall_mm": round(sum(rainfalls)/len(rainfalls), 2),
        "avg_humidity_3pm": round(sum(humidity)/len(humidity), 2),
    })

@app.route("/api/monthly-rainfall")
def monthly_rainfall():
    location = request.args.get("location")
    monthly = defaultdict(list)
    for r in ALL_ROWS:
        if location and r.get("Location") != location:
            continue
        try:
            month = int(r["Date"].split("-")[1])
            val = safe_float(r.get("Rainfall"))
            if val is not None:
                monthly[month].append(val)
        except:
            pass
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    values = [round(sum(monthly[i])/len(monthly[i]),2) if monthly[i] else None for i in range(1,13)]
    return jsonify({"labels":months,"values":values})

@app.route("/api/temp-trend")
def temp_trend():
    location = request.args.get("location")
    yearly_max = defaultdict(list)
    yearly_min = defaultdict(list)
    for r in ALL_ROWS:
        if location and r.get("Location") != location:
            continue
        try:
            year = int(r["Date"].split("-")[0])
            mx = safe_float(r.get("MaxTemp"))
            mn = safe_float(r.get("MinTemp"))
            if mx is not None: yearly_max[year].append(mx)
            if mn is not None: yearly_min[year].append(mn)
        except:
            pass
    years = sorted(yearly_max.keys())
    return jsonify({
        "labels": years,
        "max_temp": [round(sum(yearly_max[y])/len(yearly_max[y]),2) for y in years],
        "min_temp": [round(sum(yearly_min[y])/len(yearly_min[y]),2) for y in years],
    })

@app.route("/api/rain-by-location")
def rain_by_location():
    loc_total = defaultdict(int)
    loc_rain = defaultdict(int)
    for r in ALL_ROWS:
        loc = r.get("Location")
        if not loc: continue
        loc_total[loc] += 1
        if r.get("RainToday") == "Yes":
            loc_rain[loc] += 1
    top = sorted(loc_total.keys(), key=lambda l: loc_total[l], reverse=True)[:15]
    pcts = {l: round(loc_rain[l]/loc_total[l]*100,2) for l in top}
    sorted_locs = sorted(pcts.keys(), key=lambda l: pcts[l], reverse=True)
    return jsonify({"labels":sorted_locs,"values":[pcts[l] for l in sorted_locs]})

@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    row_x = []
    for col in NUMERIC_COLS:
        row_x.append(float(payload.get(col, medians[col])))
    for col in CAT_COLS:
        val = payload.get(col,'Unknown') or 'Unknown'
        row_x.append(encoders[col].get(val,0))
    pred = clf.predict([row_x])[0]
    proba = clf.predict_proba([row_x])[0]
    label = "Yes" if pred==1 else "No"
    return jsonify({
        "prediction": label,
        "probability_rain": round(float(proba[1]),4),
        "model_accuracy": round(MODEL_ACCURACY,4),
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
