# 🌦 Australia Weather Analytics Dashboard

A full-stack data analytics web application built on 10 years of daily Australian weather data (145,460 records across 49 locations). Features an interactive dashboard with live charts and a machine learning model that predicts whether it will rain tomorrow.

---

## 🚀 Features

- **Summary Stats** — Total records, date range, locations, average temperature, rainfall, and rainy day percentage
- **Monthly Rainfall Chart** — Average rainfall per month across all locations or filtered by city
- **Yearly Temperature Trend** — Max and min temperature trends from 2007 to 2017
- **Rainy Days by Location** — Top 15 locations ranked by percentage of rainy days
- **Rain Prediction** — Enter today's weather conditions and get a real-time ML prediction on whether it will rain tomorrow, with probability score
- **Location Filter** — Filter all charts by any of the 49 Australian cities in the dataset

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, Flask, Flask-CORS |
| Machine Learning | scikit-learn (Random Forest Classifier) |
| Data Processing | pandas, NumPy |
| Frontend | HTML, CSS, JavaScript |
| Charts | Chart.js |
| Dataset | weatherAUS.csv (Kaggle) |

---

## 📁 Project Structure

```
weather-project/
├── backend/
│   ├── app.py              # Flask API server
│   ├── requirements.txt    # Python dependencies
│   └── weatherAUS.csv      # Dataset (145,460 rows)
└── frontend/
    └── index.html          # Dashboard UI (charts + prediction form)
```

---

## ⚙️ How to Run Locally

### Prerequisites
- Python 3.10 or higher
- pip (comes with Python)

### Step 1 — Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/weather-analytics.git
cd weather-analytics
```

### Step 2 — Set up the backend
```bash
cd backend
python -m venv venv

# Activate virtual environment:
# Windows:
venv\Scripts\Activate.ps1
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### Step 3 — Run the Flask server
```bash
python app.py
```
The server starts on `http://127.0.0.1:5000`. It may take 15–30 seconds on first run while it loads and trains the model on the dataset.

### Step 4 — Open the frontend
Open `frontend/index.html` in your browser (double-click or use Live Server in VS Code).

The dashboard will load automatically and pull data from the running Flask API.

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Server status, row count, model accuracy |
| `/api/summary` | GET | Overall dataset statistics |
| `/api/locations` | GET | List of all 49 locations |
| `/api/monthly-rainfall` | GET | Average rainfall per month (optional `?location=`) |
| `/api/temp-trend` | GET | Yearly avg max/min temperature (optional `?location=`) |
| `/api/rain-by-location` | GET | % rainy days for top 15 locations |
| `/api/predict` | POST | Rain prediction from user-supplied conditions |

### Example prediction request
```json
POST /api/predict
{
  "MinTemp": 18,
  "MaxTemp": 22,
  "Rainfall": 5,
  "Humidity3pm": 85,
  "Pressure3pm": 1005,
  "WindGustSpeed": 45,
  "RainToday": "Yes",
  "Location": "Sydney"
}
```

### Example prediction response
```json
{
  "prediction": "Yes",
  "probability_rain": 0.823,
  "model_accuracy": 0.8493
}
```

---

## 🤖 Machine Learning Model

- **Algorithm:** Random Forest Classifier (150 trees, max depth 12)
- **Features used:** MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustSpeed, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm, Temp9am, Temp3pm, Location, WindGustDir, WindDir9am, WindDir3pm, RainToday
- **Target:** RainTomorrow (Yes/No)
- **Train/Test split:** 80/20
- **Test accuracy: 84.93%**
- Missing values handled by median imputation (numeric) and label encoding (categorical)

---

## 📊 Dataset

- **Source:** [Kaggle — Rain in Australia](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)
- **Records:** 145,460 daily observations
- **Locations:** 49 Australian cities
- **Date range:** November 2007 — June 2017
- **Columns:** 23 (Date, Location, temperatures, rainfall, evaporation, sunshine, wind, humidity, pressure, cloud cover, RainToday, RainTomorrow)

---

## 📌 Key Insights from the Data

- **Cairns** has the highest percentage of rainy days among all 49 locations
- Average max temperature across Australia is **23.22°C**
- Only **21.92%** of all days recorded rain
- Rainfall is highest in the early months of the year (January–March)
- Max temperatures have remained relatively stable from 2007 to 2017

---

## 👤 Author

**Yoga**
- Dataset from Kaggle: weatherAUS.csv
- Project Domain: Data Analytics

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
