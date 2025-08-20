
from flask import Flask, render_template, request, jsonify
import joblib, json, numpy as np, pandas as pd, os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
META_PATH = os.path.join(os.path.dirname(__file__), "model_meta.json")

model = None
meta = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
if os.path.exists(META_PATH):
    meta = json.load(open(META_PATH))

FEATURE_ORDER = (meta["numerical_features"] + meta["categorical_features"]) if meta else []

def predict_proba(payload: dict):
    df = pd.DataFrame([payload])
    # Ensure columns exist
    for k in FEATURE_ORDER:
        if k not in df.columns:
            df[k] = 0
    df = df[FEATURE_ORDER]
    proba = model.predict_proba(df)[0,1]
    return float(proba)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    proba = predict_proba(data)
    return jsonify({"fraud_probability": proba, "fraud": proba >= 0.5})

@app.route("/predict", methods=["POST"])
def form_predict():
    form = request.form
    payload = {
        "amount": float(form.get("amount", 0)),
        "hour": int(form.get("hour", 12)),
        "dayofweek": int(form.get("dayofweek", 0)),
        "merchant_category": form.get("merchant_category", "p2p"),
        "device_change": int(form.get("device_change", 0)),
        "location_mismatch": int(form.get("location_mismatch", 0)),
        "is_blacklisted_merchant": int(form.get("is_blacklisted_merchant", 0)),
        "recent_chargebacks": int(form.get("recent_chargebacks", 0)),
        "user_tenure_months": int(form.get("user_tenure_months", 6)),
        "past_txn_count_7d": int(form.get("past_txn_count_7d", 10)),
        "avg_amount_30d": float(form.get("avg_amount_30d", 800.0)),
    }
    proba = predict_proba(payload)
    return render_template("index.html", score=round(proba*100,2), payload=payload, is_fraud= (proba>=0.5))

if __name__ == "__main__":
    app.run(debug=True)
