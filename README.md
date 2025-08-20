
# UPI Fraud Detection — Flask + ML (Starter)

End-to-end demo that trains a simple model on synthetic UPI-like transactions and serves a web form with real-time fraud scoring.

## 🧰 Tech
- Flask (frontend + API)
- scikit-learn (Logistic Regression pipeline)
- Joblib (model persistence)
- Chart.js (client-side visualization)

## ▶️ Run locally
```bash
cd upi-fraud-detection
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py      # creates model.pkl
python app.py        # serves http://127.0.0.1:5000
```

## 🔗 API
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"amount":1200,"hour":23,"dayofweek":5,"merchant_category":"shopping","device_change":1,"location_mismatch":0,"is_blacklisted_merchant":0,"recent_chargebacks":0,"user_tenure_months":6,"past_txn_count_7d":30,"avg_amount_30d":1500}'
```

## ⚠️ Notes
- Data is **synthetic** and for learning only — do not use in production.
- Improve with real data, better models (XGBoost), feature store, thresholds, monitoring.
