
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

np.random.seed(42)

# Synthetic data generator for UPI-like transactions
N = 20000
amount = np.random.gamma(shape=2.0, scale=500.0, size=N)  # typical UPI amounts skewed
hour = np.random.randint(0,24,size=N)
dayofweek = np.random.randint(0,7,size=N)

merchant_category = np.random.choice(
    ["food","grocery","utilities","shopping","travel","entertainment","p2p"], size=N, p=[0.15,0.2,0.1,0.2,0.1,0.1,0.15]
)

device_change = np.random.binomial(1, 0.1, size=N)  # whether device differs from usual
location_mismatch = np.random.binomial(1, 0.08, size=N)
is_blacklisted_merchant = np.random.binomial(1, 0.02, size=N)
recent_chargebacks = np.clip(np.random.poisson(0.05, size=N),0,3)
user_tenure_months = np.clip(np.random.normal(18, 8, size=N), 1, None).astype(int)
past_txn_count_7d = np.clip(np.random.poisson(10, size=N),0,100)
avg_amount_30d = np.random.gamma(2.0, 400.0, size=N)

# Fraud probability (synthetic rule-based ground truth)
logit = (
    0.002*(amount-1000) +
    0.3*device_change +
    0.5*location_mismatch +
    1.2*is_blacklisted_merchant +
    0.15*(recent_chargebacks>0).astype(int) +
    0.001*(avg_amount_30d>1500).astype(int) +
    0.0005*(past_txn_count_7d>25).astype(int) +
    0.2*((hour<5) | (hour>22)) +
    0.1*(merchant_category=="shopping").astype(int) +
    0.15*(merchant_category=="travel").astype(int) -
    0.01*user_tenure_months
)
prob = 1/(1+np.exp(-logit))
y = (np.random.rand(N) < prob*0.7).astype(int)  # scale base rate ~1-3%

df = pd.DataFrame({
    "amount": amount.round(2),
    "hour": hour,
    "dayofweek": dayofweek,
    "merchant_category": merchant_category,
    "device_change": device_change,
    "location_mismatch": location_mismatch,
    "is_blacklisted_merchant": is_blacklisted_merchant,
    "recent_chargebacks": recent_chargebacks,
    "user_tenure_months": user_tenure_months,
    "past_txn_count_7d": past_txn_count_7d,
    "avg_amount_30d": avg_amount_30d.round(2),
    "is_fraud": y
})

os.makedirs("data", exist_ok=True)
df.to_csv("data/upi_synth.csv", index=False)

# Train/val split
X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]

num_cols = ["amount","hour","dayofweek","device_change","location_mismatch","is_blacklisted_merchant",
            "recent_chargebacks","user_tenure_months","past_txn_count_7d","avg_amount_30d"]
cat_cols = ["merchant_category"]

pre = ColumnTransformer([
    ("num", StandardScaler(with_mean=True, with_std=True), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

clf = Pipeline(steps=[
    ("pre", pre),
    ("model", LogisticRegression(max_iter=200, class_weight="balanced"))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
clf.fit(X_train, y_train)

y_proba = clf.predict_proba(X_test)[:,1]
y_pred = (y_proba>0.5).astype(int)

print("AUC:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred))

# Save model and metadata
joblib.dump(clf, "model.pkl")

meta = {
    "numerical_features": num_cols,
    "categorical_features": cat_cols,
    "classes": ["legit","fraud"]
}
import json
json.dump(meta, open("model_meta.json","w"))
print("Saved model.pkl and model_meta.json")
