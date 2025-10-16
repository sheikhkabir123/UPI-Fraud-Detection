import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import classification_report, precision_recall_curve, auc
from imblearn.over_sampling import RandomOverSampler
from sklearn.calibration import CalibratedClassifierCV
import shap

def preprocess_df(df):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['amount_log'] = np.log1p(df['amount'].astype(float))
    df = df.sort_values('timestamp')
    df['payer_tx_count_24h'] = df.groupby('payer_id').cumcount()
    df['is_night'] = ((df['hour'] < 6) | (df['hour'] >= 22)).astype(int)

    for col in ['merchant_category','device_os','channel','auth_type']:
        df[col] = df.get(col, '').astype(str)
        df[col + '_enc'] = df[col].factorize()[0]

    if 'prev_tx_amount_24h' not in df.columns:
        df['prev_tx_amount_24h'] = 0.0

    feature_cols = [
        'amount_log','is_new_payee','payer_tx_count_24h','is_night',
        'hour','dayofweek',
        'merchant_category_enc','device_os_enc','channel_enc','auth_type_enc',
        'prev_tx_amount_24h'
    ]
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0
    return df, feature_cols

def train(data_path, out_path):
    df = pd.read_csv(data_path)
    df, feature_cols = preprocess_df(df)
    X = df[feature_cols]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # oversample minority class to ~5% in training set
    ros = RandomOverSampler(sampling_strategy=0.05, random_state=42)
    X_res, y_res = ros.fit_resample(X_train, y_train)

    model = LGBMClassifier(n_estimators=1000, learning_rate=0.05, max_depth=9)
    model.fit(
        X_res, y_res,
        eval_set=[(X_test, y_test)],
        callbacks=[early_stopping(stopping_rounds=50)]
    )

    # calibrate probabilities using hold-out test set
    calibrator = CalibratedClassifierCV(model, cv='prefit', method='isotonic')
    calibrator.fit(X_test, y_test)

    preds = calibrator.predict_proba(X_test)[:,1]
    precision, recall, _ = precision_recall_curve(y_test, preds)
    pr_auc = auc(recall, precision)
    print('PR AUC (calibrated):', pr_auc)
    try:
        print(classification_report(y_test, (preds>0.01).astype(int)))
    except Exception:
        pass

    explainer = shap.TreeExplainer(model)
    artifact = {
        'model': calibrator,
        'raw_model': model,
        'feature_cols': feature_cols,
        'shap_explainer': explainer
    }
    joblib.dump(artifact, out_path)
    print('Saved artifact to', out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', default='model_artifact.pkl')
    args = parser.parse_args()
    train(args.data, args.out)
