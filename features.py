import pandas as pd
import numpy as np
from dateutil import parser

CAT_FEATURES = ['merchant_category','device_os','channel','auth_type']

def preprocess_df(df: pd.DataFrame):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['amount_log'] = np.log1p(df['amount'].astype(float))
    df = df.sort_values('timestamp')
    df['payer_tx_count_24h'] = df.groupby('payer_id').cumcount()
    df['is_night'] = ((df['hour'] < 6) | (df['hour'] >= 22)).astype(int)
    for col in CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col + '_enc'] = df[col].factorize()[0]
        else:
            df[col + '_enc'] = 0
    feature_cols = [
        'amount_log', 'is_new_payee', 'payer_tx_count_24h', 'is_night',
        'hour', 'dayofweek'
    ] + [c + '_enc' for c in CAT_FEATURES]
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0
    return df, feature_cols

def featurize(tx_json: dict):
    df = pd.DataFrame([tx_json])
    if 'timestamp' not in df.columns:
        df['timestamp'] = pd.Timestamp.now()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['amount'] = pd.to_numeric(df.get('amount', 0))
    df['amount_log'] = np.log1p(df['amount'])

    is_new = int(df['is_new_payee'].iloc[0]) if 'is_new_payee' in df.columns else 0
    df['is_new_payee'] = is_new

    # take optional fields from payload if present
    payer_count = int(tx_json.get('payer_tx_count_24h', 0))
    prev_amount = float(tx_json.get('prev_tx_amount_24h', 0.0))

    df['payer_tx_count_24h'] = payer_count
    df['prev_tx_amount_24h'] = prev_amount
    df['is_night'] = int(((df['hour'] < 6) | (df['hour'] >= 22)).iloc[0])

    def encode_cat(val):
        return abs(hash(str(val))) % 1000

    merchant_enc = encode_cat(tx_json.get('merchant_category'))
    device_enc = encode_cat(tx_json.get('device_os'))
    channel_enc = encode_cat(tx_json.get('channel'))
    auth_enc = encode_cat(tx_json.get('auth_type'))

    fv = [
        float(df['amount_log'].iloc[0]),
        int(df['is_new_payee'].iloc[0]),
        int(df['payer_tx_count_24h'].iloc[0]),
        int(df['is_night'].iloc[0]),
        int(df['hour'].iloc[0]),
        int(df['dayofweek'].iloc[0]),
        merchant_enc,
        device_enc,
        channel_enc,
        auth_enc,
        float(df['prev_tx_amount_24h'].iloc[0])
    ]
    names = [
        'amount_log','is_new_payee','payer_tx_count_24h','is_night','hour','dayofweek',
        'merchant_category_enc','device_os_enc','channel_enc','auth_type_enc','prev_tx_amount_24h'
    ]
    return fv, names

