import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

RND = np.random.RandomState(42)

def random_timestamp(start, n):
    base = pd.to_datetime(start)
    secs = RND.randint(0, 60*60*24*30, size=n)
    return (base + pd.to_timedelta(secs, unit='s')).astype(str)

def generate_synthetic(n=50000):
    payer_ids = [f'payer_{i}' for i in range(2000)]
    payee_ids = [f'payee_{i}' for i in range(3000)]
    merchant_cats = ['grocery','recharge','utilities','travel','food','fashion']
    device_os = ['android','ios']
    channel = ['app','scan_qr','intent','collect']
    auth = ['pin','biometric','otp']

    ts = random_timestamp('2025-09-01', n)
    amount = np.round(np.exp(RND.normal(3.5, 1.5, size=n)), 2)
    payer = RND.choice(payer_ids, size=n)
    payee = RND.choice(payee_ids, size=n)
    mcat = RND.choice(merchant_cats, size=n)
    mid = ['merchant_'+str(RND.randint(0,2000)) for _ in range(n)]
    did = ['device_'+str(RND.randint(0,10000)) for _ in range(n)]
    dos = RND.choice(device_os, size=n)
    ch = RND.choice(channel, size=n)
    at = RND.choice(auth, size=n)
    is_new_payee = RND.binomial(1, 0.12, size=n)

    hour = [int(pd.to_datetime(t).hour) for t in ts]
    prob = 0.0005 + (amount > 1000) * 0.01 + (np.isin(mcat, ['travel','fashion']))*0.002
    prob += np.array(hour) * 0.00005
    prob += is_new_payee * 0.003
    prob = np.clip(prob, 0, 0.2)
    labels = RND.binomial(1, prob)

    df = pd.DataFrame({
        'transaction_id': [f'tx_{i}' for i in range(n)],
        'timestamp': ts,
        'amount': amount,
        'payer_id': payer,
        'payee_id': payee,
        'merchant_category': mcat,
        'merchant_id': mid,
        'device_id': did,
        'device_os': dos,
        'channel': ch,
        'auth_type': at,
        'is_new_payee': is_new_payee,
        'label': labels
    })
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data/synthetic_transactions.csv')
    parser.add_argument('--n', type=int, default=50000)
    args = parser.parse_args()
    df = generate_synthetic(args.n)
    df.to_csv(args.out, index=False)
    print('Wrote', args.out, 'rows=', len(df))
