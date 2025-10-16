import joblib, json, logging
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from features import featurize

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
app = Flask(__name__, template_folder='templates')

artifact = joblib.load('model_artifact.pkl')
model = artifact['model']
feature_cols = artifact.get('feature_cols', None)
explainer = artifact.get('shap_explainer', None)

def top_k_shap(shap_values, feature_names, k=3):
    idx = np.argsort(-np.abs(shap_values))[:k]
    return [(feature_names[i], float(shap_values[i])) for i in idx]

@app.route('/api/predict', methods=['POST'])
def predict():
    payload = request.get_json()
    logging.info("INCOMING PAYLOAD: %s", json.dumps(payload))
    fv, names = featurize(payload)
    logging.info("FEAT NAMES: %s", names)
    logging.info("FEAT VALUES: %s", fv)

    runtime_map = dict(zip(names, fv))
    if feature_cols:
        row = {}
        for col in feature_cols:
            row[col] = runtime_map.get(col, 0)
        X = pd.DataFrame([row], columns=feature_cols)
    else:
        X = pd.DataFrame([runtime_map], columns=names)

    logging.info("MODEL INPUT DF:\\n%s", X.to_string(index=False))

    try:
        proba = float(model.predict_proba(X)[:, 1][0])
    except Exception as e:
        logging.exception("PREDICTION ERROR")
        return jsonify({'error': str(e)}), 500

    response = {
        'transaction_id': payload.get('transaction_id'),
        'fraud_probability': proba,
        'features': dict(zip(X.columns.tolist(), X.iloc[0].tolist()))
    }

    if explainer is not None:
        try:
            sv = explainer.shap_values(X)
            if isinstance(sv, list):
                shap_vals = np.array(sv[1])[0]
            else:
                shap_vals = np.array(sv)[0]
            response['shap_top_features'] = top_k_shap(shap_vals, X.columns.tolist(), k=3)
        except Exception as e:
            response['shap_error'] = str(e)

    logging.info("RETURNING proba=%s", proba)
    return jsonify(response)

# Serve splash page
@app.route('/', methods=['GET'])
def index():
    return send_from_directory('templates', 'index.html')

# Demo page
@app.route('/demo', methods=['GET'])
def demo_page():
    return send_from_directory('templates', 'demo.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
