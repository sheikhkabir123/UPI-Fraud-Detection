# add imports
import joblib
import numpy as np
from flask import Flask, request, jsonify
from features import featurize  # your featurize function returning (fv, names)
import shap

app = Flask(__name__)

# load artifact saved by train.py
artifact = joblib.load('model_artifact.pkl')
model = artifact['model']
feature_cols = artifact['feature_cols']
explainer = artifact.get('shap_explainer', None)
shap_background = artifact.get('shap_background', None)

def top_k_shap(shap_values, feature_names, k=3):
    # shap_values: 1D array aligned with feature_names
    idx = np.argsort(-np.abs(shap_values))[:k]
    return [(feature_names[i], float(shap_values[i])) for i in idx]

@app.route('/api/predict', methods=['POST'])
def predict():
    payload = request.get_json()
    fv, names = featurize(payload)  # ensure this returns features in same order as model expects
    try:
        proba = model.predict_proba([fv])[:, 1][0]
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    response = {'transaction_id': payload.get('transaction_id', None),
                'fraud_probability': float(proba),
                'features': dict(zip(names, fv))}

    # compute shap values if explainer available
    if explainer is not None:
        # for tree explainer we can call shap_values directly
        # shap_values shape: (n_classes, n_features) for multiclass; for binary-class tree explainer, returns array of shape (n_features,)
        try:
            # Use background if needed to speed or stabilize explainer (not required for TreeExplainer usually)
            sv = explainer.shap_values(np.array([fv]))
            # shap_values can be list for multiclass; handle common binary-case
            if isinstance(sv, list):
                # take the shap values for class 1 (fraud)
                shap_vals = np.array(sv[1])[0]
            else:
                shap_vals = np.array(sv)[0]
            response['shap_top_features'] = top_k_shap(shap_vals, names, k=3)
        except Exception as e:
            response['shap_error'] = str(e)

    return jsonify(response)
if __name__ == "__main__":
    # start Flask dev server (only for local testing/demo)
    # use host 0.0.0.0 so other devices on your LAN can also reach it
    app.run(host="0.0.0.0", port=5000, debug=True)

# Demo frontend route (serves templates/demo.html)
from flask import render_template
@app.route('/demo', methods=['GET'])
def demo_page():
    return render_template('demo.html')

# Serve demo HTML directly (robust even if template folder not configured)
from flask import send_from_directory

@app.route('/demo', methods=['GET'])
def demo_page():
    return send_from_directory('templates', 'demo.html')
