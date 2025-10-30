import joblib, argparse, json
from flask import Flask, request, jsonify
import numpy as np

app = Flask('predict_service')
model = None
features = ['duration','tests_run','failures_last_24h','changed_files']

@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json()
    x = [payload.get(f,0) for f in features]
    proba = float(model.predict_proba([x])[0][1])
    return jsonify({'predicted_failure_probability': proba})

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='model/model.pkl')
    p.add_argument('--port', type=int, default=8000)
    args = p.parse_args()
    model = joblib.load(args.model)
    app.run(host='0.0.0.0', port=args.port)
