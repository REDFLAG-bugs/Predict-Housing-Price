from flask import Flask, request, jsonify

import joblib


def create_app(model, scaler):
    app = Flask(__name__)

    @app.route('/predict', method=['Post'])
    def predict():
        data = request.get_json(force=True)
        data_transformed = scaler.transform([data])
        prediction = model.predict(data_transformed)
        return jsonify({'prediction':prediction[0]})
    return app

def save_model(model, scaler, model_path='model.pk1', scaler_path='scaler.pk1'):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

def load_model(model_path='model.pk1', scaler_path='scaler.pk1'):
    model = joblib.load(model_path)
    scaler= joblib.load(scaler_path)
    return model, scaler
