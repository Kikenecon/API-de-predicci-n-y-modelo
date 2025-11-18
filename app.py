from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Cargar modelo (archivo 'modelo_carros.pkl' en el repo)
MODEL_PATH = 'modelo_carros.pkl'  # directo, porque Railway lo pone en el directorio de trabajo
with open(MODEL_PATH, 'rb') as f:
    modelo_cargado = pickle.load(f)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "API de predicción lista. Usa POST /predict"}), 200

@app.route('/predict', methods=['POST'])
def predict_price():
    data = request.get_json(force=True)
    year = float(data.get('year', 0))
    mileage = float(data.get('mileage', 0))

    nuevo_auto = pd.DataFrame({
        'Year of manufacture': [year],
        'Mileage': [mileage]
    })

    pred = modelo_cargado.predict(nuevo_auto)[0]
    return jsonify({'predicted_price': round(float(pred), 2)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
