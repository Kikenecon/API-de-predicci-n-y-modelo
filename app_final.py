
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from flask import Flask, request, jsonify
from pyngrok import ngrok
import pickle  # PARA GUARDAR EL MODELO

# 1️CARGAR Y PREPARAR DATOS
ruta = "car_sales_data.csv"
df = pd.read_csv(ruta)

X = df[['Year of manufacture', 'Mileage']]
y = df['Price']

#  DIVIDIR Y MEDIR CALIDAD
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

# MÉTRICAS
y_pred = modelo.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Error medio absoluto (MAE): {mae:.2f}")
print(f"Coeficiente R²: {r2:.4f}")
print(f"Esto significa que el modelo explica el {r2*100:.1f}% de los precios")

# GUARDAR EL MODELO
with open('modelo_carros.pkl', 'wb') as f:
    pickle.dump(modelo, f)

print("Modelo guardado como 'modelo_carros.pkl'")

#  SERVIDOR FLASK
app = Flask(__name__)

# CARGAR EL MODELO GUARDADO 
with open('modelo_carros.pkl', 'rb') as f:
    modelo_cargado = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict_price():
    data = request.get_json()
    year = float(data.get('year', 0))
    mileage = float(data.get('mileage', 0))

    nuevo_auto = pd.DataFrame({
        'Year of manufacture': [year],
        'Mileage': [mileage]
    })

    pred = modelo_cargado.predict(nuevo_auto)[0]
    return jsonify({'predicted_price': round(pred, 2)})

# NGROK
if __name__ == '__main__':
    public_url = ngrok.connect(5000)
    print(f"URL pública: {public_url}")
    print("API lista! Usa /predict")
    app.run(port=5000)

