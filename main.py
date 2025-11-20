from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# =====================================================
# CARGA DEL MODELO Y DEL SCALER
# =====================================================
scaler = pickle.load(open("scaler.pkl", "rb"))

# compile=False para evitar error de keras.metrics.mse
model = load_model("modelo_lstm.h5", compile=False)

# Crear la app
app = FastAPI()

# =====================================================
# MODELO DE ENTRADA EXACTO: 10 valores
# =====================================================
class PredInput(BaseModel):
    valores: list  # lista de 10 valores (últimas 10 semanas)


# =====================================================
# HOME
# =====================================================
@app.get("/")
def home():
    return {"message": "API LSTM de predicción de dengue funcionando correctamente"}


# =====================================================
# ENDPOINT DE PREDICCIÓN
# =====================================================
@app.post("/predict")
def predict(data: PredInput):

    valores = data.valores

    # Validación exacta
    if len(valores) != 10:
        return {
            "error": "Debe enviar EXACTAMENTE una lista de 10 valores (las últimas 10 semanas)"
        }

    # Convertir a numpy
    X = np.array(valores).reshape(-1, 1)

    # Escalar igual que en el entrenamiento
    X_scaled = scaler.transform(X)

    # Dar forma (1, 10, 1) para LSTM
    X_scaled = X_scaled.reshape(1, 10, 1)

    # Predecir
    pred_scaled = model.predict(X_scaled)

    # Invertir escala
    pred = scaler.inverse_transform(pred_scaled)

    # Convertir a número normal
    pred_value = float(pred[0][0])

    return {
        "prediccion": round(pred_value, 2),
        "mensaje": "Predicción de casos de dengue para la próxima semana"
    }

#ejecutar
#.\venv\Scripts\activate
#python -m uvicorn main:app --reload
#http://127.0.0.1:8000/docs
#{
  #"valores": [14,28,26,27,59,35,33,22,17,13]
#}
#{
  #"valores": [10,10,10,10,10,10,10,10,10,10]
#}
