/proyecto-dengue/
├── main.py               # API FastAPI
├── modelo_lstm.h5        # Modelo LSTM entrenado
├── scaler.pkl            # Scaler usado en el entrenamiento
├── requirements.txt      # Dependencias necesarias
└── README.md             # Documentación del proyecto

--

1. Requisitos Previos
Python 3.10.x (recomendado)
pip actualizado:
python -m pip install --upgrade pip

--

2. Crear el Entorno Virtual
Windows
python -m venv venv
venv\Scripts\activate

--

3. Instalar Dependencias
Asegúrate de estar dentro del entorno virtual:
pip install -r requirements.txt

Este comando instalará:
FastAPI / Uvicorn
TensorFlow 2.15
NumPy
Scikit-learn
h5py
Pydantic
Utilidades varias

--

4. Ejecutar la API
Corre el servidor:
uvicorn main:app --reload

Accede a la documentación interactiva en:
http://127.0.0.1:8000/docs

--

5. Ejemplo de Uso del Endpoint /predict
POST /predict
Envía exactamente 10 valores (correspondientes a las últimas 10 semanas):
Request (JSON)
{
  "valores": [10, 20, 15, 18, 22, 30, 25, 27, 29, 33]
}

Response:
{
  "prediccion": 38.61,
  "mensaje": "Predicción de casos de dengue para la próxima semana"
}

--

6. Detalle Técnico del Modelo
Tipo de modelo: LSTM (Keras – TensorFlow 2.15)
Input shape: (1, 10, 1)
Escalado: MinMaxScaler
El archivo scaler.pkl debe ser el mismo usado en Colab durante el entrenamiento.