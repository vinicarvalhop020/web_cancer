import joblib
import pandas as pd
from flask import Flask, request, render_template, jsonify
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import load_model
from category_encoders import OneHotEncoder
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

categorical_columns = ['sexo', 'racacor', 'clitrat', 'histfamc', 'alcoolis', 'tabagism', 'loctudet', 'estadiam', 'pritrath', 'estadofinal']
numerical_columns = ['dias', 'idade1']

root = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(root, "models")
tools_dir = os.path.join(root, "tools")
nn_path = os.path.join(models_dir, "nn_model.keras")
xgb_path = os.path.join(models_dir, "xgb_model.joblib")
encoder_path = os.path.join(tools_dir, "encoder.pkl")
scaler_path = os.path.join(tools_dir, "scaler.pkl")

target_map = {0: 'Carcinoma', 1: 'Outros', 2: 'Linfoma', 3: 'Sarcoma', 4: 'Sistema Nervoso Central', 5: 'Melanoma'}

def load_preprocessors(encoder_path, scaler_path):
    encoder = joblib.load(encoder_path)
    scaler = joblib.load(scaler_path)
    return encoder, scaler

def preprocess_data(new_data, encoder, scaler, categorical_columns, numerical_columns):
    new_data_encoded = encoder.transform(new_data[categorical_columns])
    new_data = pd.concat([new_data.drop(categorical_columns, axis=1), new_data_encoded], axis=1)
    new_data[numerical_columns] = scaler.transform(new_data[numerical_columns])
    processed_data = new_data
    return processed_data

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = {
            'sexo': request.form['sexo'],
            'racacor': request.form['racacor'],
            'clitrat': request.form['clitrat'],
            'histfamc': request.form['histfamc'],
            'alcoolis': request.form['alcoolis'],
            'tabagism': request.form['tabagism'],
            'loctudet': request.form['loctudet'],
            'estadiam': request.form['estadiam'],
            'pritrath': request.form['pritrath'],
            'dias': request.form['dias'],
            'idade1': request.form['idade1'],
            'estadofinal': request.form['estadofinal']
        }

        new_data = pd.DataFrame([form_data])
        

        encoder, scaler = load_preprocessors(encoder_path, scaler_path)
        data = preprocess_data(new_data, encoder, scaler, categorical_columns, numerical_columns)
        
        xgb_model_loaded = joblib.load(xgb_path)
        model_nn = tf.keras.models.load_model(nn_path)
        data_for_predict = xgb_model_loaded.predict_proba(data)
        

        final_predict = model_nn.predict(data_for_predict)
        final_predict = final_predict.argmax(axis=1)
        final_predict = target_map[final_predict[0]]
     

        return render_template('results.html', prediction=final_predict)

    except Exception as e:
        print("Erro durante a predição:", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run()