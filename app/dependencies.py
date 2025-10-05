import io
import os
from typing import Literal
import tensorflow as tf
import pandas as pd
import joblib
from fastapi import HTTPException, status

from app.preprocess import prepare_data



def load_model(path: str):
    ext = os.path.splitext(path)[1].lower()
    
    if ext in [".joblib", ".pkl"]:
        return joblib.load(path)

    if ext in [".h5", ".keras", ""]:
        return tf.keras.models.load_model(path)

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Неподдерживаемый формат: {ext}"
    )

def make_predict_labels(predictions):
    return ["EXOPLANET" if p>0 else "NOT EXOPLANET" for p in predictions ]


def classify_predict_proba(predictions, threshold):
    return [1 if p>threshold else 0 for  p in predictions]
     

def parse_model_predict(model, data, threshold):
    if hasattr(model, "predict_proba"):
        return make_predict_labels(model.predict(data))
    
    else:
        return make_predict_labels(classify_predict_proba(model.predict(data).flatten(), threshold))


def predict(model_path, raw_data: bytes, dataset_type: Literal["tess", "k2"], threshold: float = 0.5):
    df = pd.read_csv(io.BytesIO(raw_data), encoding='utf-8-sig')
    model = load_model(model_path)

    if dataset_type == "tess":
        X, cols = prepare_data(df, target_col="tfopwg_disp")

    elif dataset_type == "k2":
        X, cols = prepare_data(df, target_col="disposition")

    else:
        raise HTTPException(status_code=status.HTTP_406_NOT_ACCEPTABLE, detail=f"Неподдерживаемый тип датасета: {dataset_type}")
    
    prediction = parse_model_predict(model, X, threshold)

    return prediction


# def get_predict_label(proba):
#     label = PredictionLabel.LOW
    
#     if proba>=0.66:
#         return PredictionLabel.HIGH
    
#     elif proba >=0.33:
#         return PredictionLabel.MEDIUM
    
#     return label


# def get_model_paths(request: Request):
#     return request.app.state.ml_model_paths