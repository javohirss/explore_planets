import io
import os
from typing import Literal
import tensorflow as tf
import pandas as pd
import joblib
from fastapi import HTTPException, status

from app.pro.models import PredictionLabel
from app.preprocess import tess_preprocess



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
            



def predict(model_path, raw_data: bytes, dataset_type: Literal["tess", "k2"]):
    df = pd.read_csv(io.BytesIO(raw_data), encoding='utf-8-sig')
    model = load_model(model_path)

    if dataset_type == "tess":
        X, y = tess_preprocess(df)

        prediction = model.predict(X)

    elif dataset_type == "k2":
        
        X = df.select_dtypes(include=['number'])
        prediction = model.predict(X)
    else:
        raise ValueError(f"Неподдерживаемый тип датасета: {dataset_type}")

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