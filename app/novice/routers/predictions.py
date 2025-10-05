from typing import Any
import pandas as pd
import numpy as np
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.pro.services import ModelService, PlanetService
from app.db.session import get_async_session
from app.dependencies import load_model, parse_model_predict


router = APIRouter(
    prefix="/novice/predict",
    tags=["Novice Model Prediction"]
)


# @router.post("")
# async def novice_predict(file: UploadFile = File(None), session: AsyncSession = Depends(get_async_session))-> dict[str, Any] | None:
#     if not file:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="File was not uploaded"
#         )
#     try:
#         model_obj = await ModelService.get_by_id(1, session)
        


#     except Exception as e:
#         print(f"Error: {e}")


@router.post("/{planet_id}")
async def planet_predict(planet_id: int, session: AsyncSession = Depends(get_async_session)):
    try:
        
        planet_obj = await PlanetService.get_by_id(planet_id, session)
        features_path = planet_obj.features_path
        features = pd.read_csv(features_path)
        
        if "tess_name" in features.columns:
            row = features[features["tess_name"]==planet_obj.name].drop(columns=["tess_name"], axis=1)
            model_obj = await ModelService.get_by_name("tess_random_forest", session)

        elif "k2_name" in features.columns:
            row = features[features["k2_name"]==planet_obj.name].drop(columns=["k2_name"], axis=1)
            model_obj = await ModelService.get_by_name("k2_random_forest", session)

        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Датасет не содержит название планеты: {str(e)}"
            )
        

        model = load_model(model_obj.path)
        prediction = parse_model_predict(model, row)
        if isinstance(prediction, list):
            prediction = prediction[0]
        
        return {"result": prediction}


    except Exception as e:
        print(f"Ошибка в planet_predict: {e}")
        print(f"Тип ошибки: {type(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при предсказании: {str(e)}"
        )
