from typing import Any
import pandas as pd
import numpy as np
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.pro.services import ModelService, PlanetService
from app.db.session import get_async_session
from app.dependencies import load_model


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
        features_path ="app/"+ planet_obj.features_path
        features = pd.read_csv(features_path)
        print(features.columns)
        if "tess_name" in features.columns:
            model_obj = await ModelService.get_by_id(1, session)
            row = features[features["tess_name"]==planet_obj.name].drop(columns=["tess_name"], axis=1)

        elif "k2_name" in features.columns:
            model_obj = await ModelService.get_by_id(2, session)
            row = features[features["k2_name"]==planet_obj.name].drop(columns=["k2_name"], axis=1)

        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Датасет не содержит название планеты: {str(e)}"
            )
        

        model = load_model(model_obj.path)
        proba = model.predict(row)
        
        
        if isinstance(proba, np.ndarray):
            # numpy array
            proba_list = proba.tolist()
        elif hasattr(proba, 'numpy'):
            # tensorflow tensor
            proba_list = proba.numpy().tolist()
        elif hasattr(proba, 'tolist'):
            # другие объекты с методом tolist
            proba_list = proba.tolist()
        else:
            # обычный Python объект
            proba_list = list(proba) if hasattr(proba, '__iter__') else [proba]
        
        return {"result": proba_list}


    except Exception as e:
        print(f"Ошибка в planet_predict: {e}")
        print(f"Тип ошибки: {type(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при предсказании: {str(e)}"
        )
