from typing import Any, Literal
import pandas as pd
import io
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.pro.services import ModelService
from app.dependencies import predict
from app.db.session import get_async_session


router = APIRouter(
    prefix="/pro/predict",
    tags=["Pro Model Prediction"]
)



@router.post("/{model_id}")
async def pro_model_predict(model_id: int, dataset_type: Literal["tess", "k2"], file: UploadFile = File(...), session: AsyncSession = Depends(get_async_session)):
    raw = await file.read()
    
    model_obj = await ModelService.get_by_id(model_id, session)
    if not model_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model was not found"
        )
    
    try:
        original_df = pd.read_csv(io.BytesIO(raw), encoding='utf-8-sig')
        
        predictions = predict(model_obj.path, raw, dataset_type=dataset_type)
        
        original_df['predicted_class'] = predictions
        
        csv_buffer = io.StringIO()
        original_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        csv_content = csv_buffer.getvalue()
        
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=predicted_{file.filename}"}
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(e)
        ) from e

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) from e
    


