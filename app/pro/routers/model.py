from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_async_session
from app.pro.schemas import ModelInput, ModelResponse, ModelMetrics
from app.pro.services import ModelService

router = APIRouter(
    prefix="/model",
    tags=["ML Model"]
)



@router.get("/", response_model=list[ModelResponse])
async def get_models(session: AsyncSession = Depends(get_async_session)):
    models = await ModelService.get_all(session)
    result = []
    for model in models:
        # Преобразуем dict обратно в ModelMetrics для response
        metrics = ModelMetrics(**model.metrics) if model.metrics else None
        result.append(ModelResponse(id=model.id, name=model.name, path=model.path, metrics=metrics))
    return result


@router.get("/{model_id}", response_model=ModelResponse)
async def get_models(model_id: int, session: AsyncSession = Depends(get_async_session)):
    print(f"Model id : {model_id}")
    model = await ModelService.get_by_id(model_id,session)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Модель не была найдена"
        )
    
    metrics = ModelMetrics(**model.metrics) if model.metrics else None
    return ModelResponse(id=model.id, name=model.name, path=model.path, metrics=metrics)
    


@router.post("/add")
async def add_model(inputs: ModelInput, session: AsyncSession = Depends(get_async_session)):
    try:
        model = await ModelService.add_model(inputs.name, inputs.path, inputs.metrics, session)
        # Преобразуем dict обратно в ModelMetrics для response
        metrics = ModelMetrics(**model.metrics) if model.metrics else None
        return ModelResponse(id=model.id, name=model.name, path=model.path, metrics=metrics)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при добавлении модели: {str(e)}"
        )


@router.put("/{model_id}/metrics")
async def update_model_metrics(model_id: int, metrics: ModelMetrics, session: AsyncSession = Depends(get_async_session)):
    try:
        await ModelService.update_metrics(model_id, metrics, session)
        return {"message": "Метрики успешно обновлены"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при обновлении метрик: {str(e)}"
        )