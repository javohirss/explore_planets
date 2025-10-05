from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert
from typing import Optional, Dict, Any

from app.pro.services.base import BaseService
from app.pro.models import Model
from app.pro.schemas.model import ModelMetrics


class ModelService(BaseService):
    model = Model

    @classmethod
    async def add_model(cls, name: str, path: str, metrics: Optional[ModelMetrics] = None, session: AsyncSession = None):
        metrics_dict = metrics.model_dump() if metrics else None
        query = insert(cls.model).values(name=name, path=path, metrics=metrics_dict).returning(cls.model)
        result = await session.execute(query)
        await session.commit()
        return result.scalar_one()

    @classmethod
    async def update_metrics(cls, model_id: int, metrics: ModelMetrics, session: AsyncSession):
        from sqlalchemy import update
        # Преобразуем Pydantic модель в dict для JSON поля
        metrics_dict = metrics.model_dump()
        query = update(cls.model).where(cls.model.id == model_id).values(metrics=metrics_dict)
        await session.execute(query)
        await session.commit()