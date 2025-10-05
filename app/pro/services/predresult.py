from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.pro.services import BaseService
from app.pro.models import PredictionResult


class PredictionResultService(BaseService):
    model=PredictionResult


    @classmethod
    async def add_result(cls, probability: float, label: str, model_id: int, session: AsyncSession):
        query = insert(cls.model).values(probability=probability, label=label, model_id=model_id).returning(cls.model)
        result = await session.execute(query)
        return result.scalar_one_or_none()
