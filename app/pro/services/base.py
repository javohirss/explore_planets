from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import async_session_maker


class BaseService:
    model = None

    @classmethod
    async def get_by_id(cls, id: int, session: AsyncSession):
        query = select(cls.model).where(cls.model.id == id)
        result = await session.execute(query)
        return result.scalar_one_or_none()
    

    @classmethod
    async def get_all(cls, session: AsyncSession):
        query = select(cls.model)
        result = await session.execute(query)
        return result.scalars().all()