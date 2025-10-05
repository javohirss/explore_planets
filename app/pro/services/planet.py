from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert

from app.pro.models import Planet
from app.pro.services import BaseService



class PlanetService(BaseService):
    model = Planet

    @classmethod
    async def add_planet(cls, name: str, features_path: str, session: AsyncSession):
        query = insert(cls.model).values(name=name, features_path=features_path).returning(cls.model)
        await session.execute(query)
