from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession, create_async_engine
from config import settings
from typing import AsyncGenerator

url = settings.db_url

engine = create_async_engine(url)

async_session_maker = async_sessionmaker(engine, expire_on_commit=False)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session