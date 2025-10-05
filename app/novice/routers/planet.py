from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.novice.schemas import PlanetInput
from app.pro.services import PlanetService
from app.db.session import get_async_session


router = APIRouter(
    prefix="/planets",
    tags=["Planets"]
)


@router.get("")
async def get_planets(session: AsyncSession = Depends(get_async_session)):
    await PlanetService.get_all()


@router.post("/add")
async def add_planet(inputs: PlanetInput, session: AsyncSession = Depends(get_async_session)):
    async with session.begin():
        await PlanetService.add_planet(inputs.name, inputs.features_path, session)