from fastapi import APIRouter, Depends, HTTPException, Response, status
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
    try:
        return await PlanetService.get_all(session)
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Не удалось получить данные: {e}"
        )
    


@router.post("/add")
async def add_planet(inputs: PlanetInput, session: AsyncSession = Depends(get_async_session)):
    try:
        async with session.begin():
            await PlanetService.add_planet(inputs.name, inputs.features_path, session)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Не удалось добавить планету: {e}"
        )

    return {"message": "Планета успешно добавлена"}


