from fastapi import FastAPI


from app.novice.routers.predictions import router as novice_predictions_router
from app.novice.routers.planet import router as novice_planets_router
from app.pro.routers.predictions import router as pro_predicions_router
from app.pro.routers.model import router as model_router
    

app = FastAPI(
    title="Explore Planets API",
    description="API для анализа экзопланет с использованием машинного обучения",
    version="1.0.0"
)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API работает корректно"}

app.include_router(novice_predictions_router)
app.include_router(pro_predicions_router)
app.include_router(model_router)
app.include_router(novice_planets_router)
