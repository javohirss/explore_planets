# AI-Generated Code Disclaimer
# Some parts of this project were created or refined with assistance from AI tools such as ChatGPT (OpenAI GPT-5)
# and Claude 4.5 in Cursor. All AI-generated code was reviewed, debugged,
# and validated by our human development team before integration.


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.novice.routers.predictions import router as novice_predictions_router
from app.novice.routers.planet import router as novice_planets_router
from app.pro.routers.predictions import router as pro_predicions_router
from app.pro.routers.model import router as model_router
    

app = FastAPI(
    title="Explore Planets API",
    description="API для анализа экзопланет с использованием машинного обучения",
    version="1.0.0",
)

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API работает корректно"}

app.include_router(novice_predictions_router, prefix="/api")
app.include_router(pro_predicions_router, prefix="/api")
app.include_router(model_router, prefix="/api")
app.include_router(novice_planets_router, prefix="/api")
