from pydantic import BaseModel

from app.pro.models import PredictionLabel


class PredictionResponse(BaseModel):
    probability: float
    label: PredictionLabel