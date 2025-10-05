from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class DatasetInfo(BaseModel):
    train_samples: int
    test_samples: int
    features_count: int


class ModelMetrics(BaseModel):
    thresholds: List[float] = Field(..., description="Список порогов для классификации")
    positive_count: List[float] = Field(..., description="Количество предсказанных позитивно")
    negative_count: List[float] = Field(..., description="Количество предсказанных отрицательно")
    recall: List[float] = Field(..., description="Значения recall для каждого порога")
    precision: List[float] = Field(..., description="Значения precision для каждого порога")
    f1_score: List[float] = Field(..., description="Значения F1-score для каждого порога")
    auc_roc: float = Field(None, description="AUC ROC score")
    auc_pr: Optional[float] = Field(None, description="AUC PR score")
    


class ModelInput(BaseModel):
    name: str
    path: str
    metrics: Optional[ModelMetrics] = None


class ModelResponse(BaseModel):
    id: int
    name: str
    path: str
    metrics: Optional[ModelMetrics] = None