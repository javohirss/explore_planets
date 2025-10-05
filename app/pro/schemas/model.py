from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class DatasetInfo(BaseModel):
    train_samples: int
    test_samples: int
    features_count: int


class ModelMetrics(BaseModel):
    thresholds: List[float] = Field(..., description="Список порогов для классификации")
    recall: List[float] = Field(..., description="Значения recall для каждого порога")
    precision: List[float] = Field(..., description="Значения precision для каждого порога")
    f1_score: List[float] = Field(..., description="Значения F1-score для каждого порога")
    auc_roc: float = Field(..., description="AUC ROC score")
    auc_pr: Optional[float] = Field(None, description="AUC PR score")
    best_threshold: float = Field(..., description="Лучший порог по F1-score")
    best_f1: float = Field(..., description="Лучший F1-score")
    training_date: Optional[str] = Field(None, description="Дата обучения модели")
    dataset_info: Optional[DatasetInfo] = Field(None, description="Информация о датасете")


class ModelInput(BaseModel):
    name: str
    path: str
    metrics: Optional[ModelMetrics] = None


class ModelResponse(BaseModel):
    id: int
    name: str
    path: str
    metrics: Optional[ModelMetrics] = None