# Пример структуры поля metrics для модели с строгой типизацией
# Теперь JSON имеет четко определенную структуру

example_metrics = {
    "thresholds": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "recall": [0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97, 0.98],
    "precision": [0.95, 0.92, 0.88, 0.85, 0.82, 0.78, 0.75, 0.72, 0.68],
    "f1_score": [0.74, 0.80, 0.84, 0.85, 0.86, 0.84, 0.84, 0.83, 0.81],
    "auc_roc": 0.89,
    "auc_pr": 0.85,
    "best_threshold": 0.5,
    "best_f1": 0.86,
    "training_date": "2024-01-15",
    "dataset_info": {
        "train_samples": 10000,
        "test_samples": 2000,
        "features_count": 45
    }
}

# Пример использования в API с валидацией:
"""
POST /model/add
{
    "name": "TESS Random Forest Model",
    "path": "app/assets/models/TESS/RF_68ef4c4f.joblib",
    "metrics": {
        "thresholds": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "positive_count": [24, 32, 53, 66, 90, 120, 126, 133, 140, 144],
        "negative_count": [68, 47 , 33, 29, 20, 18, 15, 13, 12, 9],
        "recall": [0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97, 0.98],
        "precision": [0.95, 0.92, 0.88, 0.85, 0.82, 0.78, 0.75, 0.72, 0.68],
        "f1_score": [0.74, 0.80, 0.84, 0.85, 0.86, 0.84, 0.84, 0.83, 0.81],
        "auc_roc": 0.89,
        "auc_pr": 0.85,
        
    }
}

PUT /model/1/metrics
{
    "thresholds": [0.1, 0.2, 0.3, 0.4, 0.5],
    "recall": [0.6, 0.7, 0.8, 0.85, 0.9],
    "positive_count": [24, 32 , 53, 66, 90],
    "negative_count": [68, 47 , 33, 29, 20],
    "precision": [0.95, 0.92, 0.88, 0.85, 0.82],
    "f1_score": [0.74, 0.80, 0.84, 0.85, 0.86],
    "auc_roc": 0.89,
    
}
"""

# Обязательные поля в ModelMetrics:
# - thresholds: List[float] - список порогов
# - recall: List[float] - значения recall
# - precision: List[float] - значения precision  
# - f1_score: List[float] - значения F1-score
# - auc_roc: float - AUC ROC score
# - best_threshold: float - лучший порог
# - best_f1: float - лучший F1-score

# Опциональные поля:
# - auc_pr: Optional[float] - AUC PR score
# - training_date: Optional[str] - дата обучения
# - dataset_info: Optional[DatasetInfo] - информация о датасете
