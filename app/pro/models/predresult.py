from sqlalchemy import Column, Float, ForeignKey, Integer, Enum as SQLEnum
from sqlalchemy.orm import relationship
from enum import Enum

from app.db.base import Base

class PredictionLabel(str, Enum):
    CANDIDATE = "candidate"
    FALSE_POSITIVE = "false positive"



class PredictionResult(Base):
    __tablename__ = "predresults"

    id = Column(Integer, primary_key=True, autoincrement=True)
    label = Column(SQLEnum(PredictionLabel, name="pred_label_enum", values_callable=lambda obj: [e.value for e in obj]))
    probability = Column(Float, nullable=False)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    
    