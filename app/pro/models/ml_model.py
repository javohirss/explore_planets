from sqlalchemy import Column, Integer, String, JSON

from app.db import Base


class Model(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False) 
    path = Column(String, nullable=False, unique=True)
    metrics = Column(JSON, nullable=True)  


