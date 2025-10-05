from sqlalchemy import Column, Integer, String

from app.db.base import Base


class Planet(Base):
    __tablename__ = "planets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, unique=True)
    features_path = Column(String, nullable=False)