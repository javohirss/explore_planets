from pydantic import BaseModel


class PlanetInput(BaseModel):
    name: str
    features_path: str