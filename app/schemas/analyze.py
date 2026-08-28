from pydantic import BaseModel


class AnalyzeResponse(BaseModel):
    image_id: str
    recommended_colors: int
