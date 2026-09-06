from typing import Literal

from pydantic import BaseModel


class DesignCreateRequest(BaseModel):
    image_id: str
    color_count: int
    mode: Literal["illustration", "photo"] = "illustration"
    recommended_colors: int | None = None


class DesignCreateResponse(BaseModel):
    id: str
    status: str


class RegionSchema(BaseModel):
    id: int
    number: int
    colorHex: str
    points: list[list[int]]


class DesignStatusResponse(BaseModel):
    id: str
    status: str
    outline_image_url: str | None = None
    preview_image_url: str | None = None
    regions: list[RegionSchema] | None = None
