from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum


class DesignStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Design:
    id: str
    image_id: str
    original_image_path: str
    status: str
    color_count: int
    mode: str = "illustration"
    recommended_colors: int | None = None
    outline_image_path: str | None = None
    preview_image_path: str | None = None
    regions_json: list[dict] | None = None
    error_message: str | None = None
    created_at: datetime = None
    updated_at: datetime = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Design":
        data = dict(data)
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)
