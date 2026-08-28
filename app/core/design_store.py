import json
import uuid
from datetime import datetime, timezone

from app.core.storage import outputs_dir
from app.models.design import Design, DesignStatus


def _status_path(design_id: str):
    return outputs_dir(design_id) / "status.json"


def create(
    image_id: str,
    original_image_path: str,
    color_count: int,
    recommended_colors: int | None = None,
) -> Design:
    now = datetime.now(timezone.utc)
    design = Design(
        id=str(uuid.uuid4()),
        image_id=image_id,
        original_image_path=original_image_path,
        status=DesignStatus.PENDING.value,
        color_count=color_count,
        recommended_colors=recommended_colors,
        created_at=now,
        updated_at=now,
    )
    save(design)
    return design


def save(design: Design) -> None:
    design.updated_at = datetime.now(timezone.utc)
    _status_path(design.id).write_text(
        json.dumps(design.to_dict()), encoding="utf-8"
    )


def load(design_id: str) -> Design | None:
    path = _status_path(design_id)
    if not path.exists():
        return None
    return Design.from_dict(json.loads(path.read_text(encoding="utf-8")))
