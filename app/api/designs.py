from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.core import design_store
from app.core.storage import uploads_dir
from app.schemas.design import (
    DesignCreateRequest,
    DesignCreateResponse,
    DesignStatusResponse,
    RegionSchema,
)
from app.tasks.design_tasks import generate_design_task

router = APIRouter()


def _find_uploaded_image(image_id: str) -> Path | None:
    for path in uploads_dir().glob(f"{image_id}.*"):
        return path
    return None


@router.post("/designs", response_model=DesignCreateResponse)
def create_design(
    payload: DesignCreateRequest, background_tasks: BackgroundTasks
) -> DesignCreateResponse:
    image_path = _find_uploaded_image(payload.image_id)
    if image_path is None:
        raise HTTPException(status_code=404, detail="image_id not found")

    design = design_store.create(
        image_id=payload.image_id,
        original_image_path=str(image_path),
        color_count=payload.color_count,
        mode=payload.mode,
        recommended_colors=payload.recommended_colors,
    )

    background_tasks.add_task(generate_design_task, design.id)

    return DesignCreateResponse(id=design.id, status=design.status)


@router.get("/designs/{design_id}", response_model=DesignStatusResponse)
def get_design(design_id: str) -> DesignStatusResponse:
    design = design_store.load(design_id)
    if design is None:
        raise HTTPException(status_code=404, detail="design not found")

    outline_url = (
        f"/storage/outputs/{design.id}/outline.png" if design.outline_image_path else None
    )
    preview_url = (
        f"/storage/outputs/{design.id}/preview.png" if design.preview_image_path else None
    )
    regions = (
        [RegionSchema(**r) for r in design.regions_json] if design.regions_json else None
    )

    return DesignStatusResponse(
        id=design.id,
        status=design.status,
        outline_image_url=outline_url,
        preview_image_url=preview_url,
        regions=regions,
    )
