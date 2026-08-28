from pathlib import Path

from app.core import design_store
from app.core.storage import outputs_dir
from app.imaging.pipeline import generate_design
from app.models.design import DesignStatus


def generate_design_task(design_id: str) -> None:
    design = design_store.load(design_id)
    if design is None:
        return

    design.status = DesignStatus.PROCESSING.value
    design_store.save(design)

    try:
        result = generate_design(
            image_path=Path(design.original_image_path),
            color_count=design.color_count,
            output_dir=outputs_dir(design.id),
        )
    except Exception as exc:
        design.status = DesignStatus.FAILED.value
        design.error_message = str(exc)
        design_store.save(design)
        raise

    design.status = DesignStatus.COMPLETED.value
    design.outline_image_path = str(result.outline_image_path)
    design.preview_image_path = str(result.preview_image_path)
    design.regions_json = [
        {
            "id": region.id,
            "number": region.number,
            "colorHex": region.color_hex,
            "points": region.points,
        }
        for region in result.regions
    ]
    design_store.save(design)
