import uuid
from pathlib import Path

import cv2
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.storage import uploads_dir
from app.imaging.recommend import recommend_color_count
from app.schemas.analyze import AnalyzeResponse

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_image(file: UploadFile = File(...)) -> AnalyzeResponse:
    ext = Path(file.filename or "").suffix or ".png"
    image_id = str(uuid.uuid4())
    dest = uploads_dir() / f"{image_id}{ext}"

    contents = await file.read()
    dest.write_bytes(contents)

    image = cv2.imread(str(dest))
    if image is None:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="uploaded file is not a valid image")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    recommended = recommend_color_count(rgb)

    return AnalyzeResponse(image_id=image_id, recommended_colors=recommended)
