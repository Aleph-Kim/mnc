from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.core import design_store
from app.core.storage import outputs_dir

templates = Jinja2Templates(directory="templates")

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@router.get("/designs/{design_id}/view", response_class=HTMLResponse)
def design_view(request: Request, design_id: str):
    return templates.TemplateResponse(
        "result.html", {"request": request, "design_id": design_id}
    )


@router.get("/designs/{design_id}/status-partial", response_class=HTMLResponse)
def design_status_partial(request: Request, design_id: str):
    design = design_store.load(design_id)
    return templates.TemplateResponse(
        "partials/status.html", {"request": request, "design": design,
        "has_details": bool(design and (outputs_dir(design.id) / "details.png").exists())}
    )
