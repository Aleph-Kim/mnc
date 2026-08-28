from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.analyze import router as analyze_router
from app.api.designs import router as designs_router
from app.web.routes import router as web_router

app = FastAPI(title="Number Coloring")

app.include_router(analyze_router)
app.include_router(designs_router)
app.include_router(web_router)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/storage", StaticFiles(directory="storage"), name="storage")


@app.get("/health")
def health():
    return {"status": "ok"}
