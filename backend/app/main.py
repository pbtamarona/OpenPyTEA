"""OpenPyTEA GUI - FastAPI Backend."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import equipment, plant, analysis, io

app = FastAPI(title="OpenPyTEA GUI", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(equipment.router, prefix="/api/equipment", tags=["equipment"])
app.include_router(plant.router, prefix="/api/plant", tags=["plant"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(io.router, prefix="/api/project", tags=["project"])


@app.get("/api/health")
def health():
    return {"status": "ok"}
