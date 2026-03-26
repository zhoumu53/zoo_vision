"""
Elephant Game Backend - FastAPI server.

Provides:
  - /api/v1/galaxy/elephants - List elephants with 3D positions
  - /api/v1/galaxy/upload - Upload photo for identification
  - /api/v1/galaxy/upload/{job_id}/status - Poll upload status
  - /storage/uploads/... - Serve uploaded files

Usage:
  cd game/backend
  pip install fastapi uvicorn python-multipart numpy opencv-python ultralytics
  python main.py
"""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import UPLOAD_DIR
from galaxy import router as galaxy_router

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Elephant Game API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(galaxy_router, prefix="/api/v1")

# Serve uploaded files
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/storage/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


@app.get("/api/v1/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
