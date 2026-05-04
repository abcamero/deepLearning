from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .api.routes import router
from .core.config import settings

static_dir = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Crazy Travel Itinerary Generator",
    version="0.1.0",
    description="Backend API for the Crazy Travel Itinerary Generator project.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.frontend_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=static_dir), name="static")
app.include_router(router)

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    return static_dir.joinpath("index.html").read_text()

@app.get("/login", response_class=HTMLResponse)
async def serve_login():
    return static_dir.joinpath("login.html").read_text()

@app.get("/health", tags=["health"])
async def health_check():
    return {"status": "ok"}
