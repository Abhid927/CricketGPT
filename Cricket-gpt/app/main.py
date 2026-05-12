from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.server import app as api_app

app = FastAPI()

# API lives under /api
app.mount("/api", api_app)

# UI lives at /
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")

