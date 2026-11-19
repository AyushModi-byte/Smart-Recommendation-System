"""
AuraPath swipe API (FastAPI).  Local run, from the repo root:
  .venv\\Scripts\\python.exe -m uvicorn server.main:app --port 8000
Environment:
  SWIPE_BUNDLE     folder made by training/16_export_swipe.py (default: ./bundle_swipe)
  ALLOWED_ORIGINS  comma-separated origins for the browser app (default: *)
  FEEDBACK_LOG     optional path of a JSON-lines file for accept / reject events
"""
import json, os, threading, time
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .engine import CAT_SHORT, Engine

ENGINE = Engine()
# warm the taste mean in the background, so the first "people like you" request after a restart is not slow
threading.Thread(target=lambda: ENGINE.taste([0]), daemon=True).start()
app = FastAPI(title="AuraPath swipe API", version="0.1")
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",")],
                   allow_methods=["GET", "POST"], allow_headers=["*"])
_log_lock = threading.Lock()


class RecommendIn(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    radius_km: float = Field(25.0, gt=0, le=200)
    accepted: List[int] = Field(default_factory=list, max_length=500)
    rejected: List[int] = Field(default_factory=list, max_length=500)
    seen: List[int] = Field(default_factory=list, max_length=1000)
    mood: Optional[str] = None
    category: Optional[int] = None
    hidden: bool = False
    n: int = Field(6, ge=1, le=20)
    explore: float = Field(0.2, ge=0, le=1)
    weather: bool = True          # reorder for rain / cold / heat at the area right now (Open-Meteo)
    quiet: bool = False           # demote places that are busy in the current time slot


class FeedbackIn(BaseModel):
    session: str = Field(max_length=64)
    place: int
    action: str = Field(pattern="^(shown|accept|reject|save|undo)$")
    reason: Optional[str] = Field(None, max_length=200)


@app.get("/api/health")
def health():
    return {"ok": True, "places": ENGINE.n, "run": ENGINE.meta["run"]}


@app.get("/api/meta")
def meta():
    return {"moods": ENGINE.moods, "categories": [{"id": i, "name": n} for i, n in enumerate(CAT_SHORT)],
            "towns": ENGINE.top_towns(12), "n_places": ENGINE.n, "run": ENGINE.meta["run"]}


class TasteIn(BaseModel):
    accepted: List[int] = Field(default_factory=list, max_length=500)


@app.post("/api/taste")
def taste(body: TasteIn):
    return {"taste": ENGINE.taste(body.accepted)}


@app.get("/api/conditions")
def conditions(lat: float, lon: float):
    return ENGINE.conditions(lat, lon) or {}


@app.get("/api/towns")
def towns(q: str = ""):
    return ENGINE.towns(q[:60])


@app.post("/api/recommend")
def recommend(body: RecommendIn):
    return ENGINE.recommend(**body.model_dump())


@app.get("/api/place/{pid}")
def place(pid: int):
    r = ENGINE.rows([pid]).get(pid)
    if r is None:
        raise HTTPException(404, "unknown place")
    return ENGINE.card(pid, r)


@app.post("/api/feedback")
def feedback(body: FeedbackIn):
    path = os.environ.get("FEEDBACK_LOG")
    if path:
        with _log_lock, open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), **body.model_dump()}) + "\n")
    return {"ok": True}


# Laptop / tunnel mode: when the web app has been built (web/dist), serve it from the same address as the API,
# so a single Cloudflare Tunnel URL gives the whole app. On Render the web app lives on Vercel and dist is absent.
_DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web", "dist")
if os.path.isdir(_DIST):
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="web")
