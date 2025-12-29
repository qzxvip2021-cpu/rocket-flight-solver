"""
Rocket Flight Solver
© 2025 George Qiao. All rights reserved.

This software is for educational and competition use only.
Unauthorized copying, modification, or redistribution is prohibited.
"""

import os
import csv
from datetime import datetime

import numpy as np
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from solver.model_store import get_model, set_model, get_model_info
from solver.train_from_excel import train_model_from_excel
from solver.solver import solve_mass
from solver.constants import SITE
from solver.engines import ENGINES, engine_scale
from solver.models import RHO_REF
from solver.atmosphere import atmosphere_density


# =========================================================
# App
# =========================================================

app = FastAPI(title="Rocket Flight Solver")
app.mount("/static", StaticFiles(directory="static"), name="static")


# =========================================================
# Config
# =========================================================

TEAM_PASSWORD = os.getenv("TEAM_PASSWORD")
if TEAM_PASSWORD is None:
    raise RuntimeError("TEAM_PASSWORD is not set")

ARC_EXCEL_PATH = os.getenv(
    "ARC_EXCEL_PATH",
    os.path.join("data", "ARC_flights.xlsx")
)

DATA_DIR = "data"
PENDING_FILE = os.path.join(DATA_DIR, "pending_flights.csv")
FINAL_FILE = os.path.join(DATA_DIR, "flights.csv")


# =========================================================
# Models
# =========================================================

class SolveRequest(BaseModel):
    engine: str
    temperature: float
    pressure: float
    humidity: float
    target_apogee: float


class FlightSubmitRequest(BaseModel):
    password: str
    engine: str
    liftoff_mass: float   # g
    apogee: float         # ft
    flight_time: float    # s
    temperature: float    # °C
    pressure: float       # hPa
    humidity: float       # %


# =========================================================
# Startup
# =========================================================

@app.on_event("startup")
def load_model_on_startup():
    try:
        model = train_model_from_excel(
            excel_path=ARC_EXCEL_PATH,
            engine_name="F20-4W"
        )
        set_model(model)
        print("Model loaded successfully")
    except Exception as e:
        print("⚠️ Model load failed:", e)



# =========================================================
# Routes
# =========================================================

@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


# =========================================================
# 🔍 Model Info (NEW, READ-ONLY)
# =========================================================

@app.get("/api/model_info")
def model_info():
    """
    Expose current loaded model metadata for UI / debugging.
    Read-only, no authentication required.
    """
    try:
        return get_model_info()
    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )


# =========================================================
# Admin
# =========================================================

@app.post("/api/admin/reload_model")
def reload_model(payload: dict = Body(...)):
    password = payload.get("password", "")
    if password != TEAM_PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorized")

    engine = payload.get("engine", "F20-4W")

    try:
        new_model = train_model_from_excel(
            excel_path=ARC_EXCEL_PATH,
            engine_name=engine
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Model reload failed: {str(e)}"
        )

    set_model(new_model)

    return {
        "status": "ok",
        "model": get_model_info(),
    }


# =========================================================
# Solver
# =========================================================

@app.post("/api/solve")
def solve(req: SolveRequest):
    if req.engine not in ENGINES:
        raise HTTPException(status_code=400, detail="Unknown engine")

    weather = {
        "temperature": req.temperature,
        "pressure": req.pressure,
        "humidity": req.humidity,
        "elevation": SITE["elevation_m"],
    }

    try:
        model = get_model()
    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )

    return solve_mass(
        weather=weather,
        target_apogee=req.target_apogee,
        beta=model.beta,
        sigma_H=model.sigma_H,
        m0=model.m0,
        beta_T=model.beta_T,
        engine_name=req.engine,
    )


# =========================================================
# Curve
# =========================================================

@app.post("/api/curve")
def curve(req: SolveRequest):
    if req.engine not in ENGINES:
        raise HTTPException(status_code=400, detail="Unknown engine")

    try:
        model = get_model()
    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )

    rho = atmosphere_density(
        req.pressure,
        req.temperature,
        req.humidity,
        SITE["elevation_m"]
    )
    rho_ratio = rho / RHO_REF
    scale = engine_scale(req.engine)

    masses = list(range(553, 651))
    apogee, hi, lo = [], [], []

    for m in masses:
        m_c = m - model.m0
        H = float(np.array([1.0, m_c, rho_ratio]) @ model.beta) * scale
        apogee.append(H)
        hi.append(H + model.sigma_H)
        lo.append(H - model.sigma_H)

    return {
        "mass": masses,
        "apogee": apogee,
        "apogee_hi": hi,
        "apogee_lo": lo,
        "H_target": req.target_apogee,
        "H_window": 25,
    }


# =========================================================
# Submit Flight Data
# =========================================================

@app.post("/api/submit_flight")
def submit_flight(data: FlightSubmitRequest):

    if data.password != TEAM_PASSWORD:
        raise HTTPException(status_code=403, detail="Invalid team password")

    if data.liftoff_mass <= 0 or data.apogee <= 0:
        raise HTTPException(status_code=400, detail="Invalid flight data")

    if not (0 <= data.humidity <= 100):
        raise HTTPException(status_code=400, detail="Invalid humidity")

    os.makedirs(DATA_DIR, exist_ok=True)
    file_exists = os.path.exists(PENDING_FILE)

    with open(PENDING_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "engine",
                "liftoff_mass",
                "apogee",
                "flight_time",
                "temperature",
                "pressure",
                "humidity",
                "status"
            ])

        writer.writerow([
            datetime.utcnow().isoformat(),
            data.engine,
            data.liftoff_mass,
            data.apogee,
            data.flight_time,
            data.temperature,
            data.pressure,
            data.humidity,
            "pending"
        ])

    return {
        "status": "ok",
        "message": "Flight data submitted for review"
    }

