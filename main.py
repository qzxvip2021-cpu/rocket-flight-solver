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
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from solver.solver import solve_mass
from solver.constants import SITE
from solver.engines import ENGINES, engine_scale
from solver.models import beta, sigma_H, m0, beta_T, RHO_REF
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

DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "flights.csv")


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
# Routes
# =========================================================

@app.get("/")
def index():
    # 主页直接返回 static/index.html
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    # Render 用来判断服务是否起来了（你也可以自己打开 /health 看）
    return {"status": "ok"}


# ---------- Solver ----------

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

    return solve_mass(
        weather=weather,
        target_apogee=req.target_apogee,
        beta=beta,
        sigma_H=sigma_H,
        m0=m0,
        beta_T=beta_T,
        engine_name=req.engine,
    )


# ---------- Curve ----------

@app.post("/api/curve")
def curve(req: SolveRequest):
    if req.engine not in ENGINES:
        raise HTTPException(status_code=400, detail="Unknown engine")

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
        m_c = m - m0
        H = float(np.array([1.0, m_c, rho_ratio]) @ beta) * scale
        apogee.append(H)
        hi.append(H + sigma_H)
        lo.append(H - sigma_H)

    return {
        "mass": masses,
        "apogee": apogee,
        "apogee_hi": hi,
        "apogee_lo": lo,
        "H_target": req.target_apogee,
        "H_window": 25,
    }


# ---------- Submit Flight Data ----------

@app.post("/api/submit_flight")
def submit_flight(data: FlightSubmitRequest):
    if data.password != TEAM_PASSWORD:
        raise HTTPException(status_code=403, detail="Invalid team password")

    # basic validation
    if data.liftoff_mass <= 0 or data.apogee <= 0:
        raise HTTPException(status_code=400, detail="Invalid flight data")

    if not (0 <= data.humidity <= 100):
        raise HTTPException(status_code=400, detail="Invalid humidity")

    os.makedirs(DATA_DIR, exist_ok=True)
    file_exists = os.path.exists(DATA_FILE)

    with open(DATA_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "engine",
                "liftoff_mass_g",
                "apogee_ft",
                "flight_time_s",
                "temperature_c",
                "pressure_hpa",
                "humidity_pct"
            ])

        writer.writerow([
            datetime.utcnow().isoformat(),
            data.engine,
            data.liftoff_mass,
            data.apogee,
            data.flight_time,
            data.temperature,
            data.pressure,
            data.humidity
        ])

    return {"status": "ok", "message": "Flight data submitted successfully"}
