"""
Rocket Flight Solver
© 2025 George Qiao. All rights reserved.

This software is for educational and competition use only.
Unauthorized copying, modification, or redistribution is prohibited.
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import numpy as np
from scipy.stats import norm

from solver.solver import solve_mass
from solver.constants import SITE, H_WINDOW, MIN_LIFTOFF_MASS, MAX_LIFTOFF_MASS
from solver.engines import ENGINES, engine_scale
from solver.models import beta, sigma_H, m0, beta_T, RHO_REF
from solver.atmosphere import atmosphere_density

app = FastAPI(title="Rocket Flight Solver")
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------- request models ----------

class SolveRequest(BaseModel):
    engine: str
    temperature: float
    pressure: float
    humidity: float
    target_apogee: float


class CurveRequest(SolveRequest):
    pass


# ---------- pages ----------

@app.get("/")
def root():
    return FileResponse("static/index.html")


# ---------- core solver ----------

@app.post("/solve")
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


# ---------- A3-1: curve + probability ----------

@app.post("/curve")
def curve(req: CurveRequest):

    weather = {
        "temperature": req.temperature,
        "pressure": req.pressure,
        "humidity": req.humidity,
        "elevation": SITE["elevation_m"],
    }

    rho = atmosphere_density(
        p_qnh_hpa=req.pressure,
        temperature_c=req.temperature,
        humidity=req.humidity,
        elevation_m=SITE["elevation_m"],
    )
    rho_ratio = rho / RHO_REF
    scale = engine_scale(req.engine)

    masses = list(range(int(MIN_LIFTOFF_MASS), int(MAX_LIFTOFF_MASS) + 1))
    apogee, apogee_hi, apogee_lo, probability = [], [], [], []

    H_min = req.target_apogee - H_WINDOW
    H_max = req.target_apogee + H_WINDOW

    for m in masses:
        m_c = m - m0
        H = float(np.array([1.0, m_c, rho_ratio]) @ beta) * scale

        z1 = (H_min - H) / sigma_H
        z2 = (H_max - H) / sigma_H
        P = float(norm.cdf(z2) - norm.cdf(z1))

        apogee.append(H)
        apogee_hi.append(H + sigma_H)
        apogee_lo.append(H - sigma_H)
        probability.append(P)

    return {
        "mass": masses,
        "apogee": apogee,
        "apogee_hi": apogee_hi,
        "apogee_lo": apogee_lo,
        "probability": probability,
        "H_target": req.target_apogee,
        "H_window": H_WINDOW,
        "confidence_req": 0.75,
    }

