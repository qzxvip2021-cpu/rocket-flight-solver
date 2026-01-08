"""
Rocket Flight Solver
© 2025 George Qiao. All rights reserved.
"""

import os
import csv
import json
from datetime import datetime
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Body, Header, Depends
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from solver.model_store import get_model, set_model
from solver.train_from_excel import train_model_from_excel
from solver.final_solver import monte_carlo_final_solver
from solver.solver import solve_mass
from solver.constants import SITE, H_WINDOW
from solver.engines import ENGINES, engine_scale
from solver.models import RHO_REF
from solver.atmosphere import atmosphere_density

from scipy.stats import norm


# =========================================================
# Config (env)
# =========================================================

TEAM_PASSWORD = os.getenv("TEAM_PASSWORD")
if TEAM_PASSWORD is None:
    raise RuntimeError("TEAM_PASSWORD is not set")

TEAM_API_KEY = os.getenv("TEAM_API_KEY")
if TEAM_API_KEY is None:
    raise RuntimeError("TEAM_API_KEY is not set")

ARC_EXCEL_PATH = os.getenv(
    "ARC_EXCEL_PATH", os.path.join("data", "ARC_flights.xlsx")
)

DATA_DIR = "data"
ACCESS_LOG = os.path.join(DATA_DIR, "access_log.csv")


# =========================================================
# Final mode (file-based switch)
# =========================================================

FINAL_MODE_FILE = os.path.join(DATA_DIR, "final_mode.json")


def read_final_mode() -> bool:
    if not os.path.exists(FINAL_MODE_FILE):
        return False
    with open(FINAL_MODE_FILE, "r", encoding="utf-8") as f:
        return json.load(f).get("enabled", False)


def write_final_mode(enabled: bool, member: str):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(FINAL_MODE_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "enabled": enabled,
                "updated_at": datetime.utcnow().isoformat(),
                "updated_by": member,
            },
            f,
            indent=2,
        )


# =========================================================
# Models (⚠️ 必须放在路由前)
# =========================================================

class SolveRequest(BaseModel):
    engine: str
    temperature: float
    pressure: float
    humidity: float
    target_apogee: float


# =========================================================
# Access logging
# =========================================================

def log_access(
    request: Request,
    endpoint: str,
    engine: str | None,
    mode: str | None,
    success: bool,
    team_member: str,
    reason: str = "",
):
    os.makedirs(DATA_DIR, exist_ok=True)
    file_exists = os.path.exists(ACCESS_LOG)

    with open(ACCESS_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(
                [
                    "timestamp",
                    "endpoint",
                    "method",
                    "engine",
                    "mode",
                    "team_member",
                    "ip_address",
                    "user_agent",
                    "success",
                    "reason",
                ]
            )

        writer.writerow(
            [
                datetime.utcnow().isoformat(),
                endpoint,
                request.method,
                engine or "",
                mode or "",
                team_member,
                request.client.host if request.client else "unknown",
                request.headers.get("user-agent", "unknown"),
                success,
                reason,
            ]
        )


# =========================================================
# Auth dependencies
# =========================================================

def require_team_key(x_team_key: Optional[str] = Header(None)) -> None:
    if x_team_key != TEAM_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing team key")


def require_team_member(x_team_member: Optional[str] = Header(None)) -> str:
    if x_team_member is None:
        raise HTTPException(status_code=400, detail="Missing x-team-member")

    member = x_team_member.strip()
    if not member:
        raise HTTPException(status_code=400, detail="Empty x-team-member")

    if len(member) > 32:
        raise HTTPException(status_code=400, detail="x-team-member too long")

    return member


def require_final_mode_enabled():
    if not read_final_mode():
        raise HTTPException(
            status_code=403,
            detail="Final mode is currently disabled by admin"
        )


# =========================================================
# App
# =========================================================

app = FastAPI(title="Rocket Flight Solver")
app.mount("/static", StaticFiles(directory="static"), name="static")


# =========================================================
# Admin APIs
# =========================================================

@app.get("/api/admin/audit")
def audit(password: str):
    if password != TEAM_PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not os.path.exists(ACCESS_LOG):
        return {"error": "No access log found"}
    with open(ACCESS_LOG, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@app.post("/api/admin/final_mode")
def set_final_mode(
    payload: dict = Body(...),
    _: None = Depends(require_team_key),
    member: str = Depends(require_team_member),
):
    password = payload.get("password")
    enabled = payload.get("enabled")

    if password != TEAM_PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not isinstance(enabled, bool):
        raise HTTPException(status_code=400, detail="enabled must be boolean")

    write_final_mode(enabled, member)

    return {
        "final_mode_enabled": enabled,
        "updated_by": member,
    }


# =========================================================
# Startup
# =========================================================

@app.on_event("startup")
def load_model_on_startup():
    engine = "F20-4W"
    model = train_model_from_excel(ARC_EXCEL_PATH, engine)
    set_model(engine, model)
    print(f"✅ Model loaded: {engine}")


# =========================================================
# Routes
# =========================================================

@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- Curve (Normal Mode helper) ----------

@app.post("/api/curve")
def curve(
    req: SolveRequest,
    request: Request,
    _: None = Depends(require_team_key),
    member: str = Depends(require_team_member),
):
    if req.engine not in ENGINES:
        raise HTTPException(status_code=400, detail="Unknown engine")

    model = get_model(req.engine)

    rho = atmosphere_density(
        req.pressure,
        req.temperature,
        req.humidity,
        SITE["elevation_m"],
    )

    masses = list(range(553, 651))
    apogee, hi, lo = [], [], []

    for m in masses:
        m_c = m - model.m0
        scale = engine_scale(req.engine)
        H = float(np.array([1.0, m_c, rho / RHO_REF]) @ model.beta) * scale
        hi.append(H + model.sigma_H)
        lo.append(H - model.sigma_H)

    log_access(request, "/api/curve", req.engine, "normal", True, member)
    

    H_min = req.target_apogee - H_WINDOW
    H_max = req.target_apogee + H_WINDOW
    P = norm.cdf((H_max - H) / model.sigma_H) - norm.cdf((H_min - H) / model.sigma_H)
    conf.append(float(P) * 100.0)

    return {
        "mass": masses,
        "apogee": apogee,
        "apogee_hi": hi,
        "apogee_lo": lo,
        "confidence": conf,
    }


# ---------- Normal Solver ----------

@app.post("/api/solve")
def solve(
    req: SolveRequest,
    request: Request,
    _: None = Depends(require_team_key),
    member: str = Depends(require_team_member),
):
    if req.engine not in ENGINES:
        raise HTTPException(status_code=400, detail="Unknown engine")

    model = get_model(req.engine)

    weather = {
        "temperature": req.temperature,
        "pressure": req.pressure,
        "humidity": req.humidity,
        "elevation": SITE["elevation_m"],
    }

    # ✅ “时间分优先”逻辑已在 solve_mass 内部实现（见 solver/solver.py）
    result = solve_mass(
        weather=weather,
        target_apogee=req.target_apogee,
        beta=model.beta,
        sigma_H=model.sigma_H,
        m0=model.m0,
        beta_T=model.beta_T,
        engine_name=req.engine,
    )

    log_access(request, "/api/solve", req.engine, "normal", True, member)
    return result


# ---------- Final Solver ----------

@app.post("/api/final_solve")
def final_solve(
    req: SolveRequest,
    request: Request,
    _: None = Depends(require_team_key),
    __: None = Depends(require_final_mode_enabled),
    member: str = Depends(require_team_member),
):
    if req.engine not in ENGINES:
        raise HTTPException(status_code=400, detail="Unknown engine")

    model = get_model(req.engine)

    weather = {
        "temperature": req.temperature,
        "pressure": req.pressure,
        "humidity": req.humidity,
        "elevation": SITE["elevation_m"],
    }

    result = monte_carlo_final_solver(
        weather=weather,
        beta=model.beta,
        sigma_H=model.sigma_H,
        m0=model.m0,
        beta_T=model.beta_T,
        engine_name=req.engine,
        n_samples=5000,
    )

    rm = float(result.get("recommended_mass", -1))
    if rm < 450 or rm > 900:
        log_access(
            request,
            "/api/final_solve",
            req.engine,
            "final",
            False,
            member,
            f"Rejected by physical guardrail: {rm:.1f} g",
        )
        raise HTTPException(
            status_code=400,
            detail=f"Final mode rejected: recommended_mass out of range ({rm:.1f} g)"
        )

    log_access(request, "/api/final_solve", req.engine, "final", True, member)
    return result
