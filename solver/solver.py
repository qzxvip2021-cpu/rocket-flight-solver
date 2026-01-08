import numpy as np
from scipy.stats import norm

from .constants import MIN_LIFTOFF_MASS, MAX_LIFTOFF_MASS, H_WINDOW
from .engines import engine_scale
from .atmosphere import atmosphere_density
from .models import RHO_REF


# sanity check
_engine_test = engine_scale("F20-4W")
assert isinstance(_engine_test, (int, float))


def _arc_time_score(T: float) -> float:
    """
    ARC time score:
      - if 36 <= T <= 39: 0
      - if T < 36: (36 - T) * 4
      - if T > 39: (T - 39) * 4
    """
    if T < 36.0:
        return (36.0 - T) * 4.0
    if T > 39.0:
        return (T - 39.0) * 4.0
    return 0.0


def solve_mass(
    weather: dict,
    target_apogee: float,
    beta,
    sigma_H: float,
    m0: float,
    beta_T,
    engine_name: str,
):
    """
    Normal Mode solver (ARC Practice):

    Height model:
        H = (β0 + β1*(m-m0) + β2*rho_ratio) * engine_scale

    Time model (beta_T):
        T = c0 + c1 * H
    IMPORTANT:
        beta_T is trained on OFFICIAL ARC time already (includes delay),
        so DO NOT add delay again in Normal Mode.

    Selection policy (Time Score Priority):
      1) minimize time_score
      2) then minimize height_score (= |H - target|)
      3) then maximize hit probability within target window (P)
    """

    # ======================================================
    # 1. Air density
    # ======================================================
    rho = atmosphere_density(
        p_qnh_hpa=weather["pressure"],
        temperature_c=weather["temperature"],
        humidity=weather["humidity"],
        elevation_m=weather["elevation"],
    )
    rho_ratio = rho / RHO_REF

    # ======================================================
    # 2. Target window (for probability only)
    # ======================================================
    H_min = target_apogee - H_WINDOW
    H_max = target_apogee + H_WINDOW

    # ======================================================
    # 3. Engine scale
    # ======================================================
    scale = float(engine_scale(engine_name))

    best = None
    best_key = None  # (time_score, height_score, -P)

    # ======================================================
    # 4. Scan liftoff mass
    # ======================================================
    for m in range(int(MIN_LIFTOFF_MASS), int(MAX_LIFTOFF_MASS) + 1):
        m_c = float(m - m0)
        x = np.array([1.0, m_c, float(rho_ratio)], dtype=float)

        # predicted apogee
        H = float(x @ beta) * scale

        # probability of hitting target window (normal assumption)
        P = norm.cdf((H_max - H) / sigma_H) - norm.cdf((H_min - H) / sigma_H)

        # predicted flight time: OFFICIAL ARC time (already includes delay)
        T = None
        if beta_T is not None:
            T = float(np.array([1.0, H], dtype=float) @ beta_T)

        # scores
        height_score = abs(H - float(target_apogee))
        time_score = _arc_time_score(T) if T is not None else float("inf")
        key = (float(time_score), float(height_score), -float(P))

       
        if best_key is None or key < best_key:
            best_key = key
            best = {
                "engine": engine_name,
                "mass": float(m),
                "H_mean": round(H, 1),
                "confidence": round(float(P) * 100.0, 1),
                "flight_time": round(float(T), 2) if T is not None else None,
                # expose scores for debugging
                "height_score": round(float(height_score), 1),
                "time_score": round(float(time_score), 1)
                if np.isfinite(time_score)
                else None,
                "arc_score": round(float(height_score + time_score), 1)
                if np.isfinite(time_score)
                else None,
            }

    return best
