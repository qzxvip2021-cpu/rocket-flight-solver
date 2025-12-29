import numpy as np
from scipy.stats import norm

from .constants import MIN_LIFTOFF_MASS, MAX_LIFTOFF_MASS, H_WINDOW
from .engines import engine_scale
from .atmosphere import atmosphere_density
from .models import RHO_REF

# sanity check
_engine_test = engine_scale("F20-4W")
assert isinstance(_engine_test, (int, float))


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
    Solve optimal liftoff mass given target apogee.

    Height model:
        H = (β0 + β1*(m-m0) + β2*rho_ratio) * engine_scale

    Time model (ONLY valid for F20-4W):
        T = c0 + c1 * H
    """

    # ---------- air density ----------
    rho = atmosphere_density(
        p_qnh_hpa=weather["pressure"],
        temperature_c=weather["temperature"],
        humidity=weather["humidity"],
        elevation_m=weather["elevation"],
    )
    rho_ratio = rho / RHO_REF

    # ---------- target window ----------
    H_min = target_apogee - H_WINDOW
    H_max = target_apogee + H_WINDOW

    # ---------- engine factor ----------
    scale = engine_scale(engine_name)

    best = None
    best_prob = -1.0
    best_H = None

    # ---------- scan liftoff mass ----------
    for m in range(int(MIN_LIFTOFF_MASS), int(MAX_LIFTOFF_MASS) + 1):
        m_c = m - m0
        x = np.array([1.0, m_c, rho_ratio], dtype=float)

        H = float(x @ beta) * scale

        P = (
            norm.cdf((H_max - H) / sigma_H)
            - norm.cdf((H_min - H) / sigma_H)
        )

        if P > best_prob:
            best_prob = P
            best_H = H
            best = {
                "engine": engine_name,
                "mass": float(m),
                "H_mean": round(H, 1),
                "confidence": round(P * 100, 1),
            }

    # ---------- flight time ----------
    if best is not None and best_H is not None:
        if engine_name == "F20-4W":
            # Only F20-4W has valid time model
            best["flight_time"] = round(
                float(np.array([1.0, best_H]) @ beta_T),
                2
            )
        else:
            best["flight_time"] = None

    return best
