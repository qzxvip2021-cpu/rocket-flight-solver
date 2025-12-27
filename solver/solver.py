import numpy as np
from scipy.stats import norm

from .constants import MIN_LIFTOFF_MASS, MAX_LIFTOFF_MASS, H_WINDOW
from .engines import engine_scale
from .atmosphere import atmosphere_density
from .models import RHO_REF


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
    weather keys:
      temperature (°C)
      pressure (hPa)
      humidity (%)
      elevation (m)
    """

    # 1) compute rho and rho_ratio
    rho = atmosphere_density(
        p_qnh_hpa=weather["pressure"],
        temperature_c=weather["temperature"],
        humidity=weather["humidity"],
        elevation_m=weather["elevation"],
    )
    rho_ratio = rho / RHO_REF

    # 2) scan mass
    H_min = target_apogee - H_WINDOW
    H_max = target_apogee + H_WINDOW

    best = None
    best_prob = -1.0
    best_H_mean = None

    scale = engine_scale(engine_name)

    for m in range(int(MIN_LIFTOFF_MASS), int(MAX_LIFTOFF_MASS) + 1):
        m_c = m - m0
        x = np.array([1.0, m_c, rho_ratio], dtype=float)

        H_mean = float(x @ beta) * scale

        z1 = (H_min - H_mean) / sigma_H
        z2 = (H_max - H_mean) / sigma_H
        P = float(norm.cdf(z2) - norm.cdf(z1))

        if P > best_prob:
            best_prob = P
            best_H_mean = H_mean
            best = {
                "engine": engine_name,
                "mass": float(m),
                "H_mean": float(H_mean),
                "confidence": float(P * 100.0),
            }

    # 3) flight time (still using your linear time model)
    if best is not None and best_H_mean is not None:
        t_pred = float(np.array([1.0, best_H_mean]) @ beta_T)
        best["flight_time"] = t_pred

    return best
