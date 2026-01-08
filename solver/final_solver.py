import numpy as np

from .constants import MIN_LIFTOFF_MASS, MAX_LIFTOFF_MASS
from .engines import engine_scale
from .atmosphere import atmosphere_density
from .models import RHO_REF


def arc_final_score(H, T, H_target):
    """
    ARC Finals scoring rule
    All inputs can be numpy arrays
    """
    height_score = np.abs(H - H_target)

    time_score = np.zeros_like(height_score)
    time_score[T > 39] = np.abs(T[T > 39] - 39) * 4
    time_score[T < 36] = np.abs(T[T < 36] - 36) * 4

    return height_score + time_score


def monte_carlo_final_solver(
    weather: dict,
    beta,
    sigma_H: float,
    m0: float,
    beta_T,
    engine_name: str,
    n_samples: int = 5000,
):
    """
    Monte Carlo ARC Finals solver (with sigma_H uncertainty)

    Returns:
      - mass: list[int]
      - expected_score: list[float]
      - recommended_mass: float
      - best_score: float
    """

    # ---------- atmosphere ----------
    rho = atmosphere_density(
        p_qnh_hpa=weather["pressure"],
        temperature_c=weather["temperature"],
        humidity=weather["humidity"],
        elevation_m=weather["elevation"],
    )
    rho_ratio = rho / RHO_REF
    scale = engine_scale(engine_name)

    masses = np.arange(int(MIN_LIFTOFF_MASS), int(MAX_LIFTOFF_MASS) + 1)
    expected_scores = []

    # Monte Carlo target heights (ARC Finals rule)
    targets = np.random.uniform(725, 775, size=n_samples)

    for m in masses:
        m_c = m - m0

        # ---------- height mean ----------
        H_mean = float(np.array([1.0, m_c, rho_ratio]) @ beta) * scale

        # ---------- Monte Carlo height ----------
        H_samples = np.random.normal(
            loc=H_mean,
            scale=sigma_H,
            size=n_samples
        )

        # ---------- time model ----------
        if beta_T is not None:
            T_samples = beta_T[0] + beta_T[1] * H_samples
        else:
            T_samples = np.zeros_like(H_samples)

        scores = arc_final_score(H_samples, T_samples, targets)
        expected_scores.append(scores.mean())

    expected_scores = np.array(expected_scores)
    best_idx = np.argmin(expected_scores)

    return {
        "mass": masses.tolist(),
        "expected_score": expected_scores.tolist(),
        "recommended_mass": float(masses[best_idx]),
        "best_score": float(expected_scores[best_idx]),
    }
