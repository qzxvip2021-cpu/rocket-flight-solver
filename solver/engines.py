# solver/engines.py

ENGINES = {
    "F20-4W": {
        "total_impulse": 41.0,     # N·s
        "avg_thrust": 20.0,        # N
        "burn_time": 2.05,         # s
        "delay": 4.0               # s
    },
    "F20-7W": {
        "total_impulse": 41.0,
        "avg_thrust": 20.0,
        "burn_time": 2.05,
        "delay": 7.0
    }
}

# Reference engine for relative scaling
REFERENCE_ENGINE = "F20-4W"


def engine_scale(engine: str) -> float:
    """
    Relative performance scaling factor vs reference engine.

    NOTE:
    - Currently returns 1.0 for all engines.
    - This is intentional: no unvalidated thrust/impulse scaling
      is introduced without real flight data.
    """
    if engine not in ENGINES:
        raise ValueError(f"Unknown engine: {engine}")

    return 1.0


def engine_delay(engine: str) -> int:
    """
    Ejection delay time (seconds).
    """
    if engine not in ENGINES:
        raise ValueError(f"Unknown engine: {engine}")

    return int(ENGINES[engine]["delay"])


def engine_family(engine: str) -> str:
    """
    Engine family identifier.

    For now, each engine is treated as its own family.
    This keeps the interface stable for future grouping.
    """
    if engine not in ENGINES:
        raise ValueError(f"Unknown engine: {engine}")

    return engine
