ENGINES = {
    "F20-4W": {
        "total_impulse": 41.0,     # N·s
        "avg_thrust": 20.0,        # N
        "burn_time": 2.05,         # s
        "delay": 4.0               # s
    },
    "F23-4FJ": {
        "total_impulse": 48.2,
        "avg_thrust": 23.0,
        "burn_time": 2.2,
        "delay": 4.0
    },
    "F42-4T": {
        "total_impulse": 55.0,
        "avg_thrust": 42.0,
        "burn_time": 1.3,
        "delay": 4.0
    },
    "F52-8C": {
        "total_impulse": 66.2,
        "avg_thrust": 52.0,
        "burn_time": 1.3,
        "delay": 8.0
    }
}
REFERENCE_ENGINE = "F20-4W"


def engine_scale(engine_name: str) -> float:
    ref = ENGINES[REFERENCE_ENGINE]["total_impulse"]
    return ENGINES[engine_name]["total_impulse"] / ref