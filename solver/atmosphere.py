import numpy as np


def air_density(pressure_hpa, temperature_c, humidity_pct):
    T = temperature_c + 273.15
    p = pressure_hpa * 100.0

    es = 610.94 * np.exp(17.625 * temperature_c / (temperature_c + 243.04))
    e = (humidity_pct / 100.0) * es

    Rd = 287.05
    Rv = 461.5
    rho = (p - e) / (Rd * T) + e / (Rv * T)
    return rho


def qnh_to_station_pressure(p_qnh_hpa, temp_c, elevation_m):
    T = temp_c + 273.15
    g = 9.80665
    R = 287.05
    return p_qnh_hpa * 100.0 * np.exp(-g * elevation_m / (R * T))


def atmosphere_density(p_qnh_hpa, temperature_c, humidity, elevation_m):
    p_station = qnh_to_station_pressure(p_qnh_hpa, temperature_c, elevation_m)
    return air_density(p_station / 100.0, temperature_c, humidity)
