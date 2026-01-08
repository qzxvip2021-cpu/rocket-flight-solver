# solver/train_from_excel.py

from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd

from .atmosphere import atmosphere_density
from .models import RHO_REF
from .engines import engine_scale
from .model_store import ModelSnapshot
from .constants import SITE


# =========================================================
# Column schema (engineering-grade)
# =========================================================

# 训练高度模型所需的“标准列名”
REQUIRED_COLS: List[str] = [
    "engine",
    "apogee_ft",
    "liftoff_mass_g",
    "pressure_hpa",
    "temperature_c",
    "humidity_pct",
]

# 时间模型列（可选，但如果想训 time 就得有）
OPTIONAL_COLS: List[str] = [
    "flight_time_s",
]

# 列名别名兼容层：你未来改 Excel 表头时，不至于把训练炸掉
# 你当前表头已经是“标准列名”，所以这层不会改变任何东西，只是更稳。
COLUMN_ALIASES: Dict[str, List[str]] = {
    "engine": ["engine", "motor", "engine_name"],
    "apogee_ft": ["apogee_ft", "apogee", "apogee(ft)", "apogee (ft)", "apogee_feet"],
    "flight_time_s": ["flight_time_s", "flight_time", "time", "flight time", "flight time (s)", "flight_time_sec"],
    "liftoff_mass_g": ["liftoff_mass_g", "liftoff_mass", "mass_g", "liftoff mass (g)"],
    "humidity_pct": ["humidity_pct", "humidity", "humidity(%)", "humidity (%)"],
    "pressure_hpa": ["pressure_hpa", "pressure", "qnh_hpa", "pressure (hpa)"],
    "temperature_c": ["temperature_c", "temperature", "temp_c", "temperature (c)", "temp (c)"],
}

# 数值列：需要强制转成 numeric（错误值转 NaN）
NUMERIC_COLS: List[str] = [
    "apogee_ft",
    "flight_time_s",
    "liftoff_mass_g",
    "humidity_pct",
    "pressure_hpa",
    "temperature_c",
]


# =========================================================
# Helpers
# =========================================================

def _utc_version() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    1) strip 表头空格
    2) 用 alias 把各种可能表头统一成标准列名
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # 构造 “alias -> canonical” 映射
    alias_to_canonical: Dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            alias_to_canonical[str(a).strip()] = canonical

    rename_map = {}
    for c in df.columns:
        if c in alias_to_canonical:
            rename_map[c] = alias_to_canonical[c]

    df = df.rename(columns=rename_map)
    return df


def _check_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Excel missing required columns: {missing}")


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    把数值列变成 float。
    - "N/A", "", "na" -> NaN
    - 其它乱七八糟的字符串 -> NaN（避免直接崩）
    """
    df = df.copy()

    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _basic_sanity_checks(df: pd.DataFrame) -> None:
    """
    工程级“防炸”检查：只对训练必须列进行。
    这里的策略是：
    - 必须列不能有 NaN
    - 必须列数值要在合理范围内（避免录入错单位）
    """
    # 必须列不能 NaN
    for c in REQUIRED_COLS:
        if df[c].isna().any():
            # 给出前几个出错行号，方便你回 Excel 找
            bad_idx = df[df[c].isna()].index.tolist()[:10]
            raise ValueError(f"Column '{c}' has NaN at rows (first 10): {bad_idx}")

    # 合理范围（你可按实际再微调）
    if (df["liftoff_mass_g"] <= 0).any():
        bad_idx = df[df["liftoff_mass_g"] <= 0].index.tolist()[:10]
        raise ValueError(f"liftoff_mass_g must be >0. Bad rows (first 10): {bad_idx}")

    # apogee_ft 合理范围：>0，且别离谱（比如录成米/英尺错位）
    if (df["apogee_ft"] <= 0).any():
        bad_idx = df[df["apogee_ft"] <= 0].index.tolist()[:10]
        raise ValueError(f"apogee_ft must be >0. Bad rows (first 10): {bad_idx}")

    # humidity 0-100
    if ((df["humidity_pct"] < 0) | (df["humidity_pct"] > 100)).any():
        bad_idx = df[((df["humidity_pct"] < 0) | (df["humidity_pct"] > 100))].index.tolist()[:10]
        raise ValueError(f"humidity_pct must be in [0,100]. Bad rows (first 10): {bad_idx}")

    # pressure hPa：给宽一点，避免高海拔/天气边界
    if ((df["pressure_hpa"] < 800) | (df["pressure_hpa"] > 1100)).any():
        bad_idx = df[((df["pressure_hpa"] < 800) | (df["pressure_hpa"] > 1100))].index.tolist()[:10]
        raise ValueError(f"pressure_hpa out of [800,1100]. Bad rows (first 10): {bad_idx}")

    # temperature C：给宽一点
    if ((df["temperature_c"] < -30) | (df["temperature_c"] > 60)).any():
        bad_idx = df[((df["temperature_c"] < -30) | (df["temperature_c"] > 60))].index.tolist()[:10]
        raise ValueError(f"temperature_c out of [-30,60]. Bad rows (first 10): {bad_idx}")


def _rho_ratio_row(row) -> float:
    rho = atmosphere_density(
        p_qnh_hpa=float(row["pressure_hpa"]),
        temperature_c=float(row["temperature_c"]),
        humidity=float(row["humidity_pct"]),
        elevation_m=float(SITE["elevation_m"]),
    )
    return float(rho / RHO_REF)

# =========================================================
# filter outliers
# =========================================================
def _filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    简单但非常有效的异常过滤：
    - apogee 偏离中位数 > 3σ
    - 小样本（<10）直接跳过，避免误删
    """
    if len(df) < 10:
        return df.copy()

    H = df["apogee_ft"].astype(float)

    H_med = H.median()
    H_std = H.std(ddof=1)

    if not np.isfinite(H_std) or H_std <= 0:
        return df.copy()

    mask = (H - H_med).abs() <= 3 * H_std
    df_clean = df[mask].copy()

    dropped = len(df) - len(df_clean)
    if dropped > 0:
    	print(f"⚠️ Dropped {dropped} outlier rows before training")



    return df_clean


# =========================================================
# Fitters
# =========================================================

def _fit_height_model(
    df_engine: pd.DataFrame,
    scale: float
) -> Tuple[np.ndarray, float, float, Dict[str, Any]]:
    """
    拟合：(H/scale) = β0 + β1*(m-m0) + β2*rho_ratio

    返回：
    - beta(3,)
    - sigma_H（在原始 H 尺度上）
    - m0
    - meta
    """
    m = df_engine["liftoff_mass_g"].astype(float).to_numpy()
    H = df_engine["apogee_ft"].astype(float).to_numpy()
    rho_ratio = df_engine["rho_ratio"].astype(float).to_numpy()

    if len(m) < 6:
        raise ValueError(f"Not enough samples for height model: {len(m)} (<6)")

    m0 = float(np.median(m))
    m_c = m - m0

    X = np.column_stack([np.ones_like(m_c), m_c, rho_ratio])
    y = H / scale

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    H_hat = (X @ beta) * scale
    resid = H - H_hat

    sigma_H = float(np.sqrt(np.sum(resid**2) / max(len(H) - 3, 1)))

    meta = {
        "n_height": int(len(H)),
        "rmse_H": float(np.sqrt(np.mean(resid**2))),
        "sigma_H": float(sigma_H),
    }
    return beta.astype(float), float(sigma_H), float(m0), meta


def _fit_time_model(df_engine: pd.DataFrame) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
    """
    拟合：T = c0 + c1*H

    - 仅对 flight_time_s 非空样本拟合
    - 样本不足（<6）返回 None（保持“宁缺毋滥”）
    """
    if "flight_time_s" not in df_engine.columns:
        return None

    df_t = df_engine.dropna(subset=["flight_time_s"]).copy()
    if len(df_t) < 6:
        return None

    H = df_t["apogee_ft"].astype(float).to_numpy()
    T = df_t["flight_time_s"].astype(float).to_numpy()

    X = np.column_stack([np.ones_like(H), H])
    beta_T, *_ = np.linalg.lstsq(X, T, rcond=None)

    T_hat = X @ beta_T
    resid = T - T_hat

    meta = {
        "n_time": int(len(T)),
        "rmse_T": float(np.sqrt(np.mean(resid**2))),
    }
    return beta_T.astype(float), meta


# =========================================================
# Public API
# =========================================================

def train_model_from_excel(excel_path: str, engine_name: str = "F20-4W") -> ModelSnapshot:
    """
    读 Excel -> 清洗/校验 -> 训练 -> 返回 ModelSnapshot
    """
    df = pd.read_excel(excel_path)

    # ① 工程级：列名规范化（alias -> canonical）
    df = _normalize_columns(df)

    # ② 检查必须列
    _check_columns(df)

    # ③ 类型清洗（把数值列转成 float）
    df = _coerce_numeric(df)

    # ④ 字段规范化
    df["engine"] = df["engine"].astype(str).str.strip()
    
    #只取指定发动机训练
    df_e = df[df["engine"] == engine_name].copy()
    if len(df_e) == 0:
        raise ValueError(f"No rows for engine={engine_name} in Excel")

    # ⑤ 工程级：训练前最小必要健全性检查（只检查训练必须列）
    _basic_sanity_checks(df_e)

    # ⑥ 异常值过滤（小样本会自动跳过）
    df_e = _filter_outliers(df_e)
    
    # ⑦ 预计算 rho_ratio
    df_e["rho_ratio"] = df_e.apply(_rho_ratio_row, axis=1)

    # ⑧ 训练
    scale = float(engine_scale(engine_name))
    beta, sigma_H, m0, meta_H = _fit_height_model(df_e, scale=scale)

    time_fit = _fit_time_model(df_e)
    beta_T = None
    meta_T = {}
    if time_fit is not None:
        beta_T, meta_T = time_fit

    # ⑨ meta：记录关键信息（方便你 admin/reload 后查看）
    meta = {
        "engine_scale": float(scale),
        "excel_path": str(excel_path),
        "engine_name": str(engine_name),
        **meta_H,
        **meta_T,
    }

    # ⑩ 最基本的健全性检查（避免 NaN 模型上线）
    if not np.all(np.isfinite(beta)):
        raise ValueError("beta contains non-finite values")
    if sigma_H <= 0 or not np.isfinite(sigma_H):
        raise ValueError(f"Invalid sigma_H: {sigma_H}")
    if beta_T is not None and (not np.all(np.isfinite(beta_T))):
        raise ValueError("beta_T contains non-finite values")

    return ModelSnapshot(
        engine_name=engine_name,
        beta=beta,
        sigma_H=float(sigma_H),
        m0=float(m0),
        beta_T=beta_T,
        version=_utc_version(),
        meta=meta,
    )
    