# solver/model_store.py

from __future__ import annotations
from dataclasses import dataclass
from threading import Lock
from typing import Optional, Dict, Any

import numpy as np


@dataclass(frozen=True)
class ModelSnapshot:
    """
    一次模型“快照”。训练完后整体替换，避免半更新。
    """
    engine_name: str
    beta: np.ndarray        # shape (3,)
    sigma_H: float
    m0: float
    beta_T: Optional[np.ndarray]  # shape (2,) or None
    version: str            # 例如 "2025-12-28T12:34:56Z"
    meta: Dict[str, Any]    # 训练样本数、残差等信息


_lock = Lock()
_current: Optional[ModelSnapshot] = None


def get_model() -> ModelSnapshot:
    """
    供 /api/solve /api/curve 调用。
    """
    global _current
    if _current is None:
        raise RuntimeError("Model is not loaded yet.")
    return _current


def set_model(new_model: ModelSnapshot) -> None:
    """
    原子替换模型。
    """
    global _current
    with _lock:
        _current = new_model


def get_model_info() -> Dict[str, Any]:
    m = get_model()
    return {
        "engine": m.engine_name,
        "version": m.version,
        "sigma_H": m.sigma_H,
        "m0": m.m0,
        "has_time_model": m.beta_T is not None,
        "meta": m.meta,
    }
