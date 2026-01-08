# solver/model_store.py

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Optional, Dict, Any

import numpy as np


@dataclass(frozen=True)
class ModelSnapshot:
    """
    一次模型“快照”。训练完成后整体替换，避免半更新。
    """
    engine_name: str
    beta: np.ndarray              # shape (3,)
    sigma_H: float
    m0: float
    beta_T: Optional[np.ndarray]  # shape (2,) or None
    version: str
    meta: Dict[str, Any]


_lock = Lock()

# 多发动机模型仓库
_models: Dict[str, ModelSnapshot] = {}


def set_model(engine: str, new_model: ModelSnapshot) -> None:
    """
    原子写入 / 更新某一个发动机的模型。
    """
    if engine != new_model.engine_name:
        raise ValueError(
            f"Engine mismatch: key={engine}, model.engine_name={new_model.engine_name}"
        )

    with _lock:
        _models[engine] = new_model


def get_model(engine: str) -> ModelSnapshot:
    """
    按发动机获取模型。
    """
    model = _models.get(engine)
    if model is None:
        raise RuntimeError(f"Model for engine '{engine}' is not loaded yet.")
    return model


def get_model_info() -> Dict[str, Any]:
    """
    返回当前已加载的所有发动机模型信息。
    """
    return {
        engine: {
            "engine": m.engine_name,
            "version": m.version,
            "sigma_H": m.sigma_H,
            "m0": m.m0,
            "has_time_model": m.beta_T is not None,
            "meta": m.meta,
        }
        for engine, m in _models.items()
    }
