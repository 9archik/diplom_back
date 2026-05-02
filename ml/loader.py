from __future__ import annotations

import joblib
from pathlib import Path

from catboost import CatBoostClassifier

from ml.model_adapters import adapt_loaded_model
from ml.pickle_compat import register_pickle_compat

_ML_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _ML_DIR / "models"

_models: dict[str, object] = {}
_scaler: object | None = None
_load_errors: dict[str, str] = {}


class ModelNotReadyError(RuntimeError):
    pass


def load_all() -> None:
    global _scaler
    _models.clear()
    _load_errors.clear()
    _scaler = None
    register_pickle_compat()

    try:
        cb_model = CatBoostClassifier()
        cb_model.load_model(str(_MODELS_DIR / "rfecv_catboost.cbm"))
        _models["standard"] = cb_model
    except Exception as exc:
        _load_errors["standard"] = str(exc)

    try:
        loaded_risk_group = joblib.load(_MODELS_DIR / "ensemble_optuna.pkl")
        _models["risk_group"] = adapt_loaded_model("risk_group", loaded_risk_group)
    except Exception as exc:
        _load_errors["risk_group"] = str(exc)

    try:
        _models["aggressive"] = joblib.load(_MODELS_DIR / "smoteenn_xgboost.pkl")
    except Exception as exc:
        _load_errors["aggressive"] = str(exc)

    try:
        _scaler = joblib.load(_MODELS_DIR / "scaler.joblib")
    except Exception as exc:
        _load_errors["scaler"] = str(exc)


def get_model(mode: str):
    model = _models.get(mode)
    if model is None:
        error_details = _load_errors.get(mode, "unknown load error")
        raise ModelNotReadyError(f"Model '{mode}' is not available: {error_details}")
    return model


def get_scaler():
    if _scaler is None:
        error_details = _load_errors.get("scaler", "unknown load error")
        raise ModelNotReadyError(f"Scaler is not available: {error_details}")
    return _scaler
