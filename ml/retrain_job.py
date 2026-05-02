"""Офлайн-переобучение моделей на NHANES CSV + фидбек из БД.

Сохраняет: scaler.joblib, rfecv_catboost.cbm (standard), smoteenn_xgboost.pkl (aggressive),
ensemble_optuna.pkl (risk_group: CatBoost RFE + SMOTEENN XGB/LGB + Optuna веса/tilt).

Запуск: python -m ml.retrain_job (из корня проекта).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from xgboost import XGBClassifier

from app.models import Prediction
from app.schemas import PredictionRequest
from ml.ensemble_retrain import fit_and_save_ensemble_bundle
from ml.preprocessing import (
    FEATURE_COLS,
    add_engineered_features_batch,
    build_imputed_feature_frame,
    impute_training_features_median,
)

logger = logging.getLogger(__name__)

_ML_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _ML_DIR / "models"
_DEFAULT_CSV = _ML_DIR / "processed" / "nhanes_ml_ready.csv"
_SYNC_DB_URL = "sqlite:///./app.db"


def _split_stored_input(
    raw_input_data: object,
) -> tuple[dict[str, Any], dict[str, float]]:
    if not isinstance(raw_input_data, dict):
        return {}, {}

    raw_user_input_data = raw_input_data.get("user_input_data")
    raw_computed_data = raw_input_data.get("computed_data")

    if isinstance(raw_user_input_data, dict) and isinstance(raw_computed_data, dict):
        user_input_data = {
            str(key): value
            for key, value in raw_user_input_data.items()
            if value is None or isinstance(value, (int, float))
        }
        computed_data = {
            str(key): float(value)
            for key, value in raw_computed_data.items()
            if isinstance(value, (int, float))
        }
        return user_input_data, computed_data

    from ml.preprocessing import DERIVED_MODEL_FEATURES

    user_input_data: dict[str, Any] = {}
    computed_data: dict[str, float] = {}
    for key, value in raw_input_data.items():
        if value is not None and not isinstance(value, (int, float)):
            continue
        if key in DERIVED_MODEL_FEATURES and isinstance(value, (int, float)):
            computed_data[str(key)] = float(value)
            continue
        user_input_data[str(key)] = value
    return user_input_data, computed_data


def _load_xy_from_csv(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    raw = pd.read_csv(path)
    if "target" not in raw.columns:
        raise ValueError(f"В {path} нет колонки target")
    y = raw["target"].astype(int)
    drop_cols = [c for c in ("SEQN", "target") if c in raw.columns]
    base = raw.drop(columns=drop_cols)
    x_eng = add_engineered_features_batch(base)
    for col in FEATURE_COLS:
        if col not in x_eng.columns:
            x_eng[col] = np.nan
    x_eng = x_eng[FEATURE_COLS].reset_index(drop=True)
    y = y.reset_index(drop=True)
    valid = x_eng["LBXGLU"].notna()
    x_eng = x_eng.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    x_eng = impute_training_features_median(x_eng)
    return x_eng, y


def _load_xy_from_feedback(engine_url: str) -> tuple[pd.DataFrame, pd.Series]:
    engine = create_engine(engine_url)
    rows_list: list[pd.Series] = []
    y_list: list[int] = []
    with Session(engine) as session:
        preds = session.execute(
            select(Prediction).where(
                Prediction.feedback_at.isnot(None),
                Prediction.real_class.isnot(None),
            )
        ).scalars().all()

    for p in preds:
        user_input_data, _ = _split_stored_input(p.input_data)
        payload: dict[str, Any] = {"name": "", "mode": "standard", **user_input_data}
        try:
            body = PredictionRequest.model_validate(payload)
        except Exception:
            continue
        try:
            df = build_imputed_feature_frame(body)
        except Exception:
            continue
        rows_list.append(df.iloc[0])
        y_list.append(int(p.real_class))

    if not rows_list:
        return pd.DataFrame(columns=FEATURE_COLS), pd.Series(dtype=int)
    x_fb = pd.DataFrame(rows_list)[FEATURE_COLS]
    y_fb = pd.Series(y_list, dtype=int)
    return x_fb.reset_index(drop=True), y_fb.reset_index(drop=True)


def _atomic_replace(tmp: Path, final: Path) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp, final)


def _write_joblib(path: Path, obj: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(obj, tmp)
    _atomic_replace(tmp, path)


def _write_catboost(path: Path, model: CatBoostClassifier) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    model.save_model(str(tmp))
    _atomic_replace(tmp, path)


def run_retrain(
    *,
    csv_path: Path | None = None,
    db_url: str = _SYNC_DB_URL,
) -> None:
    csv_path = csv_path or _DEFAULT_CSV
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    x_csv, y_csv = _load_xy_from_csv(csv_path)
    x_fb, y_fb = _load_xy_from_feedback(db_url)

    x_all = pd.concat([x_csv, x_fb], ignore_index=True)
    y_all = pd.concat([y_csv, y_fb], ignore_index=True)
    mask = y_all.isin([0, 1, 2])
    x_all = x_all.loc[mask].reset_index(drop=True)
    y_all = y_all.loc[mask].astype(int).reset_index(drop=True)

    logger.info(
        "Обучение: NHANES=%s, фидбек=%s, всего=%s",
        len(x_csv),
        len(x_fb),
        len(x_all),
    )
    if len(x_all) < 100:
        logger.warning("Мало объектов (%s), качество может быть нестабильным", len(x_all))

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_all.to_numpy(dtype=float))
    y_arr = y_all.to_numpy()

    cb = CatBoostClassifier(
        iterations=800,
        depth=6,
        learning_rate=0.05,
        loss_function="MultiClass",
        random_seed=42,
        verbose=False,
        auto_class_weights="Balanced",
    )
    cb.fit(x_scaled, y_arr)

    smote_enn = SMOTEENN(random_state=42)
    x_res, y_res = smote_enn.fit_resample(x_scaled, y_arr)

    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        objective="multi:softprob",
        num_class=3,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    xgb.fit(x_res, y_res)

    try:
        fit_and_save_ensemble_bundle(
            x_scaled=x_scaled,
            y_arr=y_arr,
            ensemble_pkl=_MODELS_DIR / "ensemble_optuna.pkl",
        )
    except Exception:
        logger.exception("Пересборка ensemble_optuna.pkl не удалась; остальные артефакты будут сохранены")

    _write_joblib(_MODELS_DIR / "scaler.joblib", scaler)
    _write_catboost(_MODELS_DIR / "rfecv_catboost.cbm", cb)
    _write_joblib(_MODELS_DIR / "smoteenn_xgboost.pkl", xgb)

    logger.info(
        "Сохранено: scaler.joblib, rfecv_catboost.cbm, smoteenn_xgboost.pkl (+ ensemble при успехе) в %s",
        _MODELS_DIR,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_retrain()


if __name__ == "__main__":
    main()
