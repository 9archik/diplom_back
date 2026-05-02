from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_current_user, get_db_session
from app.models import Prediction, User
from app.retrain_counter import record_feedback_for_retrain, schedule_retrain_subprocess
from app.schemas import (
    FactorItem,
    PredictionFeedbackRequest,
    PredictionInputValue,
    PredictionListItemResponse,
    PredictionMode,
    PredictionModeUpdateRequest,
    PredictionRequest,
    PredictionResponse,
)
from ml.loader import ModelNotReadyError, get_model, get_scaler
from ml.model_adapters import RiskGroupEnsembleModel
from ml.preprocessing import (
    ATOMIC_FEATURE_COLS,
    DERIVED_MODEL_FEATURES,
    FEATURE_COLS,
    build_imputed_feature_frame,
    extract_clinical_derived_for_response,
)

router = APIRouter(prefix="/predictions", tags=["predictions"])

FEATURE_LABELS = {
    "RIDAGEYR": "Возраст",
    "RIAGENDR": "Пол",
    "BMXBMI": "ИМТ",
    "BMXHT": "Рост",
    "BMXWAIST": "Окружность талии",
    "hypertension": "Гипертензия",
    "LBXTC": "Общий холестерин",
    "LBDHDD": "ЛПВП",
    "LBXGLU": "Глюкоза натощак",
    "LBXTR": "Триглицериды",
    "DMDEDUC2": "Образование",
    "LBXSAT": "АЛТ",
    "LBXSASS": "АСТ",
    "LBXSUA": "Мочевая кислота",
    "LBXVIDMS": "Витамин D",
    "eGFR": "рСКФ",
    "LBXSBU": "Мочевина",
    "URXUMA": "Микроальбумин мочи",
    "LBXHGB": "Гемоглобин",
    "LBXRBCSI": "Эритроциты",
    "LBXSTB": "Билирубин",
    "LBXSAL": "Альбумин",
    "LBXSAPSI": "Щелочная фосфатаза",
    "LBXWBCSI": "Лейкоциты",
    "LBXLYPCT": "Лимфоциты",
    "LBXMCVSI": "MCV",
    "LBXRDW": "RDW",
    "LBXCRP": "СРБ",
    "MCQ300C": "Семейный анамнез диабета",
    "MCQ160A": "Инфаркт в анамнезе",
    "MCQ160E": "Инсульт в анамнезе",
    "MCQ160F": "ХСН в анамнезе",
    "HSD010": "Самооценка здоровья",
    "WHtR": "Талия / Рост",
    "Age_x_WC": "Возраст × Талия",
    "Age_x_Glucose": "Возраст × Глюкоза",
    "TyG_Index": "Индекс TyG",
    "TyG_WC": "TyG × Талия",
    "METS_IR": "Индекс METS-IR",
    "ALT_AST_Ratio": "АЛТ/АСТ",
    "Log_CRP": "Log(СРБ)",
    "Log_SUA": "Log(Мочевая кислота)",
    "PhenoAge_accel": "Биологический возраст (ускорение)",
}

NORMAL_RANGES = {
    "LBXGLU": "70–99 мг/дл",
    "BMXBMI": "18.5–25 кг/м²",
    "BMXWAIST": "<80 (ж) / <94 (м) см",
    "LBDHDD": ">40 (м) / >50 (ж) мг/дл",
    "LBXTR": "<150 мг/дл",
    "LBXTC": "<200 мг/дл",
    "eGFR": ">60 мл/мин/1.73м²",
    "LBXCRP": "<0.5 мг/дл",
    "LBXSAL": "3.5–5.0 г/дл",
    "LBXHGB": "13.5–17.5 (м) / 12.0–15.5 (ж) г/дл",
    "LBXRDW": "11.5–14.5%",
    "URXUMA": "<30 мкг/мл",
    "WHtR": "<0.5",
    "PhenoAge_accel": "0 (норма < 0)",
}

CLASS_LABELS = {0: "Норма", 1: "Преддиабет", 2: "Диабет"}
CLASS_ORDER: tuple[int, ...] = (0, 1, 2)
_VALIDATION_LOC_SKIP = frozenset({"body", "query", "path"})


def _missing_field_names_from_validation_error(exc: ValidationError) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for err in exc.errors():
        if err.get("type") != "missing":
            continue
        loc = err.get("loc") or ()
        for part in reversed(loc):
            if isinstance(part, str) and part not in _VALIDATION_LOC_SKIP:
                if part not in seen:
                    seen.add(part)
                    out.append(part)
                break
    return out


def _validation_error_brief_details(exc: ValidationError, *, limit: int = 20) -> list[str]:
    lines: list[str] = []
    for err in exc.errors()[:limit]:
        loc = err.get("loc") or ()
        tail = next(
            (
                p
                for p in reversed(loc)
                if isinstance(p, str) and p not in _VALIDATION_LOC_SKIP
            ),
            None,
        )
        msg = str(err.get("msg", ""))
        lines.append(f"{tail}: {msg}" if tail else msg)
    return lines


def _mode_switch_validation_detail(exc: ValidationError) -> dict[str, object]:
    return {
        "message": "Недостаточно исходных данных для смены режима у этого предсказания",
        "missing_fields": _missing_field_names_from_validation_error(exc),
        "details": _validation_error_brief_details(exc),
    }


def _probabilities_by_class_index(raw: object) -> dict[int, float]:
    """Нормализует сохранённые вероятности к индексам классов 0, 1, 2."""
    if not isinstance(raw, dict):
        return {i: 0.0 for i in CLASS_ORDER}
    by_idx: dict[int, float] = {}
    label_to_idx = {v: k for k, v in CLASS_LABELS.items()}
    for key, value in raw.items():
        if not isinstance(value, (int, float)):
            continue
        idx: int | None = None
        if isinstance(key, str):
            if key.isdigit():
                idx = int(key)
            elif key in label_to_idx:
                idx = label_to_idx[key]
        if idx is None or idx not in CLASS_LABELS:
            continue
        by_idx[idx] = float(value)
    if not by_idx:
        return {i: 0.0 for i in CLASS_ORDER}
    return {i: float(by_idx.get(i, 0.0)) for i in CLASS_ORDER}


def _probabilities_for_storage(proba_row: np.ndarray) -> dict[str, float]:
    """В БД — ключи \"0\"..\"2\" (стабильно для старых клиентов и миграций)."""
    row = np.asarray(proba_row, dtype=np.float64).reshape(-1)
    s = float(row.sum())
    if s > 0:
        row = row / s
    return {str(i): round(float(row[i]), 6) for i in CLASS_ORDER if i < row.shape[0]}


def _probabilities_for_response(stored: object) -> dict[str, float]:
    """В API — человекочитаемые ключи в порядке Норма → Преддиабет → Диабет."""
    by_idx = _probabilities_by_class_index(stored)
    total = sum(by_idx.values())
    if total > 0:
        by_idx = {i: by_idx[i] / total for i in CLASS_ORDER}
    return {CLASS_LABELS[i]: round(by_idx[i], 6) for i in CLASS_ORDER}


def _split_input_data(
    raw_input_data: object,
) -> tuple[dict[str, PredictionInputValue], dict[str, float]]:
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

    user_input_data: dict[str, PredictionInputValue] = {}
    computed_data: dict[str, float] = {}
    for key, value in raw_input_data.items():
        if value is not None and not isinstance(value, (int, float)):
            continue
        if key in DERIVED_MODEL_FEATURES and isinstance(value, (int, float)):
            computed_data[key] = float(value)
            continue
        user_input_data[key] = value
    return user_input_data, computed_data


def _to_prediction_response(
    prediction: Prediction,
    top_factors: list[FactorItem],
) -> PredictionResponse:
    user_input_data, computed_data = _split_input_data(prediction.input_data)
    return PredictionResponse(
        id=prediction.id,
        name=prediction.name,
        mode=prediction.mode,
        user_input_data=user_input_data,
        computed_data=computed_data,
        has_feedback=prediction.feedback_at is not None,
        predicted_class=prediction.predicted_class,
        class_label=CLASS_LABELS[prediction.predicted_class],
        probabilities=_probabilities_for_response(prediction.probabilities),
        top_factors=top_factors,
        created_at=prediction.created_at.isoformat(),
    )


def _calculate_prediction_result(
    body: PredictionRequest,
) -> tuple[int, dict[str, float], list[FactorItem], dict[str, float]]:
    df = build_imputed_feature_frame(body)
    X = get_scaler().transform(df)
    model = get_model(body.mode)

    proba_matrix = np.asarray(model.predict_proba(X), dtype=np.float64)
    if proba_matrix.ndim == 1:
        proba_matrix = proba_matrix.reshape(1, -1)
    proba_row = proba_matrix[0]
    ps = float(proba_row.sum())
    if ps > 0:
        proba_row = proba_row / ps
    predicted_class = int(np.argmax(proba_row))
    probabilities = _probabilities_for_storage(proba_row)
    clinical_derived = extract_clinical_derived_for_response(df)

    top_factors: list[FactorItem] = []
    atomic_indices = [
        i for i, name in enumerate(FEATURE_COLS) if name in ATOMIC_FEATURE_COLS
    ]
    try:
        import shap

        shap_for_class: np.ndarray | None = None
        if isinstance(model, RiskGroupEnsembleModel):
            shap_for_class = model.approx_shap_per_full_feature(X, predicted_class)
        else:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            shap_for_class = _extract_shap_for_class(shap_values, predicted_class)

        if shap_for_class is not None and atomic_indices:
            sub = shap_for_class[atomic_indices]
            top_local = np.argsort(np.abs(sub))[::-1][:5]
            for j in top_local:
                idx = atomic_indices[int(j)]
                feat = FEATURE_COLS[idx]
                top_factors.append(
                    FactorItem(
                        feature=feat,
                        label=FEATURE_LABELS.get(feat, feat),
                        value=round(float(df.iloc[0][feat]), 6),
                        normal_range=NORMAL_RANGES.get(feat, "—"),
                        shap_value=round(float(shap_for_class[idx]), 4),
                    )
                )
    except Exception:
        top_factors = []

    return predicted_class, probabilities, top_factors, clinical_derived


def _build_prediction_request_for_mode(
    prediction: Prediction,
    mode: PredictionMode,
) -> PredictionRequest:
    user_input_data, _ = _split_input_data(prediction.input_data)
    payload: dict[str, Any] = {
        "name": prediction.name,
        "mode": mode,
        **user_input_data,
    }
    return PredictionRequest.model_validate(payload)


def _extract_shap_for_class(shap_values, predicted_class: int) -> np.ndarray:
    if isinstance(shap_values, list):
        return np.array(shap_values[predicted_class])[0]

    shap_array = np.array(shap_values)
    if shap_array.ndim == 3:
        return shap_array[0, :, predicted_class]
    if shap_array.ndim == 2:
        return shap_array[0]
    raise ValueError("Unsupported SHAP output shape")


@router.post("/", response_model=PredictionResponse)
async def create_prediction(
    body: PredictionRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
) -> PredictionResponse:
    try:
        predicted_class, probabilities, top_factors, computed_data = (
            _calculate_prediction_result(body)
        )
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None

    user_input_data = body.model_dump(exclude={"name", "mode"})

    input_data: dict[str, Any] = {
        "user_input_data": user_input_data,
        "computed_data": computed_data,
    }

    prediction = Prediction(
        user_id=current_user.id,
        name=body.name,
        mode=body.mode,
        input_data=input_data,
        predicted_class=predicted_class,
        probabilities=probabilities,
        top_factors=[factor.model_dump() for factor in top_factors],
    )
    db.add(prediction)
    await db.commit()
    await db.refresh(prediction)

    return _to_prediction_response(prediction, top_factors)


@router.get("/{prediction_id}", response_model=PredictionResponse)
async def get_prediction(
    prediction_id: UUID,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
) -> PredictionResponse:
    result = await db.execute(
        select(Prediction).where(
            Prediction.id == prediction_id,
            Prediction.user_id == current_user.id,
        )
    )
    prediction = result.scalar_one_or_none()
    if prediction is None:
        raise HTTPException(status_code=404, detail="Предсказание не найдено")

    top_factors = [FactorItem(**factor) for factor in prediction.top_factors]
    return _to_prediction_response(prediction, top_factors)


@router.patch("/{prediction_id}/mode", response_model=PredictionResponse)
async def update_prediction_mode(
    prediction_id: UUID,
    payload: PredictionModeUpdateRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
) -> PredictionResponse:
    result = await db.execute(
        select(Prediction).where(
            Prediction.id == prediction_id,
            Prediction.user_id == current_user.id,
        )
    )
    prediction = result.scalar_one_or_none()
    if prediction is None:
        raise HTTPException(status_code=404, detail="Предсказание не найдено")

    if prediction.mode == payload.mode:
        top_factors = [FactorItem(**factor) for factor in prediction.top_factors]
        return _to_prediction_response(prediction, top_factors)

    try:
        body = _build_prediction_request_for_mode(prediction, payload.mode)
        predicted_class, probabilities, top_factors, clinical_derived = (
            _calculate_prediction_result(body)
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=_mode_switch_validation_detail(exc),
        ) from None
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None

    prediction.mode = payload.mode
    prediction.predicted_class = predicted_class
    prediction.probabilities = probabilities
    prediction.top_factors = [factor.model_dump() for factor in top_factors]
    user_input_data, _ = _split_input_data(prediction.input_data)
    prediction.input_data = {
        "user_input_data": user_input_data,
        "computed_data": clinical_derived,
    }
    await db.commit()
    await db.refresh(prediction)

    return _to_prediction_response(prediction, top_factors)


@router.get("/", response_model=list[PredictionListItemResponse])
async def list_predictions(
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
    skip: int = 0,
    limit: int = 20,
) -> list[PredictionListItemResponse]:
    result = await db.execute(
        select(Prediction)
        .where(Prediction.user_id == current_user.id)
        .order_by(Prediction.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    predictions = result.scalars().all()

    return [
        PredictionListItemResponse(
            id=prediction.id,
            name=prediction.name,
            mode=prediction.mode,
            predicted_class=prediction.predicted_class,
            class_label=CLASS_LABELS[prediction.predicted_class],
            probabilities=_probabilities_for_response(prediction.probabilities),
            top_factors=[FactorItem(**factor) for factor in prediction.top_factors],
            created_at=prediction.created_at.isoformat(),
        )
        for prediction in predictions
    ]


@router.post("/{prediction_id}/feedback")
async def submit_feedback(
    prediction_id: UUID,
    payload: PredictionFeedbackRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
) -> dict[str, str]:
    result = await db.execute(
        select(Prediction).where(
            Prediction.id == prediction_id,
            Prediction.user_id == current_user.id,
        )
    )
    prediction = result.scalar_one_or_none()
    if prediction is None:
        raise HTTPException(status_code=404, detail="Предсказание не найдено")

    prediction.real_class = payload.real_class
    prediction.feedback_comment = payload.comment
    prediction.feedback_at = datetime.utcnow()
    should_retrain = await record_feedback_for_retrain(db)
    await db.commit()
    if should_retrain:
        schedule_retrain_subprocess()

    return {"status": "ok"}
