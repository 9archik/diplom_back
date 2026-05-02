"""Пересборка `ensemble_optuna.pkl` для режима risk_group (как в build_nhanes_ml.ipynb)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
from catboost import CatBoostClassifier
from imblearn.combine import SMOTEENN
from lightgbm import LGBMClassifier
from optuna.samplers import TPESampler
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from ml.pickle_compat import register_pickle_compat

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
ENSEMBLE3_OPTUNA_TRIALS = 50
ENSEMBLE3_OPTUNA_VAL_SIZE = 0.2
ENSEMBLE3_UNHEALTHY_BIAS = 0.035
MIN_SAMPLES_FOR_OPTUNA = 500
LABELS_3 = (0, 1, 2)

META_ORDER = ("cb_full", "xgb_smoteenn", "lgb_smoteenn")


def _make_xgb_smoteenn() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="mlogloss",
        tree_method="hist",
    )


def _make_lgb_smoteenn() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=300,
        max_depth=-1,
        learning_rate=0.1,
        objective="multiclass",
        num_class=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def _fit_catboost_rfe_early_stopping(
    X: np.ndarray,
    y: np.ndarray,
    *,
    random_state: int,
    val_size: float = 0.15,
) -> CatBoostClassifier:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()
    if X.shape[0] < 20:
        raise ValueError("Слишком мало объектов для внутреннего train/val CatBoost")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X,
        y,
        test_size=val_size,
        stratify=y,
        random_state=random_state,
    )
    model = CatBoostClassifier(
        iterations=400,
        depth=6,
        learning_rate=0.1,
        loss_function="MultiClass",
        random_seed=random_state,
        verbose=False,
        thread_count=-1,
        auto_class_weights="Balanced",
        eval_metric="MultiClass",
        early_stopping_rounds=50,
    )
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
    return model


def _try_stratified_train_val_indices(
    y: np.ndarray,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    idx = np.arange(len(y))
    try:
        idx_tr, idx_va = train_test_split(
            idx,
            test_size=test_size,
            stratify=y,
            random_state=random_state,
        )
    except ValueError:
        return None
    if len(idx_tr) < 50 or len(idx_va) < 10:
        return None
    return idx_tr, idx_va


def _eval_ensemble_vote_tilt_mul(
    vote: np.ndarray,
    tilt: np.ndarray,
    p_cb: np.ndarray,
    p_xgb: np.ndarray,
    p_lgb: np.ndarray,
    y_va: np.ndarray,
) -> tuple[float, float, float, float]:
    vote = np.asarray(vote, dtype=float)
    vote = vote / np.clip(vote.sum(), 1e-12, None)
    base = vote[0] * p_cb + vote[1] * p_xgb + vote[2] * p_lgb
    tilted = base * np.asarray(tilt, dtype=float)
    proba = tilted / np.clip(tilted.sum(axis=1, keepdims=True), 1e-12, None)
    pred = np.argmax(proba, axis=1).astype(int)
    bacc = float(balanced_accuracy_score(y_va, pred))
    r1 = float(
        recall_score(y_va, pred, labels=[1], average="macro", zero_division=0)
    )
    r2 = float(
        recall_score(y_va, pred, labels=[2], average="macro", zero_division=0)
    )
    f1m = float(
        f1_score(y_va, pred, average="macro", labels=list(LABELS_3), zero_division=0)
    )
    return bacc, r1, r2, f1m


def _load_previous_bundle(ensemble_pkl: Path) -> dict[str, Any] | None:
    register_pickle_compat()
    if not ensemble_pkl.exists():
        return None
    raw = joblib.load(ensemble_pkl)
    return raw if isinstance(raw, dict) else None


def _extract_rfe_mask(bundle: dict[str, Any] | None, n_features: int) -> np.ndarray | None:
    if bundle is None:
        return None
    m = bundle.get("rfe_mask")
    if m is None:
        return None
    mask = np.asarray(m, dtype=bool).reshape(-1)
    if mask.shape[0] != n_features:
        logger.error(
            "Длина rfe_mask (%s) не совпадает с числом признаков (%s)",
            mask.shape[0],
            n_features,
        )
        return None
    return mask


def _fallback_vote_tilt(old: dict[str, Any] | None) -> tuple[np.ndarray, np.ndarray]:
    if old is None:
        return np.full(3, 1.0 / 3.0), np.ones(3, dtype=float)
    vw = old.get("vote_weights")
    ct = old.get("class_tilt")
    if vw is not None:
        v = np.asarray(vw, dtype=float).reshape(-1)
        if v.size == 3 and np.all(np.isfinite(v)) and float(v.sum()) > 0:
            vote = v / v.sum()
        else:
            vote = np.full(3, 1.0 / 3.0)
    else:
        vote = np.full(3, 1.0 / 3.0)
    if ct is not None:
        t = np.asarray(ct, dtype=float).reshape(-1)
        if t.size == 3 and np.all(np.isfinite(t)) and np.all(t > 0):
            tilt = t
        else:
            tilt = np.ones(3, dtype=float)
    else:
        tilt = np.ones(3, dtype=float)
    return vote, tilt


def _atomic_joblib_dump(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(obj, tmp)
    os.replace(tmp, path)


def fit_and_save_ensemble_bundle(
    *,
    x_scaled: np.ndarray,
    y_arr: np.ndarray,
    ensemble_pkl: Path,
) -> None:
    """Обучает ансамбль на уже отмасштабированных признаках и атомарно сохраняет pickle."""
    x_scaled = np.asarray(x_scaled, dtype=float)
    y_arr = np.asarray(y_arr).ravel()
    n_features = x_scaled.shape[1]

    prev = _load_previous_bundle(ensemble_pkl)
    mask = _extract_rfe_mask(prev, n_features)
    if mask is None:
        logger.error(
            "Пересборка ансамбля пропущена: нет корректной rfe_mask в %s",
            ensemble_pkl,
        )
        return

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    split = _try_stratified_train_val_indices(
        y_arr, ENSEMBLE3_OPTUNA_VAL_SIZE, RANDOM_STATE
    )
    use_optuna = (
        len(y_arr) >= MIN_SAMPLES_FOR_OPTUNA
        and split is not None
    )

    vote_best: np.ndarray
    tilt_best: np.ndarray

    if use_optuna and split is not None:
        idx_tr, idx_va = split
        X_tr = x_scaled[idx_tr]
        X_va = x_scaled[idx_va]
        y_tr = y_arr[idx_tr]
        y_va = y_arr[idx_va]

        X_rfe_tr = X_tr[:, mask]
        X_rfe_va = X_va[:, mask]

        cb_val = _fit_catboost_rfe_early_stopping(
            X_rfe_tr, y_tr, random_state=RANDOM_STATE
        )
        p_cb_va = np.asarray(cb_val.predict_proba(X_rfe_va), dtype=float)

        try:
            X_res_tr, y_res_tr = SMOTEENN(random_state=RANDOM_STATE).fit_resample(
                X_tr, y_tr
            )
        except Exception as exc:
            logger.error("SMOTEENN (val-ветка) не удался: %s", exc)
            vote_best, tilt_best = _fallback_vote_tilt(prev)
        else:
            clf_xgb_val = _make_xgb_smoteenn()
            clf_xgb_val.fit(X_res_tr, y_res_tr)
            p_xgb_va = np.asarray(clf_xgb_val.predict_proba(X_va), dtype=float)

            clf_lgb_val = _make_lgb_smoteenn()
            clf_lgb_val.fit(X_res_tr, y_res_tr)
            p_lgb_va = np.asarray(clf_lgb_val.predict_proba(X_va), dtype=float)

            def objective(trial: optuna.Trial) -> float:
                w_cb = trial.suggest_float("w_cb", 0.15, 1.0)
                w_xgb = trial.suggest_float("w_xgb", 0.15, 1.0)
                w_lgb = trial.suggest_float("w_lgb", 0.15, 1.0)
                vote = np.array([w_cb, w_xgb, w_lgb], dtype=float)

                t1 = trial.suggest_float("tilt_1", 1.0, 3.0)
                t2 = trial.suggest_float("tilt_2", 1.0, 3.5)
                tilt = np.array([1.0, t1, t2], dtype=float)

                bacc, r1, r2, _ = _eval_ensemble_vote_tilt_mul(
                    vote, tilt, p_cb_va, p_xgb_va, p_lgb_va, y_va
                )
                score = bacc + ENSEMBLE3_UNHEALTHY_BIAS * (0.5 * r1 + 0.5 * r2)
                trial.set_user_attr("bacc", bacc)
                trial.set_user_attr("recall_1", r1)
                trial.set_user_attr("recall_2", r2)
                return float(score)

            study = optuna.create_study(
                direction="maximize",
                sampler=TPESampler(seed=RANDOM_STATE),
                study_name="ensemble3_rfecv_xgb_lgb",
            )
            study.optimize(objective, n_trials=ENSEMBLE3_OPTUNA_TRIALS, show_progress_bar=False)

            bp = study.best_params
            vote_best = np.array([bp["w_cb"], bp["w_xgb"], bp["w_lgb"]], dtype=float)
            vote_best = vote_best / np.clip(vote_best.sum(), 1e-12, None)
            tilt_best = np.array([1.0, bp["tilt_1"], bp["tilt_2"]], dtype=float)

            logger.info(
                "Optuna ансамбля: best_score=%.4f, bacc=%.4f, vote=%s, tilt=%s",
                float(study.best_value),
                float(study.best_trial.user_attrs.get("bacc", float("nan"))),
                vote_best.tolist(),
                tilt_best.tolist(),
            )
    else:
        vote_best, tilt_best = _fallback_vote_tilt(prev)
        logger.warning(
            "Optuna пропущена (N=%s или split); веса/tilt из прежнего bundle или по умолчанию",
            len(y_arr),
        )

    try:
        X_res_full, y_res_full = SMOTEENN(random_state=RANDOM_STATE).fit_resample(
            x_scaled, y_arr
        )
    except Exception as exc:
        logger.error("Финальный SMOTEENN не удался: %s", exc)
        return

    clf_xgb_enn = _make_xgb_smoteenn()
    clf_xgb_enn.fit(X_res_full, y_res_full)

    clf_lgb_enn = _make_lgb_smoteenn()
    clf_lgb_enn.fit(X_res_full, y_res_full)

    cb_full = _fit_catboost_rfe_early_stopping(
        x_scaled[:, mask], y_arr, random_state=RANDOM_STATE
    )

    bundle: dict[str, Any] = {
        "cb_full": cb_full,
        "xgb_smoteenn": clf_xgb_enn,
        "lgb_smoteenn": clf_lgb_enn,
        "vote_weights": np.asarray(vote_best, dtype=float),
        "class_tilt": np.asarray(tilt_best, dtype=float),
        "rfe_mask": mask,
        "meta": {
            "name": "Ensemble_Optuna_vote_tilt_refitCB",
            "order": list(META_ORDER),
        },
    }

    _atomic_joblib_dump(ensemble_pkl, bundle)
    logger.info("Сохранён ансамбль: %s", ensemble_pkl)
