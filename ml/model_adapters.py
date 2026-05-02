from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _shap_row_for_predicted_class(shap_values: object, predicted_class: int) -> np.ndarray:
    if isinstance(shap_values, list):
        return np.array(shap_values[predicted_class])[0]

    shap_array = np.array(shap_values)
    if shap_array.ndim == 3:
        return shap_array[0, :, predicted_class]
    if shap_array.ndim == 2:
        return shap_array[0]
    raise ValueError("Unsupported SHAP output shape")


class RiskGroupEnsembleModel:
    def __init__(self, bundle: dict[str, Any]) -> None:
        self._estimators = self._extract_estimators(bundle)
        if not self._estimators:
            raise ValueError("Risk-group bundle does not contain estimators")

        self._vote_weights = self._normalize_vote_weights(bundle.get("vote_weights"))
        self._class_tilt = self._normalize_class_tilt(bundle.get("class_tilt"))
        self._rfe_mask = self._normalize_mask(bundle.get("rfe_mask"))

    def predict_proba(self, X: Any) -> np.ndarray:
        full_features = np.asarray(X, dtype=float)
        masked_features = self._apply_mask(full_features)

        stacked = np.stack(
            [
                np.asarray(
                    estimator.predict_proba(
                        self._features_for_estimator(
                            estimator=estimator,
                            full_features=full_features,
                            masked_features=masked_features,
                        )
                    ),
                    dtype=float,
                )
                for estimator in self._estimators
            ],
            axis=0,
        )
        combined = np.tensordot(self._vote_weights, stacked, axes=(0, 0))

        if self._class_tilt is not None and self._class_tilt.shape[0] == combined.shape[1]:
            combined = combined * self._class_tilt

        combined = np.clip(combined, 1e-9, None)
        return combined / combined.sum(axis=1, keepdims=True)

    def predict(self, X: Any) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

    def approx_shap_per_full_feature(self, X: Any, predicted_class: int) -> np.ndarray | None:
        """Вклад признаков для топ-факторов: взвешенное среднее SHAP базовых деревьев.

        `TreeExplainer` не применим к обёртке ансамбля, поэтому считаем SHAP по каждому
        базовому оценщику и собираем вектор длины полного набора признаков (как у X).
        """
        try:
            import shap
        except ImportError:
            return None

        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        n_full = X_arr.shape[1]
        masked = self._apply_mask(X_arr)

        accum = np.zeros(n_full, dtype=float)
        used_weight = 0.0

        for weight, estimator in zip(self._vote_weights, self._estimators):
            if not np.isfinite(weight) or float(weight) <= 0:
                continue
            try:
                X_sub = self._features_for_estimator(
                    estimator, X_arr, masked
                )
                explainer = shap.TreeExplainer(estimator)
                raw_sv = explainer.shap_values(X_sub)
                row = _shap_row_for_predicted_class(raw_sv, predicted_class)
                row = np.asarray(row, dtype=float).reshape(-1)
                if row.shape[0] != X_sub.shape[1]:
                    continue

                expanded = np.zeros(n_full, dtype=float)
                if (
                    self._rfe_mask is not None
                    and row.shape[0] == int(np.sum(self._rfe_mask))
                ):
                    expanded[self._rfe_mask] = row
                elif row.shape[0] == n_full:
                    expanded = row
                else:
                    continue

                accum += float(weight) * expanded
                used_weight += float(weight)
            except Exception:
                continue

        if used_weight <= 0:
            return None
        return accum / used_weight

    def _extract_estimators(self, bundle: dict[str, Any]) -> list[Any]:
        estimators: list[Any] = []
        order_raw = bundle.get("meta", {}).get("order")
        order: Sequence[str]
        if isinstance(order_raw, list) and all(isinstance(key, str) for key in order_raw):
            order = order_raw
        else:
            order = ("cb_full", "xgb_smoteenn", "lgb_smoteenn")

        for key in order:
            value = bundle.get(key)
            if value is None:
                continue
            if hasattr(value, "predict") and hasattr(value, "predict_proba"):
                estimators.append(value)
        return estimators

    def _normalize_vote_weights(self, weights_raw: Any) -> np.ndarray:
        size = len(self._estimators)
        default = np.full(size, 1.0 / size)
        if weights_raw is None:
            return default

        weights = np.asarray(weights_raw, dtype=float).reshape(-1)
        if weights.size != size:
            return default
        if np.any(~np.isfinite(weights)) or float(weights.sum()) <= 0:
            return default
        return weights / weights.sum()

    def _normalize_class_tilt(self, tilt_raw: Any) -> np.ndarray | None:
        if tilt_raw is None:
            return None
        tilt = np.asarray(tilt_raw, dtype=float).reshape(-1)
        if tilt.size == 0 or np.any(~np.isfinite(tilt)):
            return None
        return tilt

    def _normalize_mask(self, mask_raw: Any) -> np.ndarray | None:
        if mask_raw is None:
            return None
        mask = np.asarray(mask_raw).reshape(-1)
        if mask.size == 0:
            return None
        return mask.astype(bool, copy=False)

    def _apply_mask(self, X: np.ndarray) -> np.ndarray:
        if self._rfe_mask is None:
            return X
        if X.ndim != 2:
            return X
        if X.shape[1] != self._rfe_mask.shape[0]:
            return X
        return X[:, self._rfe_mask]

    def _features_for_estimator(
        self,
        estimator: Any,
        full_features: np.ndarray,
        masked_features: np.ndarray,
    ) -> np.ndarray:
        expected_feature_count = getattr(estimator, "n_features_in_", None)
        if expected_feature_count == full_features.shape[1]:
            return full_features
        if expected_feature_count == masked_features.shape[1]:
            return masked_features
        return full_features


def adapt_loaded_model(mode: str, loaded_object: Any) -> Any:
    if mode == "risk_group" and isinstance(loaded_object, dict):
        return RiskGroupEnsembleModel(loaded_object)
    return loaded_object
