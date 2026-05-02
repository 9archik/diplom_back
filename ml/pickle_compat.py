from __future__ import annotations

import sys
from typing import Any

import numpy as np


class EarlyStoppingBoostingClassifier:
    """Compatibility shim for unpickling notebook-trained estimators.

    Historical artifacts were serialized from notebook scope where this class
    lived in `__main__`. During API runtime `__main__` points to uvicorn, so
    joblib cannot resolve that symbol without explicit registration.
    """

    def __init__(
        self,
        model_key: str | None = None,
        random_state: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.model_key = model_key
        self.random_state = random_state
        for key, value in kwargs.items():
            setattr(self, key, value)

    def fit(self, X: Any, y: Any, **fit_params: Any) -> "EarlyStoppingBoostingClassifier":
        estimator = getattr(self, "estimator_", None)
        if estimator is None:
            raise RuntimeError("Estimator is not initialized")
        estimator.fit(X, y, **fit_params)
        return self

    def predict(self, X: Any) -> np.ndarray:
        estimator = getattr(self, "estimator_", None)
        if estimator is None:
            raise RuntimeError("Estimator is not initialized")
        return np.asarray(estimator.predict(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        estimator = getattr(self, "estimator_", None)
        if estimator is None:
            raise RuntimeError("Estimator is not initialized")
        if not hasattr(estimator, "predict_proba"):
            raise RuntimeError("Underlying estimator does not support predict_proba")
        return np.asarray(estimator.predict_proba(X))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params = {"model_key": self.model_key, "random_state": self.random_state}
        if deep and hasattr(self, "estimator_"):
            params["estimator_"] = getattr(self, "estimator_")
        return params

    def set_params(self, **params: Any) -> "EarlyStoppingBoostingClassifier":
        for key, value in params.items():
            setattr(self, key, value)
        return self


def register_pickle_compat() -> None:
    main_module = sys.modules.get("__main__")
    if main_module is None:
        return
    setattr(main_module, "EarlyStoppingBoostingClassifier", EarlyStoppingBoostingClassifier)
