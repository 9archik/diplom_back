from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.loader import get_scaler

FEATURE_COLS = [
    "RIDAGEYR",
    "RIAGENDR",
    "BMXBMI",
    "BMXHT",
    "BMXWAIST",
    "hypertension",
    "LBXTC",
    "LBDHDD",
    "LBXGLU",
    "LBXTR",
    "DMDEDUC2",
    "LBXSAT",
    "LBXSASS",
    "LBXSUA",
    "LBXVIDMS",
    "eGFR",
    "LBXSBU",
    "URXUMA",
    "LBXHGB",
    "LBXRBCSI",
    "LBXSTB",
    "LBXSAL",
    "LBXSAPSI",
    "LBXWBCSI",
    "LBXLYPCT",
    "LBXMCVSI",
    "LBXRDW",
    "LBXCRP",
    "MCQ300C",
    "MCQ160A",
    "MCQ160E",
    "MCQ160F",
    "HSD010",
    "WHtR",
    "Age_x_WC",
    "Age_x_Glucose",
    "TyG_Index",
    "TyG_WC",
    "METS_IR",
    "ALT_AST_Ratio",
    "Log_CRP",
    "Log_SUA",
    "PhenoAge_accel",
]

DERIVED_MODEL_FEATURES: frozenset[str] = frozenset(
    {
        "BMXBMI",
        "eGFR",
        "WHtR",
        "Age_x_WC",
        "Age_x_Glucose",
        "TyG_Index",
        "TyG_WC",
        "METS_IR",
        "ALT_AST_Ratio",
        "Log_CRP",
        "Log_SUA",
        "PhenoAge_accel",
    }
)

ATOMIC_FEATURE_COLS: tuple[str, ...] = tuple(
    col for col in FEATURE_COLS if col not in DERIVED_MODEL_FEATURES
)

CLINICAL_DERIVED_RESPONSE_KEYS: frozenset[str] = frozenset(
    {
        "BMXBMI",
        "eGFR",
        "WHtR",
        "TyG_Index",
        "ALT_AST_Ratio",
        "Log_CRP",
        "Log_SUA",
        "PhenoAge_accel",
    }
)

MEDIANS = {
    "RIDAGEYR": 46.0,
    "RIAGENDR": 1.0,
    "BMXBMI": 27.0,
    "BMXHT": 166.5,
    "BMXWAIST": 94.9,
    "hypertension": 0.0,
    "LBXTC": 194.0,
    "LBDHDD": 50.0,
    "LBXGLU": 98.6,
    "LBXTR": 108.0,
    "DMDEDUC2": 2.0,
    "LBXSAT": 17.0,
    "LBXSASS": 21.0,
    "LBXSUA": 5.3,
    "LBXVIDMS": 60.4,
    "eGFR": 84.13,
    "LBXSBU": 13.0,
    "URXUMA": 2.197,
    "LBXHGB": 14.1,
    "LBXRBCSI": 4.66,
    "LBXSTB": 0.6,
    "LBXSAL": 4.2,
    "LBXSAPSI": 74.0,
    "LBXWBCSI": 6.7,
    "LBXLYPCT": 31.2,
    "LBXMCVSI": 89.8,
    "LBXRDW": 13.0,
    "LBXCRP": 0.21,
    "MCQ300C": 0.0,
    "MCQ160A": 0.0,
    "MCQ160E": 0.0,
    "MCQ160F": 0.0,
    "HSD010": 3.0,
    "WHtR": 0.568,
    "Age_x_WC": 4431.0,
    "Age_x_Glucose": 4764.7,
    "TyG_Index": 8.607,
    "TyG_WC": 824.306,
    "METS_IR": 140.756,
    "ALT_AST_Ratio": 0.826,
    "Log_CRP": 0.191,
    "Log_SUA": 1.841,
    "PhenoAge_accel": -0.346,
}

PHENOAGE_GAMMA = 0.0076927
PHENOAGE_LIN_INTERCEPT = -19.907
PHENOAGE_PHENO_INTERCEPT = 141.50225
PHENOAGE_PHENO_SLOPE = 0.090165
PHENOAGE_MORTALITY_INNER = -0.00553
CRITICAL_LAB_FEATURES = frozenset({"LBXGLU"})
_ML_DIR = Path(__file__).resolve().parent
GROUP_MEDIANS_PATH = _ML_DIR / "models" / "group_medians.json"
GLOBAL_MEDIANS_PATH = _ML_DIR / "models" / "global_medians.json"
LBXCRP_GROUP_STATS_PATH = _ML_DIR / "models" / "lbxcrp_group_stats.json"


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as file:
        loaded = json.load(file)
    if isinstance(loaded, dict):
        return loaded
    return {}


GROUP_MEDIANS = _load_json_if_exists(GROUP_MEDIANS_PATH)
GLOBAL_MEDIANS_RAW = _load_json_if_exists(GLOBAL_MEDIANS_PATH)
LBXCRP_GROUP_STATS = _load_json_if_exists(LBXCRP_GROUP_STATS_PATH)

GLOBAL_MEDIANS: dict[str, float] = {}
for feature_name, value in GLOBAL_MEDIANS_RAW.items():
    try:
        GLOBAL_MEDIANS[feature_name] = float(value)
    except (TypeError, ValueError):
        continue


def _normalize_lbxcrp_quartiles(data: dict[str, Any]) -> tuple[float, float, float] | None:
    quartiles = data.get("bmi_quartiles")
    if not isinstance(quartiles, list) or len(quartiles) != 3:
        return None
    q1, q2, q3 = quartiles
    if q1 is None or q2 is None or q3 is None:
        return None
    try:
        return float(q1), float(q2), float(q3)
    except (TypeError, ValueError):
        return None


def _normalize_lbxcrp_group_medians(data: dict[str, Any]) -> dict[str, float]:
    group_medians = data.get("group_medians")
    if not isinstance(group_medians, dict):
        return {}
    normalized: dict[str, float] = {}
    for group_key, value in group_medians.items():
        if not isinstance(group_key, str):
            continue
        try:
            normalized[group_key] = float(value)
        except (TypeError, ValueError):
            continue
    return normalized


LBXCRP_BMI_QUARTILES = _normalize_lbxcrp_quartiles(LBXCRP_GROUP_STATS)
LBXCRP_GROUP_MEDIANS = _normalize_lbxcrp_group_medians(LBXCRP_GROUP_STATS)
try:
    LBXCRP_GLOBAL_MEDIAN = float(LBXCRP_GROUP_STATS.get("global_median"))
except (TypeError, ValueError):
    LBXCRP_GLOBAL_MEDIAN = None


def calc_bmi(weight_kg: float, height_cm: float) -> float:
    return round(weight_kg / (height_cm / 100.0) ** 2, 2)


def calc_egfr(creatinine: float, age: int, gender: int) -> float:
    kappa = 0.7 if gender == 1 else 0.9
    alpha = -0.241 if gender == 1 else -0.302
    scr_k = creatinine / kappa
    egfr = (
        142
        * min(scr_k, 1.0) ** alpha
        * max(scr_k, 1.0) ** (-1.200)
        * 0.9938**age
    )
    if gender == 1:
        egfr *= 1.012
    return round(egfr, 2)


def calc_phenoage_accel(
    age: float,
    albumin_gdl: float,
    creatinine_mgdl: float,
    glucose_mgdl: float,
    crp_mgdl: float,
    lymph_pct: float,
    mcv_fl: float,
    rdw_pct: float,
    alp_ul: float,
    wbc: float,
) -> float:
    albumin_gl = albumin_gdl * 10.0
    creat_umol = creatinine_mgdl * 88.42
    glucose_mmol = glucose_mgdl / 18.0182
    crp_mgl = crp_mgdl * 10.0
    log_crp = np.log(max(crp_mgl, 1e-6))

    xb = (
        PHENOAGE_LIN_INTERCEPT
        + 0.0804 * age
        - 0.0336 * albumin_gl
        + 0.0095 * creat_umol
        + 0.1953 * glucose_mmol
        + 0.0954 * log_crp
        - 0.0120 * lymph_pct
        + 0.0268 * mcv_fl
        + 0.3306 * rdw_pct
        + 0.00188 * alp_ul
        + 0.0554 * wbc
    )

    b = (np.exp(120.0 * PHENOAGE_GAMMA) - 1.0) / PHENOAGE_GAMMA
    mortality = 1.0 - np.exp(-np.exp(xb) * b)
    mortality = np.clip(mortality, 1e-12, 1.0 - 1e-12)

    inner = PHENOAGE_MORTALITY_INNER * np.log(1.0 - mortality)
    inner = max(inner, 1e-300)
    pheno_age = PHENOAGE_PHENO_INTERCEPT + np.log(inner) / PHENOAGE_PHENO_SLOPE
    return round(float(pheno_age - age), 4)


def add_engineered_features(df: pd.DataFrame, lbxscr: float | None) -> pd.DataFrame:
    out = df.copy()

    out["WHtR"] = out["BMXWAIST"] / max(float(out["BMXHT"].iloc[0]), 1e-8)
    out["Age_x_WC"] = out["RIDAGEYR"] * out["BMXWAIST"]

    out["Age_x_Glucose"] = (
        out["RIDAGEYR"] * out["LBXGLU"] if out["LBXGLU"].notna().all() else np.nan
    )

    if out["LBXTR"].notna().all() and out["LBXGLU"].notna().all():
        out["TyG_Index"] = np.log(np.maximum((out["LBXTR"] * out["LBXGLU"]) / 2.0, 1e-8))
    else:
        out["TyG_Index"] = np.nan

    out["TyG_WC"] = (
        out["TyG_Index"] * out["BMXWAIST"] if out["TyG_Index"].notna().all() else np.nan
    )

    if all(out[c].notna().all() for c in ["LBXGLU", "LBXTR", "BMXWAIST", "LBDHDD"]):
        out["METS_IR"] = (
            np.log(2 * out["LBXGLU"] + out["LBXTR"])
            * out["BMXWAIST"]
            / np.log(out["LBDHDD"].clip(lower=1e-8))
        )
    else:
        out["METS_IR"] = np.nan

    out["ALT_AST_Ratio"] = (
        out["LBXSAT"] / (out["LBXSASS"] + 0.1)
        if out["LBXSAT"].notna().all() and out["LBXSASS"].notna().all()
        else np.nan
    )

    out["Log_CRP"] = np.log1p(out["LBXCRP"]) if out["LBXCRP"].notna().all() else np.nan
    out["Log_SUA"] = np.log1p(out["LBXSUA"]) if out["LBXSUA"].notna().all() else np.nan

    phenoage_cols = [
        "LBXSAL",
        "LBXGLU",
        "LBXCRP",
        "LBXLYPCT",
        "LBXMCVSI",
        "LBXRDW",
        "LBXSAPSI",
        "LBXWBCSI",
    ]
    has_all = lbxscr is not None and all(out[c].notna().all() for c in phenoage_cols)
    if has_all:
        row = out.iloc[0]
        out["PhenoAge_accel"] = calc_phenoage_accel(
            age=float(row["RIDAGEYR"]),
            albumin_gdl=float(row["LBXSAL"]),
            creatinine_mgdl=float(lbxscr),
            glucose_mgdl=float(row["LBXGLU"]),
            crp_mgdl=float(row["LBXCRP"]),
            lymph_pct=float(row["LBXLYPCT"]),
            mcv_fl=float(row["LBXMCVSI"]),
            rdw_pct=float(row["LBXRDW"]),
            alp_ul=float(row["LBXSAPSI"]),
            wbc=float(row["LBXWBCSI"]),
        )
    else:
        out["PhenoAge_accel"] = np.nan

    return out


def _phenoage_accel_vectorized_numpy(df: pd.DataFrame, lbxscr: pd.Series) -> np.ndarray:
    """Тот же расчёт, что calc_phenoage_accel, для всех строк; NaN при неполных данных."""

    def col(name: str) -> np.ndarray:
        return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)

    n = len(df)
    age = col("RIDAGEYR")
    albumin_gdl = col("LBXSAL")
    creat = pd.to_numeric(lbxscr, errors="coerce").to_numpy(dtype=float)
    glucose_mgdl = col("LBXGLU")
    crp_mgdl = col("LBXCRP")
    lymph_pct = col("LBXLYPCT")
    mcv_fl = col("LBXMCVSI")
    rdw_pct = col("LBXRDW")
    alp_ul = col("LBXSAPSI")
    wbc = col("LBXWBCSI")

    valid = (
        np.isfinite(age)
        & np.isfinite(albumin_gdl)
        & np.isfinite(creat)
        & (creat > 0)
        & np.isfinite(glucose_mgdl)
        & np.isfinite(crp_mgdl)
        & np.isfinite(lymph_pct)
        & np.isfinite(mcv_fl)
        & np.isfinite(rdw_pct)
        & np.isfinite(alp_ul)
        & np.isfinite(wbc)
    )

    albumin_gl = albumin_gdl * 10.0
    creat_umol = creat * 88.42
    glucose_mmol = glucose_mgdl / 18.0182
    crp_mgl = crp_mgdl * 10.0
    log_crp = np.log(np.maximum(crp_mgl, 1e-6))

    xb = (
        PHENOAGE_LIN_INTERCEPT
        + 0.0804 * age
        - 0.0336 * albumin_gl
        + 0.0095 * creat_umol
        + 0.1953 * glucose_mmol
        + 0.0954 * log_crp
        - 0.0120 * lymph_pct
        + 0.0268 * mcv_fl
        + 0.3306 * rdw_pct
        + 0.00188 * alp_ul
        + 0.0554 * wbc
    )

    b = (np.exp(120.0 * PHENOAGE_GAMMA) - 1.0) / PHENOAGE_GAMMA
    mortality = 1.0 - np.exp(-np.exp(xb) * b)
    mortality = np.clip(mortality, 1e-12, 1.0 - 1e-12)
    inner = PHENOAGE_MORTALITY_INNER * np.log(1.0 - mortality)
    inner = np.maximum(inner, 1e-300)
    pheno_age = PHENOAGE_PHENO_INTERCEPT + np.log(inner) / PHENOAGE_PHENO_SLOPE
    accel = np.round(pheno_age - age, 4)
    out = np.full(n, np.nan, dtype=float)
    out[valid] = accel[valid]
    return out


def add_engineered_features_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Инженерные признаки для таблицы (NHANES CSV и т.п.), по строкам с NaN там, где не хватает входов."""
    out = df.copy()
    ht = np.maximum(pd.to_numeric(out["BMXHT"], errors="coerce").to_numpy(dtype=float), 1e-8)
    waist = pd.to_numeric(out["BMXWAIST"], errors="coerce").to_numpy(dtype=float)
    ridage = pd.to_numeric(out["RIDAGEYR"], errors="coerce").to_numpy(dtype=float)
    out["WHtR"] = waist / ht
    out["Age_x_WC"] = ridage * waist

    lbxglu = pd.to_numeric(out["LBXGLU"], errors="coerce").to_numpy(dtype=float)
    out["Age_x_Glucose"] = ridage * lbxglu

    lbxtr = pd.to_numeric(out["LBXTR"], errors="coerce").to_numpy(dtype=float)
    tyg = np.log(np.maximum((lbxtr * lbxglu) / 2.0, 1e-8))
    out["TyG_Index"] = tyg
    out["TyG_WC"] = tyg * waist

    lbdhdd = pd.to_numeric(out["LBDHDD"], errors="coerce").to_numpy(dtype=float)
    out["METS_IR"] = (
        np.log(np.maximum(2.0 * lbxglu + lbxtr, 1e-8))
        * waist
        / np.log(np.maximum(lbdhdd, 1e-8))
    )

    sat = pd.to_numeric(out["LBXSAT"], errors="coerce").to_numpy(dtype=float)
    sass = pd.to_numeric(out["LBXSASS"], errors="coerce").to_numpy(dtype=float)
    out["ALT_AST_Ratio"] = sat / (sass + 0.1)

    lbxcrp = pd.to_numeric(out["LBXCRP"], errors="coerce").to_numpy(dtype=float)
    lbxsua = pd.to_numeric(out["LBXSUA"], errors="coerce").to_numpy(dtype=float)
    out["Log_CRP"] = np.log1p(np.clip(lbxcrp, 0.0, None))
    out["Log_SUA"] = np.log1p(np.clip(lbxsua, 0.0, None))

    lbxscr = (
        pd.to_numeric(out["LBXSCR"], errors="coerce")
        if "LBXSCR" in out.columns
        else pd.Series(np.nan, index=out.index)
    )
    out["PhenoAge_accel"] = _phenoage_accel_vectorized_numpy(out, lbxscr)
    return out


def impute_training_features_median(df: pd.DataFrame) -> pd.DataFrame:
    """Для обучения: отбрасываем строки без LBXGLU, остальные пропуски — медиана по столбцу."""
    out = df[list(FEATURE_COLS)].copy()
    out = out[out["LBXGLU"].notna()].copy()
    for col in FEATURE_COLS:
        med = out[col].median()
        if pd.isna(med):
            med = float(MEDIANS.get(col, 0.0))
        out[col] = out[col].fillna(med)
    return out


def _make_group_key(age: float, sex: float) -> str:
    return f"{int(age)}_{int(sex)}"


def _make_lbxcrp_group_key(age: float, sex: float, bmi_quartile: int) -> str:
    return f"{int(age)}_{int(sex)}_{int(bmi_quartile)}"


def _get_grouped_median(col: str, age: float, sex: float) -> float | None:
    by_feature = GROUP_MEDIANS.get(col)
    if not isinstance(by_feature, dict):
        return None
    key = _make_group_key(age, sex)
    value = by_feature.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_bmi_quartile(bmi: float) -> int | None:
    if LBXCRP_BMI_QUARTILES is None:
        return None
    q1, q2, q3 = LBXCRP_BMI_QUARTILES
    if bmi <= q1:
        return 0
    if bmi <= q2:
        return 1
    if bmi <= q3:
        return 2
    return 3


def _get_lbxcrp_grouped_median(age: float, sex: float, bmi: float) -> float | None:
    bmi_quartile = _get_bmi_quartile(bmi)
    if bmi_quartile is None:
        return None
    key = _make_lbxcrp_group_key(age, sex, bmi_quartile)
    value = LBXCRP_GROUP_MEDIANS.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_global_median(col: str) -> float:
    if col == "LBXCRP" and LBXCRP_GLOBAL_MEDIAN is not None:
        return LBXCRP_GLOBAL_MEDIAN
    if col in GLOBAL_MEDIANS:
        return GLOBAL_MEDIANS[col]
    return MEDIANS.get(col, 0.0)


def _apply_grouped_imputation(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    row = out.iloc[0]
    age = float(row["RIDAGEYR"])
    sex = float(row["RIAGENDR"])
    bmi = float(row["BMXBMI"])

    for col in FEATURE_COLS:
        current_value = row[col]
        if pd.notna(current_value):
            continue

        if col in CRITICAL_LAB_FEATURES:
            raise ValueError(f"{col} is required and cannot be imputed")

        if col == "LBXCRP":
            grouped_value = _get_lbxcrp_grouped_median(age=age, sex=sex, bmi=bmi)
        else:
            grouped_value = _get_grouped_median(col=col, age=age, sex=sex)

        fallback_value = _get_global_median(col)
        imputed_value = fallback_value if grouped_value is None else grouped_value
        out.at[out.index[0], col] = imputed_value

    return out


def build_imputed_feature_frame(data: Any) -> pd.DataFrame:
    bmxbmi = calc_bmi(data.weight_kg, data.BMXHT)

    lbxscr_for_calc: float | None = None
    if data.LBXSCR is not None and data.LBXSCR > 0:
        lbxscr_for_calc = data.LBXSCR

    egfr = None
    if lbxscr_for_calc is not None:
        egfr = calc_egfr(lbxscr_for_calc, data.RIDAGEYR, data.RIAGENDR)

    raw = data.model_dump(exclude={"mode", "weight_kg", "LBXSCR"})
    raw["BMXBMI"] = bmxbmi
    raw["eGFR"] = egfr

    df = pd.DataFrame([raw])
    df = add_engineered_features(df, lbxscr=lbxscr_for_calc)

    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[FEATURE_COLS]
    df = _apply_grouped_imputation(df)

    for col in FEATURE_COLS:
        if df[col].isna().any():
            df[col] = df[col].fillna(_get_global_median(col))

    return df


def extract_clinical_derived_for_response(df: pd.DataFrame) -> dict[str, float]:
    row = df.iloc[0]
    out: dict[str, float] = {}
    for key in sorted(CLINICAL_DERIVED_RESPONSE_KEYS):
        value = row.get(key)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        try:
            out[key] = round(float(value), 6)
        except (TypeError, ValueError):
            continue
    return out


def prepare_features(data: Any) -> np.ndarray:
    df = build_imputed_feature_frame(data)
    scaler = get_scaler()
    return scaler.transform(df)
