"""
Сборка единого NHANES-датасета для 3-классовой классификации (норма / преддиабет / диабет).

Целевая переменная задаётся только по лабораторному HbA1c (пороги ADA: <5.7% / 5.7–6.4% / ≥6.5%);
самоотчёт о диагнозе и прочие поля в разметку target не входят.

Inner join по SEQN по модулям clean CSV.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# --- Пути по умолчанию: входные *_clean.csv в папке nhames_datasets рядом со скриптом ---
BASE_DIR = Path(__file__).resolve().parent
NHANES_DATA_DIR = BASE_DIR / "nhames_datasets"
DEFAULT_FILES = {
    "demographics": NHANES_DATA_DIR / "demographics_clean.csv",
    "questionnaire": NHANES_DATA_DIR / "questionnaire_clean.csv",
    "response": NHANES_DATA_DIR / "response_clean.csv",
    "dietary": NHANES_DATA_DIR / "dietary_clean.csv",
    "weights": NHANES_DATA_DIR / "weights_clean.csv",
    "chemicals": NHANES_DATA_DIR / "chemicals_clean.csv",
}
OUTPUT_CSV = BASE_DIR / "nhanes_ml_ready.csv"
OUTPUT_META = BASE_DIR / "nhanes_ml_meta.json"

POSSIBLE_HBA1C_NAMES = ("LBXGH", "HbA1c", "LBXA1C", "hba1c", "GH")

# Анамнез ССЗ и самооценка здоровья (опросник; опциональный подмёрж).
CV_HISTORY_AND_SELF_HEALTH_COLS: tuple[str, ...] = (
    "MCQ160A",  # когда-либо инфаркт миокарда
    "MCQ160E",  # инсульт
    "MCQ160F",  # сердечная недостаточность
    "HSD010",  # самооценка здоровья 1 (отлично) … 5 (плохо)
)

# Осложнения/лечение из модуля NHANES DIQ (опросник; опциональный подмёрж).
DIQ_COMPLICATION_COLS: tuple[str, ...] = (
    "DIQ090",  # retinopathy
    "DIQ100",  # neuropathy / extreme fatigue
)


class MetaJson(TypedDict, total=False):
    target: dict[str, Any]
    feature_columns: list[str]
    class_distribution: dict[str, int]
    missing_counts_before: dict[str, int]
    label_encoders_mapping: dict[str, dict[str, int]]
    warnings: list[str]
    hba1c_column_resolved: str | None
    dropped_hba1c_columns: list[str]
    nominal_columns_post_split: dict[str, str]
    seqn_duplicates_count: int


def warn(msg: str, bucket: list[str]) -> None:
    bucket.append(msg)


def resolve_hba1c_column(columns: pd.Index) -> str | None:
    cols_set = {str(c) for c in columns}
    for name in POSSIBLE_HBA1C_NAMES:
        if name in cols_set:
            return name
    lower_map = {str(c).lower(): str(c) for c in columns}
    for name in POSSIBLE_HBA1C_NAMES:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    for c in columns:
        s = str(c).upper()
        if "A1C" in s or (s.endswith("GH") and "LBX" in s):
            return str(c)
    return None


def hba1c_feature_columns(columns: pd.Index) -> list[str]:
    """Все столбцы гликированного гемоглобина — убрать из X."""
    out: list[str] = []
    for c in columns:
        sc = str(c).upper()
        if sc == "LBXGH" or sc.startswith("LBXGH"):
            out.append(str(c))
        elif "A1C" in sc and "LBX" in sc:
            out.append(str(c))
    return list(dict.fromkeys(out))


def read_csv_columns(path: Path, cols: list[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    available = [c for c in cols if c in header.columns]
    missing = [c for c in cols if c not in header.columns]
    if missing:
        raise ValueError(f"{path}: нет колонок {', '.join(missing)}")
    return pd.read_csv(path, usecols=available)


def dedupe_seqn(df: pd.DataFrame, label: str, warnings_list: list[str]) -> pd.DataFrame:
    n = len(df)
    out = df.drop_duplicates(subset=["SEQN"], keep="first")
    if len(out) < n:
        warn(f"{label}: удалено {n - len(out)} дубликатов по SEQN (оставлена первая строка).", warnings_list)
    return out


def merge_optional_columns(
    df: pd.DataFrame,
    path: Path,
    cols: list[str],
    label: str,
    warnings_list: list[str],
) -> pd.DataFrame:
    """Подмёрживает к df колонки из CSV, если они есть в заголовке; иначе — предупреждение."""
    header = pd.read_csv(path, nrows=0)
    available = [c for c in cols if c in header.columns and c not in df.columns]
    missing = [c for c in cols if c not in header.columns]
    for m in missing:
        warn(f"{label}: колонка {m} отсутствует в CSV.", warnings_list)
    if not available:
        return df
    extra = dedupe_seqn(
        pd.read_csv(path, usecols=["SEQN", *available]),
        f"{label} optional",
        warnings_list,
    )
    return df.merge(extra, on="SEQN", how="left")


def merge_first_available_name(
    df: pd.DataFrame,
    path: Path,
    target_col: str,
    candidates: list[str],
    label: str,
    warnings_list: list[str],
) -> pd.DataFrame:
    """
    Берёт первое доступное имя колонки из candidates в CSV и мёржит в df как target_col.
    Разные циклы NHANES используют разные суффиксы (например LBXSAT vs LBXSATSI).
    """
    if target_col in df.columns:
        return df
    header = pd.read_csv(path, nrows=0)
    for cand in candidates:
        if cand not in header.columns:
            continue
        extra = dedupe_seqn(
            pd.read_csv(path, usecols=["SEQN", cand]),
            f"{label} {target_col}←{cand}",
            warnings_list,
        )
        extra = extra.rename(columns={cand: target_col})
        return df.merge(extra, on="SEQN", how="left")
    warn(
        f"{label}: ни одна из колонок {candidates} не найдена (целевое имя в датасете: {target_col}).",
        warnings_list,
    )
    return df


def load_merged_inner(paths: dict[str, Path], warnings_list: list[str]) -> pd.DataFrame:
    demo_cols = ["SEQN", "RIDAGEYR", "RIAGENDR", "DMDEDUC2"]
    q_cols = [
        "SEQN",
        "BPQ020",
        "BPQ030",
    ]
    resp_cols = [
        "SEQN",
        "LBXGH",
        "LBXGLU",
        "LBXTC",
        "LBDHDD",
        "LBXTR",
        "LBXCRP",
        "BMXBMI",
        "BMXWT",
        "BMXHT",
        "BMXWAIST",
        "BMXHIP",
        "BPXSY1",
        "BPXDI1",
    ]

    demo = dedupe_seqn(read_csv_columns(paths["demographics"], demo_cols), "demographics", warnings_list)
    demo = merge_optional_columns(
        demo,
        paths["demographics"],
        ["DBQ700"],
        "demographics",
        warnings_list,
    )
    q = dedupe_seqn(read_csv_columns(paths["questionnaire"], q_cols), "questionnaire", warnings_list)
    q = merge_optional_columns(
        q,
        paths["questionnaire"],
        list(
            dict.fromkeys(
                [
                    *CV_HISTORY_AND_SELF_HEALTH_COLS,
                    *DIQ_COMPLICATION_COLS,
                    "MCQ300C",
                    "DIQ010",
                ],
            ),
        ),
        "questionnaire",
        warnings_list,
    )
    resp = dedupe_seqn(read_csv_columns(paths["response"], resp_cols), "response", warnings_list)
    # Лабораторные: в разных циклах NHANES — разные имена (SI / суффиксы); канонические имена в датасете.
    resp = merge_first_available_name(
        resp,
        paths["response"],
        "LBXSAT",
        ["LBXSAT", "LBXSATSI", "LBXSATSI1", "LBXSATSI2"],
        "response",
        warnings_list,
    )
    resp = merge_first_available_name(
        resp,
        paths["response"],
        "LBXSASS",
        ["LBXSASS", "LBXSASSI", "LBXSASSI1", "LBXSASSI2"],
        "response",
        warnings_list,
    )
    resp = merge_first_available_name(
        resp,
        paths["response"],
        "LBXSUA",
        ["LBXSUA", "LBXSUA1", "LBXSUA2"],
        "response",
        warnings_list,
    )
    resp = merge_first_available_name(
        resp,
        paths["response"],
        "LBXHGB",
        ["LBXHGB", "LBXHGBSI", "LBXHGB1", "LBXHGB2"],
        "response",
        warnings_list,
    )
    resp = merge_first_available_name(
        resp,
        paths["response"],
        "LBXRBCSI",
        ["LBXRBCSI", "LBXRBCSI1", "LBXRBCSI2", "LBXRBC", "LBXRBC1", "LBXRBC2"],
        "response",
        warnings_list,
    )
    resp = merge_first_available_name(
        resp,
        paths["response"],
        "LBXSTB",
        ["LBXSTB", "LBXSTBSI", "LBXSTB1", "LBXSTB2"],
        "response",
        warnings_list,
    )
    resp = merge_first_available_name(
        resp,
        paths["response"],
        "LBDLDL",
        ["LBDLDL", "LBDLDLSI", "LBDLDL1", "LBDLDL2"],
        "response",
        warnings_list,
    )
    resp = merge_first_available_name(
        resp,
        paths["response"],
        "LBXSTP",
        ["LBXSTP", "LBXSTPSI", "LBXSTP1", "LBXSTP2"],
        "response",
        warnings_list,
    )
    # Почки: креатинин сыворотки, eGFR (NHANES VNEGFR/VNEGFRADJ), мочевина крови (BUN), микроальбумин мочи
    resp = merge_first_available_name(
        resp,
        paths["response"],
        "LBXSCR",
        ["LBXSCR", "LBXSCR1", "LBXSCR2"],
        "response",
        warnings_list,
    )
    resp = merge_first_available_name(
        resp,
        paths["response"],
        "eGFR",
        ["VNEGFR", "VNEGFRADJ"],
        "response",
        warnings_list,
    )
    resp = merge_first_available_name(
        resp,
        paths["response"],
        "LBXSBU",
        ["LBXSBU", "LBXSBU1", "LBXSBU2", "LBDSBUSI", "LBDSBUSI1", "LBDSBUSI2"],
        "response",
        warnings_list,
    )
    resp = merge_first_available_name(
        resp,
        paths["response"],
        "URXUMA",
        ["URXUMA", "URXUMA1", "URXUMA2"],
        "response",
        warnings_list,
    )
    # ОАК / биохимия для формулы PhenoAge (Levin et al.): LBXSAL — альбумин сыворотки (не путать с URXUMA — моча).
    resp = merge_first_available_name(
        resp,
        paths["response"],
        "LBXSAL",
        ["LBXSAL", "LBXSAL1", "LBXSAL2"],
        "response",
        warnings_list,
    )
    resp = merge_first_available_name(
        resp,
        paths["response"],
        "LBXSAPSI",
        ["LBXSAPSI", "LBXSAPSI1", "LBXSAPSI2"],
        "response",
        warnings_list,
    )
    resp = merge_first_available_name(
        resp,
        paths["response"],
        "LBXWBCSI",
        ["LBXWBCSI", "LBXWBCSI1", "LBXWBCSI2"],
        "response",
        warnings_list,
    )
    resp = merge_first_available_name(
        resp,
        paths["response"],
        "LBXLYPCT",
        ["LBXLYPCT", "LBXLYPCT1", "LBXLYPCT2"],
        "response",
        warnings_list,
    )
    resp = merge_first_available_name(
        resp,
        paths["response"],
        "LBXMCVSI",
        ["LBXMCVSI", "LBXMCVSI1", "LBXMCVSI2"],
        "response",
        warnings_list,
    )
    resp = merge_first_available_name(
        resp,
        paths["response"],
        "LBXRDW",
        ["LBXRDW", "LBXRDW1", "LBXRDW2"],
        "response",
        warnings_list,
    )
    df = demo.merge(q, on="SEQN", how="inner", suffixes=("", "_q"))
    df = df.merge(resp, on="SEQN", how="inner", suffixes=("", "_r"))
    df = merge_first_available_name(
        df,
        paths["chemicals"],
        "LBXVIDMS",
        ["LBXVIDMS", "LBXVIDMS1", "LBXVIDMS2"],
        "chemicals",
        warnings_list,
    )
    return df


# Refused / Don't know (опросники NHANES) — не использовать как числовые значения.
QUESTIONNAIRE_SENTINEL = {7, 9, 77, 99}


def add_derived_features(df: pd.DataFrame, warnings_list: list[str]) -> pd.DataFrame:
    """
    Восстановление ИМТ из веса/роста при пропуске BMXBMI; бинарный признак гипертензии по АД и опросу.
    Прочие производные (WHtR/WHR, HOMA, TyG, …) — в ноутбуке `add_engineered_features`.
    """
    out = df.copy()
    bmi = pd.to_numeric(out["BMXBMI"], errors="coerce")
    wt = pd.to_numeric(out["BMXWT"], errors="coerce")
    ht_cm = pd.to_numeric(out["BMXHT"], errors="coerce")
    computed = wt / ((ht_cm / 100.0) ** 2)
    mask = bmi.isna() & computed.notna()
    if mask.any():
        bmi = bmi.where(~mask, computed)
    out["BMXBMI"] = bmi
    if bmi.isna().all():
        warn("BMXBMI и расчёт из веса/роста недоступны.", warnings_list)

    sys_bp = pd.to_numeric(out["BPXSY1"], errors="coerce")
    dia_bp = pd.to_numeric(out["BPXDI1"], errors="coerce")
    bpq20 = pd.to_numeric(out["BPQ020"], errors="coerce")
    bpq30 = pd.to_numeric(out["BPQ030"], errors="coerce")
    bpq20 = bpq20.mask(bpq20.isin(QUESTIONNAIRE_SENTINEL), np.nan)
    bpq30 = bpq30.mask(bpq30.isin(QUESTIONNAIRE_SENTINEL), np.nan)
    out["hypertension"] = (
        (bpq20 == 1)
        | (bpq30 == 1)
        | (sys_bp >= 140)
        | (dia_bp >= 90)
    ).astype(int)

    return out


def clean_new_metabolic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Новые признаки: печень (LBXSAT, LBXSASS), почки (LBXSCR, eGFR, LBXSBU, URXUMA),
    наследственность (MCQ300C); анамнез ССЗ (MCQ160A/E/F); самооценка здоровья (HSD010).

    PhenoAge / ОАК: LBXSAL (альбумин сыворотки, г/дл), LBXSAPSI (щелочная фосфатаза), LBXWBCSI (WBC),
    LBXLYPCT (% лимфоцитов), LBXMCVSI (MCV), LBXRDW (RDW). URXUMA — микроальбумин мочи, отдельно от LBXSAL.

    MCQ300C: трёхзначное кодирование: 1=да, 0=нет, -1=неизвестно.
    Непрерывные: только to_numeric; пост-сплит импьютация — в build_preprocessor.
    """
    out = df.copy()
    for col in (
        *CV_HISTORY_AND_SELF_HEALTH_COLS,
        *DIQ_COMPLICATION_COLS,
        "MCQ300C",
    ):
        if col not in out.columns:
            continue
        v = pd.to_numeric(out[col], errors="coerce")
        out[col] = v.mask(v.isin(QUESTIONNAIRE_SENTINEL), np.nan)

    if "PAD680" in out.columns:
        v = pd.to_numeric(out["PAD680"], errors="coerce")
        bad_pad680 = QUESTIONNAIRE_SENTINEL | {7777, 9999, 99999}
        out["PAD680"] = v.mask(v.isin(bad_pad680), np.nan)

    if "MCQ300C" in out.columns:
        mcq = pd.to_numeric(out["MCQ300C"], errors="coerce")
        out["MCQ300C"] = np.where(mcq == 1, 1, np.where(mcq == 2, 0, -1)).astype(np.int8)

    for col in ("MCQ160A", "MCQ160E", "MCQ160F"):
        if col not in out.columns:
            continue
        v = pd.to_numeric(out[col], errors="coerce").fillna(2.0)
        out[col] = np.where(v == 1, 1, np.where(v == 2, 0, 0)).astype(np.int8)

    if "RIAGENDR" in out.columns:
        v = pd.to_numeric(out["RIAGENDR"], errors="coerce").fillna(1.0)
        out["RIAGENDR"] = np.where(v == 1, 0, np.where(v == 2, 1, 0)).astype(np.int8)

    for col in ("DIQ090", "DIQ100"):
        if col not in out.columns:
            continue
        v = pd.to_numeric(out[col], errors="coerce").fillna(2.0)
        out[col] = np.where(v == 1, 1, np.where(v == 2, 0, 0)).astype(np.int8)

    for col in ("BPQ020", "BPQ030"):
        if col not in out.columns:
            continue
        v = pd.to_numeric(out[col], errors="coerce").fillna(2.0)
        out[col] = np.where(v == 1, 1, np.where(v == 2, 0, 0)).astype(np.int8)

    if "DBQ700" in out.columns:
        v = pd.to_numeric(out["DBQ700"], errors="coerce")
        out["DBQ700"] = v.mask(v.isin(QUESTIONNAIRE_SENTINEL), np.nan)

    for col in (
        "LBXSAT",
        "LBXSASS",
        *CV_HISTORY_AND_SELF_HEALTH_COLS,
        *DIQ_COMPLICATION_COLS,
        "LBXSUA",
        "LBXVIDMS",
        "LBXSCR",
        "eGFR",
        "LBXSBU",
        "URXUMA",
        "LBXHGB",
        "LBXRBCSI",
        "LBXSTB",
        "LBDLDL",
        "LBXSTP",
        "LBXSAL",
        "LBXSAPSI",
        "LBXWBCSI",
        "LBXLYPCT",
        "LBXMCVSI",
        "LBXRDW",
        "LBXCRP",
        "DBQ700",
        "PAD680",
    ):
        if col not in out.columns:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def build_target_from_hba1c(df: pd.DataFrame, hba1c_col: str) -> pd.Series:
    """
    Три класса только по числовому HbA1c (ADA): <5.7 → 0; [5.7, 6.5) → 1; ≥6.5 → 2.
    При пропуске HbA1c — NaN (строка отбрасывается вызывающим кодом).
    """
    vals = pd.to_numeric(df[hba1c_col], errors="coerce")
    t = pd.Series(np.nan, index=df.index, dtype=float)
    t = t.mask(vals < 5.7, 0)
    t = t.mask((vals >= 5.7) & (vals < 6.5), 1)
    t = t.mask(vals >= 6.5, 2)
    return t


# Ключевые лабораторные признаки: строки с пропусками удаляются, а не заполняются.
# Эти показатели берутся из одного забора крови натощак; если их нет —
# участник не проходил утреннюю лабораторную сессию, и заполнение медианой
# создаёт ~46% одинаковых значений, что ломает предсказательную силу.
CRITICAL_LAB_FEATURES = frozenset({"LBXGLU"})


def drop_missing_critical_lab(
    X: pd.DataFrame, y: pd.Series, warnings_list: list[str], seqn: pd.Series
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Удаляет строки, где пропущены ключевые лабораторные признаки."""
    critical_present = [c for c in CRITICAL_LAB_FEATURES if c in X.columns]
    if not critical_present:
        return X, y, seqn.reset_index(drop=True)
    mask = X[critical_present].notna().all(axis=1)
    n_dropped = (~mask).sum()
    if n_dropped > 0:
        warn(
            f"Удалено {n_dropped} строк без лабораторных данных натощак "
            f"({', '.join(critical_present)}). "
            f"Осталось {mask.sum()} строк.",
            warnings_list,
        )
    return (
        X.loc[mask].reset_index(drop=True),
        y.loc[mask].reset_index(drop=True),
        seqn.loc[mask].reset_index(drop=True),
    )


def impute_numeric_grouped(
    X: pd.DataFrame,
    numeric_cols: list[str],
    group_cols: list[str] | None = None,
) -> None:
    """
    Импьютация медианой по группам (возраст × пол), затем fallback на
    глобальную медиану. Для ключевых лабораторных — пропуски уже удалены
    на предыдущем шаге.
    """
    cols_to_impute = [c for c in numeric_cols if c not in CRITICAL_LAB_FEATURES]
    if group_cols is None:
        group_cols = []
    # Определяем столбцы для группировки, которые реально есть в X
    available_groups = [g for g in group_cols if g in X.columns]

    for c in cols_to_impute:
        if X[c].isna().sum() == 0:
            continue
        # Групповая медиана
        if available_groups:
            group_med = X.groupby(available_groups, observed=True)[c].transform("median")
            X[c] = X[c].fillna(group_med)
        # Глобальная медиана как fallback
        med = X[c].median()
        if pd.isna(med):
            med = 0.0
        X[c] = X[c].fillna(med)


# Обратная совместимость: старое имя → новая функция
def impute_numeric_median(X: pd.DataFrame, numeric_cols: list[str]) -> None:
    impute_numeric_grouped(X, numeric_cols, group_cols=["RIDAGEYR", "RIAGENDR"])


def format_category_label(value: Any) -> str:
    if pd.isna(value) or value is None or (isinstance(value, float) and np.isnan(value)):
        return "missing"
    if isinstance(value, (np.floating, float)) and np.isfinite(value) and float(value).is_integer():
        return str(int(value))
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    return str(value).strip()


def impute_categorical_mode_or_missing(X: pd.DataFrame, cat_cols: list[str]) -> None:
    for c in cat_cols:
        col = X[c].map(format_category_label)
        mode = col.mode(dropna=True)
        fill = mode.iloc[0] if len(mode) > 0 else "missing"
        X[c] = col.replace("nan", "missing").fillna(fill)


# Бинарные / почти бинарные признаки: перцентильный clip (1–99%) может «съесть» редкие 1 при доминировании 0.
SKIP_PERCENTILE_CLIP = frozenset(
    {
        "hypertension",
        "gum_disease",
        "MCQ300C",
        "HSD010",
        "DIQ090",
        "DIQ100",
    }
)


def clip_numeric_percentiles(X: pd.DataFrame, numeric_cols: list[str], low: float, high: float) -> None:
    for c in numeric_cols:
        s = pd.to_numeric(X[c], errors="coerce")
        lo = s.quantile(low)
        hi = s.quantile(high)
        if pd.notna(lo) and pd.notna(hi):
            X[c] = s.clip(lower=lo, upper=hi)


class NHANESPreprocessor:
    """Preprocessor to fit on train and transform test."""

    def __init__(
        self,
        numeric_cols: list[str],
        categorical_cols: list[str],
        skip_percentile_clip: frozenset[str],
        clip_low: float = 0.01,
        clip_high: float = 0.99,
    ) -> None:
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.skip_percentile_clip = skip_percentile_clip
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.numeric_medians: dict[str, float] = {}
        self.clip_bounds: dict[str, tuple[float, float]] = {}
        self.cat_fill_values: dict[str, str] = {}
        self.cat_encoders: dict[str, LabelEncoder] = {}
        self.label_mappings: dict[str, dict[str, int]] = {}
        self.lbxcrp_global_median: float = 0.0
        self.lbxcrp_group_medians: dict[tuple[int, int, int], float] = {}
        self.lbxcrp_bmi_quartiles: tuple[float, float, float] | None = None

    @staticmethod
    def _mask_dmdeduc2_sentinel(X: pd.DataFrame) -> None:
        if "DMDEDUC2" in X.columns:
            v = pd.to_numeric(X["DMDEDUC2"], errors="coerce")
            X["DMDEDUC2"] = v.mask(v.isin(QUESTIONNAIRE_SENTINEL), np.nan)

    def _fit_lbxcrp_group_stats(self, X: pd.DataFrame) -> None:
        if "LBXCRP" not in X.columns:
            return
        crp = pd.to_numeric(X["LBXCRP"], errors="coerce")
        med = crp.median()
        self.lbxcrp_global_median = float(0.0 if pd.isna(med) else med)
        if not {"RIDAGEYR", "RIAGENDR", "BMXBMI"}.issubset(X.columns):
            return

        bmi = pd.to_numeric(X["BMXBMI"], errors="coerce")
        q1, q2, q3 = bmi.quantile([0.25, 0.5, 0.75]).tolist()
        if not (pd.notna(q1) and pd.notna(q2) and pd.notna(q3)):
            return
        if not (q1 < q2 < q3):
            return
        self.lbxcrp_bmi_quartiles = (float(q1), float(q2), float(q3))

        age = pd.to_numeric(X["RIDAGEYR"], errors="coerce")
        sex = pd.to_numeric(X["RIAGENDR"], errors="coerce")
        bmi_q = pd.cut(
            bmi,
            bins=[-np.inf, float(q1), float(q2), float(q3), np.inf],
            labels=False,
            include_lowest=True,
        )
        grp_df = pd.DataFrame(
            {
                "age": age,
                "sex": sex,
                "bmi_q": bmi_q,
                "crp": crp,
            }
        ).dropna(subset=["age", "sex", "bmi_q", "crp"])
        if grp_df.empty:
            return

        grouped = grp_df.groupby(["age", "sex", "bmi_q"], observed=True)["crp"].median()
        self.lbxcrp_group_medians = {
            (int(a), int(s), int(b)): float(v) for (a, s, b), v in grouped.items()
        }

    def _impute_lbxcrp_grouped(self, X: pd.DataFrame) -> None:
        if "LBXCRP" not in X.columns:
            return
        s = pd.to_numeric(X["LBXCRP"], errors="coerce")
        na_mask = s.isna()
        if not na_mask.any():
            if "LBXCRP" in self.clip_bounds:
                lo, hi = self.clip_bounds["LBXCRP"]
                s = s.clip(lower=lo, upper=hi)
            X["LBXCRP"] = s
            return

        can_use_group = (
            self.lbxcrp_bmi_quartiles is not None
            and {"RIDAGEYR", "RIAGENDR", "BMXBMI"}.issubset(X.columns)
            and len(self.lbxcrp_group_medians) > 0
        )
        if can_use_group:
            q1, q2, q3 = self.lbxcrp_bmi_quartiles
            age = pd.to_numeric(X["RIDAGEYR"], errors="coerce")
            sex = pd.to_numeric(X["RIAGENDR"], errors="coerce")
            bmi = pd.to_numeric(X["BMXBMI"], errors="coerce")
            bmi_q = pd.cut(
                bmi,
                bins=[-np.inf, q1, q2, q3, np.inf],
                labels=False,
                include_lowest=True,
            )
            key_mask = na_mask & age.notna() & sex.notna() & bmi_q.notna()
            if key_mask.any():
                keys = list(
                    zip(
                        age.loc[key_mask].astype(int),
                        sex.loc[key_mask].astype(int),
                        bmi_q.loc[key_mask].astype(int),
                    )
                )
                mapped = pd.Series(
                    [self.lbxcrp_group_medians.get(k, np.nan) for k in keys],
                    index=s.loc[key_mask].index,
                    dtype=float,
                )
                s.loc[key_mask] = mapped

        s = s.fillna(self.lbxcrp_global_median)
        if "LBXCRP" in self.clip_bounds:
            lo, hi = self.clip_bounds["LBXCRP"]
            s = s.clip(lower=lo, upper=hi)
        X["LBXCRP"] = s

    def fit(self, X_train: pd.DataFrame) -> "NHANESPreprocessor":
        X = X_train.copy()
        self._mask_dmdeduc2_sentinel(X)
        self._fit_lbxcrp_group_stats(X)

        for c in self.numeric_cols:
            if c not in X.columns:
                continue
            s = pd.to_numeric(X[c], errors="coerce")
            med = s.median()
            self.numeric_medians[c] = float(0.0 if pd.isna(med) else med)
            if c in self.skip_percentile_clip:
                continue
            lo = s.quantile(self.clip_low)
            hi = s.quantile(self.clip_high)
            if pd.notna(lo) and pd.notna(hi):
                self.clip_bounds[c] = (float(lo), float(hi))

        for c in self.categorical_cols:
            if c not in X.columns:
                continue
            col = X[c].map(format_category_label)
            mode = col.mode(dropna=True)
            fill = mode.iloc[0] if len(mode) > 0 else "missing"
            self.cat_fill_values[c] = fill
            le = LabelEncoder()
            values = col.replace("nan", "missing").fillna(fill)
            le.fit(values)
            self.cat_encoders[c] = le
            self.label_mappings[c] = {str(lab): int(i) for i, lab in enumerate(le.classes_)}
        return self

    def transform(self, X_data: pd.DataFrame) -> pd.DataFrame:
        X = X_data.copy()
        self._mask_dmdeduc2_sentinel(X)
        self._impute_lbxcrp_grouped(X)

        for c, med in self.numeric_medians.items():
            if c not in X.columns:
                continue
            if c == "LBXCRP":
                continue
            s = pd.to_numeric(X[c], errors="coerce").fillna(med)
            if c in self.clip_bounds:
                lo, hi = self.clip_bounds[c]
                s = s.clip(lower=lo, upper=hi)
            X[c] = s

        for c, fill in self.cat_fill_values.items():
            if c not in X.columns:
                continue
            values = X[c].map(format_category_label).replace("nan", "missing").fillna(fill)
            le = self.cat_encoders.get(c)
            if le is None:
                continue
            known = set(le.classes_)
            values = values.map(lambda v: v if v in known else fill)
            X[c] = le.transform(values)
        return X


def build_preprocessor(
    X_train: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    skip_percentile_clip: frozenset[str],
    clip_low: float = 0.01,
    clip_high: float = 0.99,
) -> NHANESPreprocessor:
    prep = NHANESPreprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        skip_percentile_clip=skip_percentile_clip,
        clip_low=clip_low,
        clip_high=clip_high,
    )
    return prep.fit(X_train)


def main() -> None:
    parser = argparse.ArgumentParser(description="NHANES → nhanes_ml_ready.csv")
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--output-meta", type=Path, default=OUTPUT_META)
    args = parser.parse_args()

    warnings_list: list[str] = []
    paths = {k: Path(v) for k, v in DEFAULT_FILES.items()}
    for name, p in paths.items():
        if not p.is_file():
            print(f"Ошибка: нет файла {p} ({name})", file=sys.stderr)
            sys.exit(1)

    df = load_merged_inner(paths, warnings_list)
    hba1c_col = resolve_hba1c_column(df.columns)

    if hba1c_col is None:
        resp_header = pd.read_csv(paths["response"], nrows=0)
        if "LBXGLU" in resp_header.columns:
            print(
                "Ошибка: в объединённых данных нет столбца HbA1c (LBXGH и аналоги). "
                "Альтернатива: целевая переменная по натощак глюкозе LBXGLU (ADA <100 / 100–125 / ≥126). "
                "Добавьте колонку LBXGH или укажите другой источник.",
                file=sys.stderr,
            )
        else:
            print("Ошибка: нет HbA1c и нет LBXGLU.", file=sys.stderr)
        sys.exit(1)

    gh_cols_to_drop = [c for c in hba1c_feature_columns(df.columns) if c in df.columns]
    if hba1c_col not in gh_cols_to_drop:
        gh_cols_to_drop.append(hba1c_col)

    df["target"] = build_target_from_hba1c(df, hba1c_col)
    n_before_target = len(df)
    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)
    n_after_target = len(df)

    # Колонки HbA1c (сырой) убираем из признаков — только таргет; DIQ010 — утечка диагноза.
    cols_drop_after_target = [*gh_cols_to_drop, "DIQ010"]
    df = df.drop(columns=[c for c in cols_drop_after_target if c in df.columns], errors="ignore")

    df = add_derived_features(df, warnings_list)
    df = clean_new_metabolic_features(df)
    # Убраны мёртвые/бесполезные признаки:
    #   WHR      — >91% строк заполнены одной медианой (данные отсутствовали)
    #   gum_disease       — 100% нулей после left join (данные не совпали по SEQN)
    #   eating_window_hours — 100% одно значение (данные не совпали по SEQN)
    #   avg_daily_mims    — 100% одно значение (данные не совпали по SEQN)
    feature_candidates = [
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
        "LBXSCR",
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
        *CV_HISTORY_AND_SELF_HEALTH_COLS,
        "DBQ700",
        "PAD680",
    ]
    present = [c for c in feature_candidates if c in df.columns]
    missing_feats = [c for c in feature_candidates if c not in df.columns]
    for m in missing_feats:
        warn(f"Признак отсутствует и пропущен: {m}", warnings_list)

    X = df[present].copy()
    y = df["target"].copy()
    seqn = df["SEQN"].copy()

    # --- Удаляем строки без ключевой лаборатории натощак (GLU) ---
    # Участники без утренней лабсессии — пропуски по этим полям.
    # Заполнение медианой создавало ~46% одинаковых значений (data contamination).
    X, y, seqn = drop_missing_critical_lab(X, y, warnings_list, seqn)

    if "RIDAGEYR" in X.columns:
        age = pd.to_numeric(X["RIDAGEYR"], errors="coerce")
        adult_mask = age >= 18
        n_dropped_age = int((~adult_mask).sum())
        if n_dropped_age > 0:
            warn(f"Удалено {n_dropped_age} строк с RIDAGEYR < 18 (только взрослые 18+).", warnings_list)
        X = X.loc[adult_mask].reset_index(drop=True)
        y = y.loc[adult_mask].reset_index(drop=True)
        seqn = seqn.loc[adult_mask].reset_index(drop=True)

    # Сохраняем сырые признаки без предсплитовой импьютации/клиппинга/кодирования.
    # DMDEDUC2: sentinel 7/9 -> NaN до любого downstream-препроцессинга.
    if "DMDEDUC2" in X.columns:
        v = pd.to_numeric(X["DMDEDUC2"], errors="coerce")
        X["DMDEDUC2"] = v.mask(v.isin(QUESTIONNAIRE_SENTINEL), np.nan).round().astype("Int64")

    if "HSD010" in X.columns:
        X["HSD010"] = pd.to_numeric(X["HSD010"], errors="coerce").round().astype("Int64")

    missing_before = X.isna().sum().to_dict()
    missing_before = {str(k): int(v) for k, v in missing_before.items()}

    encoders_map: dict[str, dict[str, int]] = {}

    out_df = X.copy()
    out_df.insert(0, "SEQN", seqn.values)
    out_df["target"] = y.values

    out_df.to_csv(args.output_csv, index=False)

    counts = y.value_counts().sort_index()
    class_dist = {str(int(k)): int(v) for k, v in counts.items()}
    seqn_duplicates_count = int(seqn.duplicated().sum())

    meta: MetaJson = {
        "target": {
            "name": "target",
            "type": "multiclass_integer",
            "n_classes": 3,
            "encoding": {
                "0": "HbA1c < 5.7% (норма)",
                "1": "5.7% ≤ HbA1c < 6.5% (преддиабет)",
                "2": "HbA1c ≥ 6.5% (диабет)",
            },
            "rows_dropped_for_missing_hba1c": int(n_before_target - n_after_target),
        },
        "feature_columns": list(X.columns),
        "class_distribution": class_dist,
        "missing_counts_before": missing_before,
        "label_encoders_mapping": encoders_map,
        "nominal_columns_post_split": {},
        "seqn_duplicates_count": seqn_duplicates_count,
        "warnings": warnings_list,
        "hba1c_column_resolved": hba1c_col,
        "dropped_hba1c_columns": gh_cols_to_drop,
        "dropped_dead_features": ["WHR", "gum_disease", "eating_window_hours", "avg_daily_mims"],
        "critical_lab_features_require_real_data": sorted(CRITICAL_LAB_FEATURES),
        "age_policy": "Drop only RIDAGEYR < 18; no upper-age cap (NHANES top-coding kept as-is).",
        "imputation_strategy": "drop rows missing critical lab (LBXGLU); "
                               "all other preprocessing deferred to post-split build_preprocessor",
        "inner_join_rows": int(n_after_target),
        "final_rows": int(len(out_df)),
        "output_csv": str(args.output_csv),
        "output_meta": str(args.output_meta),
    }
    with open(args.output_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    total_na_before = sum(missing_before.values())
    print(f"Пропуски по признакам (raw): всего {total_na_before}")
    print(f"Дубликаты по SEQN в итоговом датасете: {seqn_duplicates_count}")
    print(f"Сохранено: {args.output_csv} ({len(out_df)} строк, {out_df.shape[1]} столбцов)")
    print(f"Метаданные: {args.output_meta}")
    for t in (0, 1, 2):
        c = int(class_dist.get(str(t), 0))
        pct = 100.0 * c / len(out_df) if len(out_df) else 0.0
        print(f"  target={t}: {c} ({pct:.2f}%)")
    if warnings_list:
        print("\nПредупреждения:")
        for w in warnings_list:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
