#!/usr/bin/env python3
"""Список предсказаний, для которых не хватает полей в user_input_data для PATCH .../mode."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.schemas import PredictionRequest  # noqa: E402
from ml.preprocessing import DERIVED_MODEL_FEATURES  # noqa: E402


def _required_user_input_keys() -> set[str]:
    return {
        name
        for name, finfo in PredictionRequest.model_fields.items()
        if name not in ("name", "mode") and finfo.is_required()
    }


def _user_input_snapshot(raw: dict[str, Any]) -> dict[str, Any]:
    raw_user = raw.get("user_input_data")
    raw_computed = raw.get("computed_data")
    if isinstance(raw_user, dict) and isinstance(raw_computed, dict):
        return {
            str(k): v
            for k, v in raw_user.items()
            if v is None or isinstance(v, (int, float))
        }
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if value is None or isinstance(value, (int, float)):
            if key in DERIVED_MODEL_FEATURES:
                continue
            out[str(key)] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Проверка SQLite: какие записи predictions не содержат обязательных "
            "полей PredictionRequest (кроме name/mode) в сохранённых входных данных."
        ),
    )
    parser.add_argument(
        "--db",
        default=str(ROOT / "app.db"),
        help="Путь к файлу SQLite (по умолчанию ./app.db от корня проекта)",
    )
    args = parser.parse_args()
    required = _required_user_input_keys()
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT id, name, input_data FROM predictions").fetchall()
    conn.close()

    problems: list[tuple[str, str, list[str]]] = []
    for row in rows:
        raw = row["input_data"]
        data = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(data, dict):
            problems.append((row["id"], row["name"], ["<input_data не объект>"]))
            continue
        present = set(_user_input_snapshot(data))
        missing = sorted(required - present)
        if missing:
            problems.append((row["id"], row["name"], missing))

    if not problems:
        print("Все записи содержат обязательные поля для смены режима (по ключам).")
        return

    print(f"Записей с недостающими полями: {len(problems)} (всего в БД: {len(rows)})")
    for pid, name, missing in problems:
        print(f"  {pid} | {name!r} | отсутствуют: {', '.join(missing)}")


if __name__ == "__main__":
    main()
