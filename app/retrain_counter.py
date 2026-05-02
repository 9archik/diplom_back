from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import MlRetrainCounter, MlTrainingEvent

logger = logging.getLogger(__name__)

FEEDBACK_RETRAIN_THRESHOLD = 50
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


async def record_feedback_for_retrain(db: AsyncSession) -> bool:
    """Увеличивает счётчик; при достижении порога сбрасывает и логирует событие.

    Возвращает True, если нужно запустить внешний job переобучения (после commit).
    """
    result = await db.execute(select(MlRetrainCounter).where(MlRetrainCounter.id == 1))
    counter = result.scalar_one_or_none()
    if counter is None:
        counter = MlRetrainCounter(id=1, feedback_count=0)
        db.add(counter)
        await db.flush()

    counter.feedback_count += 1
    if counter.feedback_count < FEEDBACK_RETRAIN_THRESHOLD:
        return False

    counter.feedback_count = 0
    db.add(
        MlTrainingEvent(
            reason="feedback_threshold",
            threshold=FEEDBACK_RETRAIN_THRESHOLD,
        )
    )
    return True


def schedule_retrain_subprocess() -> None:
    try:
        subprocess.Popen(
            [sys.executable, "-m", "ml.retrain_job"],
            cwd=str(_PROJECT_ROOT),
            start_new_session=True,
        )
    except OSError as exc:
        logger.exception("Не удалось запустить переобучение: %s", exc)
