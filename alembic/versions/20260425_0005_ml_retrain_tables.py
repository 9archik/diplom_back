"""ml retrain counter and training events

Revision ID: 20260425_0005
Revises: 20260425_0004
Create Date: 2026-04-25 20:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260425_0005"
down_revision: Union[str, None] = "20260425_0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    names = insp.get_table_names()
    if "ml_retrain_counter" not in names:
        op.create_table(
            "ml_retrain_counter",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("feedback_count", sa.Integer(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
    if "ml_training_events" not in names:
        op.create_table(
            "ml_training_events",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=False), nullable=False),
            sa.Column("reason", sa.String(length=64), nullable=False),
            sa.Column("threshold", sa.Integer(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
    op.execute(
        sa.text(
            "INSERT OR IGNORE INTO ml_retrain_counter (id, feedback_count) VALUES (1, 0)"
        )
    )


def downgrade() -> None:
    op.execute(sa.text("DROP TABLE IF EXISTS ml_training_events"))
    op.execute(sa.text("DROP TABLE IF EXISTS ml_retrain_counter"))
