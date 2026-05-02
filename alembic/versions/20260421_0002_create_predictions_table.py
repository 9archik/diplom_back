"""create predictions table

Revision ID: 20260421_0002
Revises: 20260421_0001
Create Date: 2026-04-21 00:10:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260421_0002"
down_revision: Union[str, None] = "20260421_0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "predictions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("mode", sa.String(length=50), nullable=False),
        sa.Column("input_data", sa.JSON(), nullable=False),
        sa.Column("predicted_class", sa.Integer(), nullable=False),
        sa.Column("probabilities", sa.JSON(), nullable=False),
        sa.Column("top_factors", sa.JSON(), nullable=False),
        sa.Column("real_class", sa.Integer(), nullable=True),
        sa.Column("hba1c", sa.Float(), nullable=True),
        sa.Column("feedback_comment", sa.Text(), nullable=True),
        sa.Column("feedback_at", sa.DateTime(timezone=False), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=False), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_predictions_user_id"), "predictions", ["user_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_predictions_user_id"), table_name="predictions")
    op.drop_table("predictions")
