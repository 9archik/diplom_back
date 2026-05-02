"""drop hba1c from predictions

Revision ID: 20260425_0004
Revises: 20260425_0003
Create Date: 2026-04-25 18:26:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260425_0004"
down_revision: Union[str, None] = "20260425_0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("predictions") as batch_op:
        batch_op.drop_column("hba1c")


def downgrade() -> None:
    with op.batch_alter_table("predictions") as batch_op:
        batch_op.add_column(sa.Column("hba1c", sa.Float(), nullable=True))
