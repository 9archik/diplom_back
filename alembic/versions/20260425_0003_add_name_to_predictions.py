"""add name to predictions

Revision ID: 20260425_0003
Revises: 20260421_0002
Create Date: 2026-04-25 13:42:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260425_0003"
down_revision: Union[str, None] = "20260421_0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("predictions", sa.Column("name", sa.String(length=255), nullable=True))
    op.execute("UPDATE predictions SET name = 'Без названия' WHERE name IS NULL")
    with op.batch_alter_table("predictions") as batch_op:
        batch_op.alter_column("name", existing_type=sa.String(length=255), nullable=False)


def downgrade() -> None:
    with op.batch_alter_table("predictions") as batch_op:
        batch_op.drop_column("name")
