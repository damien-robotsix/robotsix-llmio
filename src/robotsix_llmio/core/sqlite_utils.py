"""Additive SQLite column migration utilities.

promotable verbatim into a fleet-shared library
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class _SQLiteConn(Protocol):
    """Structural type matching any connection that offers ``execute`` + ``commit``.

    Both :class:`sqlite3.Connection` and SQLAlchemy
    :class:`~sqlalchemy.engine.Connection` satisfy this protocol, so the
    functions in this module work with either backend without an adapter.
    """

    def execute(self, sql: str, parameters: object = ...) -> Any: ...
    def commit(self) -> None: ...


def add_column_if_missing(conn: _SQLiteConn, table: str, column_ddl: str) -> bool:
    """Add *column_ddl* to *table* when the column is not already present.

    Returns ``True`` when the column was newly created, ``False`` when it
    already existed.
    """
    column_name = column_ddl.strip().split(maxsplit=1)[0].strip('"').strip()
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    existing = {row[1] for row in rows}
    if column_name in existing:
        return False
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column_ddl}")
    conn.commit()
    return True


def run_additive_migrations(
    conn: _SQLiteConn,
    table: str,
    column_ddls: Sequence[str],
) -> None:
    """Apply every DDL in *column_ddls* to *table* via :func:`add_column_if_missing`."""
    for ddl in column_ddls:
        add_column_if_missing(conn, table, ddl)
