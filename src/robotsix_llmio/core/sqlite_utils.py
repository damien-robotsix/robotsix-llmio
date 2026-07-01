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


@runtime_checkable
class _SAConn(Protocol):
    """Structural type matching a SQLAlchemy Connection.

    SQLAlchemy 2 connections reject raw-string ``.execute()`` calls;
    ``.exec_driver_sql()`` is the bypass for raw DDL / queries.
    """

    def exec_driver_sql(self, sql: str, parameters: object = ...) -> Any: ...
    def commit(self) -> None: ...


AnyConn = _SQLiteConn | _SAConn


def _execute(conn: AnyConn, sql: str) -> Any:
    """Execute *sql* against *conn*, routing through ``exec_driver_sql``
    when the connection is a SQLAlchemy Connection."""
    if isinstance(conn, _SAConn):
        return conn.exec_driver_sql(sql)
    return conn.execute(sql)


def add_column_if_missing(conn: AnyConn, table: str, column_ddl: str) -> bool:
    """Add *column_ddl* to *table* when the column is not already present.

    Returns ``True`` when the column was newly created, ``False`` when it
    already existed.
    """
    column_name = column_ddl.strip().split(maxsplit=1)[0].strip('"').strip()
    rows = _execute(conn, f"PRAGMA table_info({table})").fetchall()
    existing = {row[1] for row in rows}
    if column_name in existing:
        return False
    _execute(conn, f"ALTER TABLE {table} ADD COLUMN {column_ddl}")
    conn.commit()
    return True


def run_additive_migrations(
    conn: AnyConn,
    table: str,
    column_ddls: Sequence[str],
) -> list[bool]:
    """Apply every DDL in *column_ddls* to *table*.

    Returns a ``list[bool]`` parallel to *column_ddls*: ``True`` when the
    column was newly added, ``False`` when it already existed.
    """
    return [add_column_if_missing(conn, table, ddl) for ddl in column_ddls]


def run_multi_table_migrations(
    conn: AnyConn,
    migrations: dict[str, Sequence[str]],
) -> dict[str, list[bool]]:
    """Apply additive column migrations across multiple tables.

    *migrations* maps each table name to its sequence of column DDL strings.
    Returns a dict mapping each table name to its per-column ``list[bool]``
    result (same semantics as :func:`run_additive_migrations`).
    """
    return {
        table: run_additive_migrations(conn, table, column_ddls)
        for table, column_ddls in migrations.items()
    }
