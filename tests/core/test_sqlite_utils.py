"""Unit tests for ``robotsix_llmio.core.sqlite_utils``."""

from __future__ import annotations

import sqlite3
from typing import Any
from unittest.mock import MagicMock

from robotsix_llmio.core.sqlite_utils import (
    _SQLiteConn,
    add_column_if_missing,
    run_additive_migrations,
)

# ---------------------------------------------------------------------------
# _SQLiteConn protocol
# ---------------------------------------------------------------------------


def test_sqliteconn_protocol_is_runtime_checkable() -> None:
    """``_SQLiteConn`` is decorated with ``@runtime_checkable``."""
    assert hasattr(_SQLiteConn, "_is_runtime_protocol")
    assert _SQLiteConn._is_runtime_protocol is True


def test_real_sqlite3_connection_satisfies_protocol() -> None:
    """A real ``sqlite3.Connection`` structurally matches ``_SQLiteConn``."""
    conn = sqlite3.connect(":memory:")
    try:
        assert isinstance(conn, _SQLiteConn)
    finally:
        conn.close()


class _FakeConn:
    """Minimal concrete type satisfying ``_SQLiteConn`` structurally."""

    def execute(self, sql: str, parameters: object = None) -> Any:
        return self

    def fetchall(self) -> list[tuple]:
        return []

    def commit(self) -> None:
        pass


def test_fake_conn_satisfies_protocol() -> None:
    """A concrete class with ``execute`` and ``commit`` passes ``isinstance``."""
    assert isinstance(_FakeConn(), _SQLiteConn)


def test_mock_missing_execute_fails_protocol() -> None:
    """A mock without ``execute`` does not satisfy ``_SQLiteConn``."""
    mock = MagicMock(spec=["commit"])
    assert not isinstance(mock, _SQLiteConn)


# ---------------------------------------------------------------------------
# add_column_if_missing
# ---------------------------------------------------------------------------


def _make_conn_with_columns(*column_names: str) -> MagicMock:
    """Return a mock connection whose ``PRAGMA table_info`` returns rows
    simulating the given column names."""
    conn = MagicMock(spec=["execute", "commit"])
    # Each row from PRAGMA table_info has cid, name, type, notnull, dflt_value, pk
    rows = [(i, name, "TEXT", 0, None, 0) for i, name in enumerate(column_names)]
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = rows
    conn.execute.return_value = mock_cursor
    return conn


def test_add_column_if_missing_column_exists() -> None:
    """When the column already exists, return False and do not ALTER."""
    conn = _make_conn_with_columns("id", "name", "email")
    result = add_column_if_missing(conn, "users", "email TEXT NOT NULL")
    assert result is False
    # Only the PRAGMA call should have been made; no ALTER.
    assert conn.execute.call_count == 1
    conn.commit.assert_not_called()


def test_add_column_if_missing_column_missing() -> None:
    """When the column is missing, return True, issue ALTER, and commit."""
    conn = _make_conn_with_columns("id", "name")
    result = add_column_if_missing(conn, "users", "email TEXT NOT NULL")
    assert result is True
    assert conn.execute.call_count == 2  # PRAGMA + ALTER
    conn.commit.assert_called_once()


def test_add_column_if_missing_quoted_column_name() -> None:
    """Column name extraction strips surrounding double-quotes."""
    conn = _make_conn_with_columns("id", "name")
    result = add_column_if_missing(conn, "t", '"my col" INTEGER')
    assert result is True
    # Verify ALTER used the full DDL text.
    alter_call = conn.execute.call_args_list[1]
    assert "ADD COLUMN" in alter_call[0][0]
    assert '"my col" INTEGER' in alter_call[0][0]


def test_add_column_if_missing_preserves_ddl_text() -> None:
    """The full *column_ddl* string is passed verbatim into ALTER TABLE."""
    conn = _make_conn_with_columns("id")
    ddl = "status TEXT NOT NULL DEFAULT 'active'"
    add_column_if_missing(conn, "tasks", ddl)
    alter_call = conn.execute.call_args_list[1]
    assert alter_call[0][0] == f"ALTER TABLE tasks ADD COLUMN {ddl}"


# ---------------------------------------------------------------------------
# run_additive_migrations
# ---------------------------------------------------------------------------


def test_run_additive_migrations_empty() -> None:
    """An empty sequence produces no calls."""
    conn = MagicMock(spec=["execute", "commit"])
    run_additive_migrations(conn, "t", [])
    conn.execute.assert_not_called()
    conn.commit.assert_not_called()


def test_run_additive_migrations_single_existing() -> None:
    """A single DDL for an already-existing column results in no ALTER."""
    conn = _make_conn_with_columns("id", "name")
    run_additive_migrations(conn, "users", ["name TEXT"])
    assert conn.execute.call_count == 1  # pragma only
    conn.commit.assert_not_called()


def test_run_additive_migrations_single_missing() -> None:
    """A single DDL for a missing column results in ALTER + commit."""
    conn = _make_conn_with_columns("id")
    run_additive_migrations(conn, "users", ["email TEXT"])
    assert conn.execute.call_count == 2  # pragma + ALTER
    conn.commit.assert_called_once()


def test_run_additive_migrations_multiple_mixed() -> None:
    """Multiple DDLs: some columns exist, some don't."""
    conn = _make_conn_with_columns("id", "name")
    run_additive_migrations(conn, "users", ["name TEXT", "email TEXT", "age INTEGER"])
    # PRAGMA called 3 times (once per DDL), ALTER called 2 times (email + age)
    assert conn.execute.call_count == 5
    assert conn.commit.call_count == 2
    # Verify ALTER calls include the correct columns
    alter_calls = [
        c[0][0] for c in conn.execute.call_args_list if "ADD COLUMN" in c[0][0]
    ]
    assert "email TEXT" in alter_calls[0]
    assert "age INTEGER" in alter_calls[1]
