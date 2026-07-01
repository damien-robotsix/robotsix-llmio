"""Unit tests for ``robotsix_llmio.core.sqlite_utils``."""

from __future__ import annotations

import sqlite3
from typing import Any
from unittest.mock import MagicMock

from robotsix_llmio.core.sqlite_utils import (
    _SAConn,
    _SQLiteConn,
    add_column_if_missing,
    run_additive_migrations,
    run_multi_table_migrations,
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


# ---------------------------------------------------------------------------
# _SAConn protocol
# ---------------------------------------------------------------------------


def test_saconn_protocol_is_runtime_checkable() -> None:
    """``_SAConn`` is decorated with ``@runtime_checkable``."""
    assert hasattr(_SAConn, "_is_runtime_protocol")
    assert _SAConn._is_runtime_protocol is True


class _FakeSAConn:
    """Concrete class with ``exec_driver_sql`` and ``commit`` for SA testing."""

    def __init__(self, *column_names: str):
        self._rows = [
            (i, name, "TEXT", 0, None, 0) for i, name in enumerate(column_names)
        ]
        self.exec_driver_sql_calls: list[str] = []
        self.commit_calls = 0

    def exec_driver_sql(self, sql: str, parameters: object = None) -> Any:
        self.exec_driver_sql_calls.append(sql)
        c = MagicMock()
        c.fetchall.return_value = self._rows
        return c

    def commit(self) -> None:
        self.commit_calls += 1


def test_fake_saconn_satisfies_protocol() -> None:
    """A concrete class with ``exec_driver_sql`` + ``commit`` passes ``isinstance``."""
    assert isinstance(_FakeSAConn(), _SAConn)


def test_mock_missing_exec_driver_sql_fails_saconn() -> None:
    """A mock with only ``commit`` does not satisfy ``_SAConn``."""
    mock = MagicMock(spec=["commit"])
    assert not isinstance(mock, _SAConn)


# ---------------------------------------------------------------------------
# add_column_if_missing with SA connection fallback
# ---------------------------------------------------------------------------


def _new_sa_conn(*column_names: str) -> _FakeSAConn:
    """Return a ``_FakeSAConn`` preloaded with the given column names."""
    return _FakeSAConn(*column_names)


def test_add_column_if_missing_sa_conn_exists() -> None:
    """Column already present → ``False``, only ``exec_driver_sql`` called."""
    conn = _new_sa_conn("id", "name", "email")
    result = add_column_if_missing(conn, "users", "email TEXT NOT NULL")
    assert result is False
    assert len(conn.exec_driver_sql_calls) == 1
    assert conn.commit_calls == 0


def test_add_column_if_missing_sa_conn_missing() -> None:
    """Column absent → ``True``, ``exec_driver_sql`` called twice, ``commit`` once."""
    conn = _new_sa_conn("id", "name")
    result = add_column_if_missing(conn, "users", "email TEXT NOT NULL")
    assert result is True
    assert len(conn.exec_driver_sql_calls) == 2
    assert conn.commit_calls == 1


# ---------------------------------------------------------------------------
# run_additive_migrations returns list[bool]
# ---------------------------------------------------------------------------


def test_run_additive_migrations_returns_empty_list() -> None:
    """Empty DDL sequence → ``[]``."""
    conn = _make_conn_with_columns("id")
    result = run_additive_migrations(conn, "t", [])
    assert result == []
    conn.execute.assert_not_called()


def test_run_additive_migrations_returns_list_all_existing() -> None:
    """All columns present → ``[False, False]``."""
    conn = _make_conn_with_columns("id", "name", "email")
    result = run_additive_migrations(conn, "users", ["name TEXT", "email TEXT"])
    assert result == [False, False]


def test_run_additive_migrations_returns_mixed_list() -> None:
    """Some present, some missing → mixed ``[False, True, True]``."""
    conn = _make_conn_with_columns("id", "name")
    result = run_additive_migrations(
        conn, "users", ["name TEXT", "email TEXT", "age INTEGER"]
    )
    assert result == [False, True, True]


# ---------------------------------------------------------------------------
# run_multi_table_migrations
# ---------------------------------------------------------------------------


def test_run_multi_table_migrations_empty_dict() -> None:
    """Empty migrations dict → ``{}``."""
    conn = MagicMock(spec=["execute", "commit"])
    result = run_multi_table_migrations(conn, {})
    assert result == {}
    conn.execute.assert_not_called()


def test_run_multi_table_migrations_single_table() -> None:
    """Single table matches ``run_additive_migrations`` result."""
    conn = _make_conn_with_columns("id", "name")
    result = run_multi_table_migrations(
        conn, {"users": ["name TEXT", "email TEXT", "age INTEGER"]}
    )
    assert list(result.keys()) == ["users"]
    assert result["users"] == [False, True, True]


def test_run_multi_table_migrations_multiple_tables() -> None:
    """Two tables with different columns — independent per-table results."""
    conn = _make_conn_with_columns("id", "name")
    result = run_multi_table_migrations(
        conn,
        {
            "users": ["name TEXT", "email TEXT"],
            "tasks": ["title TEXT", "status TEXT", "priority INTEGER"],
        },
    )
    assert list(result.keys()) == ["users", "tasks"]
    assert result["users"] == [False, True]
    assert result["tasks"] == [True, True, True]


def test_run_multi_table_migrations_sa_conn() -> None:
    """Multi-table through a mock SA connection — exercises ``exec_driver_sql``."""
    conn = _new_sa_conn("id", "name")
    result = run_multi_table_migrations(
        conn,
        {
            "users": ["name TEXT", "email TEXT"],
            "tasks": ["title TEXT"],
        },
    )
    assert result == {"users": [False, True], "tasks": [True]}
    # exec_driver_sql is the only call path for SA conn
    # 2 users pragmas (name + email) + 1 users ALTER (email) +
    # 1 tasks pragma (title) + 1 tasks ALTER (title) = 5
    assert len(conn.exec_driver_sql_calls) == 5
