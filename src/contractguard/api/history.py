"""Append-only SQLite history with a separate connection for each operation."""

import sqlite3
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


class HistoryError(Exception):
    """History storage could not be opened, initialized, read, or written."""


@dataclass(frozen=True)
class CheckRecord:
    created_at: datetime
    file_name: str | None
    file_sha256: str | None
    policy_hash: str
    prompt_version: str
    followup: bool
    http_status: int
    error_code: str | None
    overall_status: str | None
    counted: bool
    llm_calls: int | None
    cache_hits: int | None
    input_tokens: int | None
    output_tokens: int | None
    elapsed_ms: float | None
    report_json: str | None


@dataclass(frozen=True)
class RuleRecord:
    rule_id: str
    decision: str
    reason: str | None


def _format_time(value: datetime) -> str:
    if value.utcoffset() is None:
        raise ValueError("History timestamps must be timezone-aware")
    utc = value.astimezone(timezone.utc)
    # Explicit year padding also handles years below 1000 on all platforms.
    return f"{utc.year:04d}" + utc.strftime("-%m-%dT%H:%M:%S.%fZ")


def _parse_time(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
        tzinfo=timezone.utc
    )


def _check_dict(row: sqlite3.Row) -> dict:
    result = dict(row)
    result["followup"] = bool(result["followup"])
    result["counted"] = bool(result["counted"])
    return result


class HistoryStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise HistoryError(str(exc)) from exc
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("BEGIN IMMEDIATE")
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version == 1:
                return
            if version != 0:
                raise HistoryError(f"Unsupported history schema version: {version}")
            for statement in (
                """CREATE TABLE checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    file_name TEXT, file_sha256 TEXT,
                    policy_hash TEXT NOT NULL, prompt_version TEXT NOT NULL,
                    followup INTEGER NOT NULL, http_status INTEGER NOT NULL,
                    error_code TEXT, overall_status TEXT,
                    counted INTEGER NOT NULL,
                    llm_calls INTEGER, cache_hits INTEGER,
                    input_tokens INTEGER, output_tokens INTEGER,
                    elapsed_ms REAL, report_json TEXT
                )""",
                """CREATE TABLE check_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    check_id INTEGER NOT NULL REFERENCES checks(id),
                    rule_id TEXT NOT NULL, decision TEXT NOT NULL, reason TEXT
                )""",
                "CREATE INDEX checks_created_at ON checks(created_at)",
                "CREATE INDEX checks_file_sha256 ON checks(file_sha256)",
                "CREATE INDEX check_rules_check_id ON check_rules(check_id)",
                "PRAGMA user_version=1",
            ):
                connection.execute(statement)

    @contextmanager
    def _connection(self):
        try:
            connection = sqlite3.connect(self.db_path, timeout=5.0)
            try:
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA busy_timeout=5000")
                connection.execute("PRAGMA foreign_keys=ON")
                with connection:
                    yield connection
            finally:
                connection.close()
        except (sqlite3.Error, OSError) as exc:
            raise HistoryError(str(exc)) from exc

    def record(self, check: CheckRecord, rules: Sequence[RuleRecord] = ()) -> int:
        created_at = _format_time(check.created_at)
        with self._connection() as connection:
            cursor = connection.execute(
                """INSERT INTO checks (
                    created_at, file_name, file_sha256, policy_hash, prompt_version,
                    followup, http_status, error_code, overall_status, counted,
                    llm_calls, cache_hits, input_tokens, output_tokens,
                    elapsed_ms, report_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    created_at, check.file_name, check.file_sha256, check.policy_hash,
                    check.prompt_version, check.followup, check.http_status,
                    check.error_code, check.overall_status, check.counted,
                    check.llm_calls, check.cache_hits, check.input_tokens,
                    check.output_tokens, check.elapsed_ms, check.report_json,
                ),
            )
            check_id = cursor.lastrowid
            connection.executemany(
                """INSERT INTO check_rules (check_id, rule_id, decision, reason)
                   VALUES (?, ?, ?, ?)""",
                [
                    (check_id, rule.rule_id, rule.decision, rule.reason)
                    for rule in rules
                ],
            )
            return check_id

    def counted_since(self, since: datetime) -> int:
        with self._connection() as connection:
            return connection.execute(
                "SELECT COUNT(*) FROM checks WHERE counted = 1 AND created_at >= ?",
                (_format_time(since),),
            ).fetchone()[0]

    def oldest_counted_since(self, since: datetime) -> datetime | None:
        with self._connection() as connection:
            value = connection.execute(
                "SELECT MIN(created_at) FROM checks "
                "WHERE counted = 1 AND created_at >= ?",
                (_format_time(since),),
            ).fetchone()[0]
            return _parse_time(value) if value is not None else None

    def list_checks(
        self, *, name=None, sha256=None, status=None,
        date_from=None, date_to=None, limit=50, offset=0,
    ) -> list[dict]:
        pattern = None
        if name is not None:
            escaped = name.replace("!", "!!").replace("%", "!%").replace("_", "!_")
            pattern = "%" + escaped + "%"
        start = _format_time(date_from) if date_from is not None else None
        end = _format_time(date_to) if date_to is not None else None
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT id, created_at, file_name, file_sha256, policy_hash,
                          prompt_version, followup, http_status, error_code,
                          overall_status, counted, llm_calls, cache_hits,
                          input_tokens, output_tokens, elapsed_ms
                   FROM checks
                   WHERE (? IS NULL OR file_name LIKE ? ESCAPE '!')
                     AND (? IS NULL OR file_sha256 = ?)
                     AND (? IS NULL OR overall_status = ? OR error_code = ?)
                     AND (? IS NULL OR created_at >= ?)
                     AND (? IS NULL OR created_at < ?)
                   ORDER BY id DESC LIMIT ? OFFSET ?""",
                (
                    pattern, pattern, sha256, sha256, status, status, status,
                    start, start, end, end, max(1, min(200, limit)), max(0, offset),
                ),
            ).fetchall()
            return [_check_dict(row) for row in rows]

    def get_check(self, check_id: int) -> dict | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM checks WHERE id = ?", (check_id,)
            ).fetchone()
            if row is None:
                return None
            result = _check_dict(row)
            result["rules"] = [
                dict(rule) for rule in connection.execute(
                    "SELECT rule_id, decision, reason FROM check_rules "
                    "WHERE check_id = ? ORDER BY id", (check_id,),
                )
            ]
            return result
