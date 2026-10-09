"""A SQLite incident repository: the review queue and its audit trail survive a restart.

Each record is stored whole as its JSON wire form, with the columns the store filters on
(status, dedup key, retention deadline) pulled out and indexed. One connection is shared across
threads behind a lock; :class:`~llm_guardian.incidents.IncidentStore` already serialises the
read-modify-write of a fold, so the lock here only protects the connection itself.

Timestamps are stored as ISO-8601 UTC strings, which sort lexicographically in time order, so the
retention purge is a plain string comparison in SQL.

The schema is versioned with ``PRAGMA user_version``; opening an older file migrates it in place
(a new column is backfilled from the stored JSON), so an upgrade never strands a review queue.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

from .incidents import IncidentFilter, dedup_key
from .models import IncidentRecord, IncidentStatus

SCHEMA_VERSION = 2

_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS incidents (
    case_id            TEXT PRIMARY KEY,
    dedup_key          TEXT NOT NULL,
    status             TEXT NOT NULL,
    created_at         TEXT NOT NULL,
    retention_deadline TEXT NOT NULL,
    record             TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS incidents_dedup ON incidents (dedup_key);
CREATE INDEX IF NOT EXISTS incidents_status ON incidents (status);
CREATE INDEX IF NOT EXISTS incidents_retention ON incidents (retention_deadline);
CREATE TABLE IF NOT EXISTS audit (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id  TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    label    TEXT NOT NULL,
    at       TEXT NOT NULL,
    note     TEXT NOT NULL
);
"""

#: v2: tenant and provenance become filterable columns, backfilled from the stored record.
_MIGRATE_V2 = """
ALTER TABLE incidents ADD COLUMN tenant TEXT;
ALTER TABLE incidents ADD COLUMN provenance TEXT;
UPDATE incidents SET tenant = json_extract(record, '$.tenant'),
                     provenance = json_extract(record, '$.provenance');
CREATE INDEX IF NOT EXISTS incidents_tenant ON incidents (tenant, created_at);
"""


class SqliteIncidentRepository:
    """Incident records and the audit trail in one SQLite file (``:memory:`` for tests)."""

    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._migrate()

    @property
    def schema_version(self) -> int:
        with self._lock:
            return int(self._conn.execute("PRAGMA user_version").fetchone()[0])

    def _migrate(self) -> None:
        version = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
        if version < 1:
            self._conn.executescript(_SCHEMA_V1)
        if version < 2:
            columns = {row[1] for row in self._conn.execute("PRAGMA table_info(incidents)")}
            if "tenant" not in columns:
                self._conn.executescript("BEGIN;" + _MIGRATE_V2 + "COMMIT;")
        self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def get(self, case_id: str) -> IncidentRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT record FROM incidents WHERE case_id = ?", (case_id,)
            ).fetchone()
        return None if row is None else IncidentRecord.model_validate_json(row[0])

    def find(self, key: str) -> IncidentRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT record FROM incidents WHERE dedup_key = ? "
                "ORDER BY created_at DESC, case_id DESC LIMIT 1",
                (key,),
            ).fetchone()
        return None if row is None else IncidentRecord.model_validate_json(row[0])

    def put(self, record: IncidentRecord) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO incidents (case_id, dedup_key, status, created_at, "
                "retention_deadline, record, tenant, provenance) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(case_id) DO UPDATE SET status = excluded.status, "
                "retention_deadline = excluded.retention_deadline, record = excluded.record",
                (
                    record.case_id,
                    dedup_key(record.tenant, record.provenance, record.content_digest),
                    record.status.value,
                    _iso(record.created_at),
                    _iso(record.retention_deadline),
                    record.model_dump_json(),
                    record.tenant,
                    record.provenance.value,
                ),
            )

    def records(
        self,
        where: IncidentFilter | None = None,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[IncidentRecord]:
        clause, params = _where(where)
        query = f"SELECT record FROM incidents{clause} ORDER BY created_at, case_id"
        query += " LIMIT ? OFFSET ?"
        params.extend([-1 if limit is None else limit, offset])
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [IncidentRecord.model_validate_json(row[0]) for row in rows]

    def count(self, where: IncidentFilter | None = None) -> int:
        clause, params = _where(where)
        with self._lock:
            row = self._conn.execute(f"SELECT COUNT(*) FROM incidents{clause}", params).fetchone()
        return int(row[0])

    def pending_count(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM incidents WHERE status = ?",
                (IncidentStatus.PENDING.value,),
            ).fetchone()
        return int(row[0])

    def purge_expired(self, now: datetime) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM incidents WHERE retention_deadline <= ?", (_iso(now),)
            )
        return cursor.rowcount

    def append_audit(self, entry: dict[str, str]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO audit (case_id, reviewer, label, at, note) VALUES (?, ?, ?, ?, ?)",
                (entry["case_id"], entry["reviewer"], entry["label"], entry["at"], entry["note"]),
            )

    def audit_entries(self) -> list[dict[str, str]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT case_id, reviewer, label, at, note FROM audit ORDER BY id"
            ).fetchall()
        return [
            {"case_id": c, "reviewer": r, "label": lb, "at": a, "note": n}
            for c, r, lb, a, n in rows
        ]


def _where(where: IncidentFilter | None) -> tuple[str, list[str | int]]:
    """A parameterised WHERE clause for a filter (column names are fixed, never input)."""
    conditions: list[str] = []
    params: list[str | int] = []
    if where is not None:
        if where.status is not None:
            conditions.append("status = ?")
            params.append(where.status.value)
        if where.tenant is not None:
            conditions.append("tenant = ?")
            params.append(where.tenant)
        if where.provenance is not None:
            conditions.append("provenance = ?")
            params.append(where.provenance.value)
    return (" WHERE " + " AND ".join(conditions) if conditions else ""), params


def _iso(moment: datetime) -> str:
    """UTC ISO-8601 with a fixed width, so string order is time order."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
