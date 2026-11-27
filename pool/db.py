"""The server's state (#45 section 12). SQLite for one box, Postgres for multi-user pools.

    --db <data>/pool.db                    (the default: SQLite)
    --db postgresql://user:pw@host/db      (needs psycopg)

SQL is written once with `?` placeholders and portable types; the Postgres connection rewrites
the placeholders. Claims are conditional UPDATEs, so several server processes can share Postgres.

Tables:
  pools, pool_members           groups of members and machines; role admin | member | viewer, quotas
  members                       people; is_admin = server admin (may create pools)
  codes                         one-time invite and machine codes, each for one pool
  machines                      agents: pool, owner, labels, profile, use policy, last free report
  chunks, artifacts             64 MiB content-addressed chunks; artifacts are manifests of them
  artifact_owners               who may read an artifact (and whose storage quota it counts against)
  artifact_chunks               which chunks an artifact uses
  chunk_proofs                  chunks a member has uploaded the bytes of (dedupe without a leak)
  chunk_locations               which machine caches which chunk (cache affinity, M5 peer fetch)
  models                        named ModelRefs: pool, name, tag -> artifact plus architecture
  jobs, logs                    container runs and their output
  gang_members                  the ranks of a gang (each rank is a job)
  peer_links                    measured machine-to-machine links (islands)
  refs, queues, tasks,          M3 coordination: versioned refs, task queues and their leases and results
  task_leases, task_results     (pool/coord.py)
  rounds, contributions         M5 rounds (pool/rounds.py); round_partials: the island stages of a tree round
  held_artifacts                artifacts whose chunks stay on the machine that made them (tree rounds)

The schema is created and upgraded by numbered migrations when the server starts (pool/migrations.py):
to change it, append a migration.
"""

import json
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

FINAL = {"succeeded", "failed", "cancelled"}


def now() -> float:
    return time.time()


def new_id(prefix: str) -> str:
    return prefix + secrets.token_hex(4)


class DB:
    """One connection per thread; `?` placeholders; rows as dicts."""

    dialect = "sqlite"

    def __init__(self):
        self._local = threading.local()
        self.lock = threading.RLock()  # serialises read-then-write sections within this process

    def _connect(self):
        raise NotImplementedError

    def conn(self):
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
        return conn

    def _sql(self, sql: str) -> str:
        return sql

    def one(self, sql: str, *args) -> dict | None:
        rows = self.all(sql, *args)
        return rows[0] if rows else None

    def all(self, sql: str, *args) -> list[dict]:
        cursor = self.conn().execute(self._sql(sql), args)
        if cursor.description is None:
            return []
        names = [column[0] for column in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchall()]

    def run(self, sql: str, *args) -> int:
        return self.conn().execute(self._sql(sql), args).rowcount

    def many(self, sql: str, rows: list[tuple]) -> None:
        if rows:
            self.conn().cursor().executemany(self._sql(sql), rows)

    @contextmanager
    def transaction(self):
        raise NotImplementedError
        yield  # pragma: no cover

    def setup(self) -> None:
        from pool import migrations

        migrations.migrate(self)


class SqliteDB(DB):
    dialect = "sqlite"

    def __init__(self, path: str | Path):
        super().__init__()
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.path) as probe:  # an M1 database has jobs but no meta table
                tables = {row[0] for row in probe.execute("select name from sqlite_master where type='table'")}
            if "jobs" in tables and "meta" not in tables:
                raise SystemExit(f"{self.path} is an M1 database; M2 does not migrate it: use a new --data folder")
        self.setup()

    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None, check_same_thread=False)
        # Switching a new database to WAL needs an exclusive lock, and this pragma returns "database is
        # locked" at once instead of waiting out the busy timeout: retry while another server holds it.
        deadline = time.monotonic() + 30
        while True:
            try:
                conn.execute("pragma journal_mode=wal")
                break
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc) or time.monotonic() > deadline:
                    raise
                time.sleep(0.02)
        conn.execute("pragma synchronous=normal")
        return conn

    @contextmanager
    def transaction(self):
        with self.lock:
            conn = self.conn()
            conn.execute("begin immediate")
            try:
                yield
                conn.execute("commit")
            except BaseException:
                conn.execute("rollback")
                raise


class PostgresDB(DB):
    dialect = "postgres"

    def __init__(self, url: str):
        super().__init__()
        try:
            import psycopg  # noqa: F401
        except ImportError:
            raise SystemExit("a postgresql:// database needs psycopg: pip install 'psycopg[binary]'") from None
        self.url = url
        self.setup()

    def _connect(self):
        import psycopg

        return psycopg.connect(self.url, autocommit=True)

    def _sql(self, sql: str) -> str:
        return sql.replace("%", "%%").replace("?", "%s")

    @contextmanager
    def transaction(self):
        with self.lock:
            with self.conn().transaction():
                yield


def open_db(url: str | Path) -> DB:
    url = str(url)
    if url.startswith(("postgresql://", "postgres://")):
        return PostgresDB(url)
    return SqliteDB(url[len("sqlite://"):] if url.startswith("sqlite://") else url)


def load_json(value, default):
    return json.loads(value) if value else default


def load_job(row: dict | None) -> dict | None:
    if row is None:
        return None
    job = dict(row)
    job["spec"] = json.loads(job["spec"])
    job["gpus"] = load_json(job.get("gpus"), [])
    job["placement"] = load_json(job.get("placement"), None)
    return job


def load_machine(row: dict | None) -> dict | None:
    if row is None:
        return None
    machine = dict(row)
    for key, default in (("labels", {}), ("profile", {}), ("policy", {}), ("free", None)):
        machine[key] = load_json(machine.get(key), default)
    machine.pop("token_hash", None)
    return machine
