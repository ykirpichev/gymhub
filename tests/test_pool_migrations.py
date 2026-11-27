"""Schema migrations (pool/migrations.py) and Prometheus metrics (pool/metrics.py) for the pool server (#45).

Database tests run on SQLite, or on a fresh Postgres database when POOL_TEST_POSTGRES is set.
pool_db_before_migrations.sql is a dump of a database made by the server from before migrations
(origin/main at 57fd71f): the bootstrap, a member, a machine and a queued job.
"""

import json
import re
import sqlite3
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pool import metrics, migrations
from pool.db import now, open_db
from pool.migrations import Migration

ALL = [m.version for m in migrations.MIGRATIONS]
N = migrations.latest()  # the newest migration this server ships
from pool.server import create_app

from tests.test_pool import PG, db_url  # noqa: F401  (fixture)

DUMP = Path(__file__).with_name("pool_db_before_migrations.sql")


def versions(db) -> list[int]:
    return [row["version"] for row in db.all("select version from schema_version order by version")]


def meta_version(db) -> str:
    return db.one("select value from meta where key='schema_version'")["value"]


class Raw:
    """Just enough of a DB to lay down the schema the way servers from before migrations did."""

    def __init__(self, url: str):
        self.dialect = "postgres" if url.startswith("postgres") else "sqlite"
        if self.dialect == "postgres":
            import psycopg

            self.conn = psycopg.connect(url, autocommit=True)
        else:
            self.conn = sqlite3.connect(url, isolation_level=None)

    def run(self, sql: str, *args):
        if self.dialect == "postgres":
            sql = sql.replace("?", "%s")
        self.conn.execute(sql, args)

    def close(self):
        self.conn.close()


def legacy_database(url: str) -> None:
    """What DB.setup did before migrations: every `create ... if not exists`, then meta.schema_version = 2."""
    raw = Raw(url)
    for statement in migrations.statements(migrations.BASELINE, raw.dialect):
        raw.run(statement)
    raw.run("insert into meta (key, value) values ('schema_version', '2')")
    raw.run("insert into members (id, name, is_admin, token_hash, created_at) values ('m1', 'ann', 1, 'h', ?)", now())
    raw.close()


# ---- migrations ---------------------------------------------------------------------------------------


def test_a_new_database_gets_every_migration(db_url):
    db = open_db(db_url)
    assert versions(db) == [m.version for m in migrations.MIGRATIONS]
    assert meta_version(db) == str(migrations.latest() + 1)
    assert {"jobs", "machines", "tasks", "rounds", "reputation", "schema_version"} <= migrations.tables(db)
    assert migrations.migrate(db) == [], "a second start applies nothing"


def test_a_database_from_before_migrations_upgrades_in_place(db_url):
    legacy_database(db_url)
    db = open_db(db_url)
    assert versions(db) == ALL
    assert meta_version(db) == str(N + 1)
    assert db.one("select name from members where id='m1'")["name"] == "ann"


def test_the_dump_of_an_older_server_upgrades_and_serves(tmp_path):
    data = tmp_path / "server"
    data.mkdir()
    with sqlite3.connect(data / "pool.db") as conn:
        conn.executescript(DUMP.read_text())
        before = {name: conn.execute(f"select count(*) from {name}").fetchone()[0]
                  for (name,) in conn.execute("select name from sqlite_master where type='table'").fetchall()}
    assert "schema_version" not in before
    app = create_app(data, gc_seconds=None)
    db = app.state.db
    assert versions(db) == ALL
    assert app.state.bootstrap_code is None, "its admin is still there"
    for name, count in before.items():
        assert db.one(f"select count(*) as n from {name}")["n"] == count, name
    assert db.one("select status from jobs")["status"] == "queued"


def test_new_migrations_apply_in_order_once(db_url):
    db = open_db(db_url)
    seen = []

    def backfill(db):
        seen.append(migrations.has_column(db, "jobs", "note"))
        db.run("update jobs set note='old'")

    extra = migrations.MIGRATIONS + [
        Migration(N + 1, "jobs.note", "alter table jobs add column note text"),
        Migration(N + 2, "backfill notes", run=backfill),
    ]
    db.run("insert into jobs (id, pool, owner, spec, status, created_at) values ('j1', 'p', 'm', '{}', 'queued', ?)", now())
    assert migrations.migrate(db, extra) == [N + 1, N + 2]
    assert seen == [True], "the backfill runs after the column exists"
    assert versions(db) == ALL + [N + 1, N + 2] and meta_version(db) == str(N + 3)
    assert db.one("select note from jobs where id='j1'")["note"] == "old"
    assert migrations.migrate(db, extra) == []
    with pytest.raises(SystemExit, match=f"schema {N + 2}; this server knows up to {N}"):
        open_db(db_url)  # a server that predates migrations 2 and 3


def test_a_failed_migration_leaves_nothing_behind(db_url):
    db = open_db(db_url)

    def boom(db):
        raise RuntimeError("boom")

    extra = migrations.MIGRATIONS + [Migration(N + 1, "half done", "create table half (x integer); alter table jobs add column y text", boom)]
    with pytest.raises(RuntimeError):
        migrations.migrate(db, extra)
    assert versions(db) == ALL and meta_version(db) == str(N + 1)
    assert "half" not in migrations.tables(db) and not migrations.has_column(db, "jobs", "y")
    fixed = migrations.MIGRATIONS + [Migration(N + 1, "done right", "create table half (x integer)")]
    assert migrations.migrate(db, fixed) == [N + 1]


def test_servers_starting_together_migrate_once(db_url):
    errors, dbs = [], []

    def start():
        try:
            dbs.append(open_db(db_url))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=start) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors and len(dbs) == 4
    assert versions(dbs[0]) == ALL


def test_unknown_databases_are_refused(tmp_path):
    path = str(tmp_path / "odd.db")
    raw = Raw(path)
    raw.run("create table meta (key text primary key, value text not null)")
    raw.run("insert into meta values ('schema_version', '7')")
    raw.close()
    with pytest.raises(SystemExit, match="schema 7"):
        open_db(path)
    with pytest.raises(ValueError):
        migrations.migrate(open_db(str(tmp_path / "ok.db")), migrations.MIGRATIONS + [Migration(N + 2, "gap")])


# ---- metrics ------------------------------------------------------------------------------------------

SAMPLE = re.compile(r'^[a-z0-9_]+(\{([a-z_]+="([^"\\]|\\.)*",?)*\})? -?[0-9.e+-]+$')


def samples(text: str) -> dict[str, float]:
    out = {}
    for line in text.splitlines():
        if line.startswith("#"):
            assert re.match(r"^# (HELP|TYPE) [a-z0-9_]+ .+$", line), line
            continue
        assert SAMPLE.match(line), line
        key, value = line.rsplit(" ", 1)
        out[key] = float(value)
    return out


def test_metrics_need_an_admin_or_the_scrape_token(tmp_path):
    app = create_app(tmp_path / "server", gc_seconds=None, metrics_token="scrape-secret")
    client = TestClient(app)
    admin = client.post("/v1/login", json={"code": app.state.bootstrap_code, "name": "ann"}).json()["token"]
    code = client.post("/v1/invites", json={"role": "admin"}, headers={"Authorization": f"Bearer {admin}"}).json()["code"]
    pool_admin = client.post("/v1/login", json={"code": code, "name": "bob"}).json()["token"]
    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/metrics", headers={"Authorization": f"Bearer {pool_admin}"}).status_code == 403, "pool admins see one pool only"
    for token in (admin, "scrape-secret"):
        reply = client.get("/metrics", headers={"Authorization": f"Bearer {token}"})
        assert reply.status_code == 200 and reply.headers["content-type"].startswith("text/plain; version=0.0.4")
        found = samples(reply.text)
        assert found["pool_schema_version"] == migrations.latest()
        assert found['pool_jobs{pool="default",state="queued"}'] == 0


def test_metrics_count_jobs_machines_gpu_hours_and_bytes(db_url):
    db = open_db(db_url)
    t = now()
    db.run("insert into pools (id, name, settings, created_at) values ('p1', 'lab', '{}', ?)", t)
    gpus = json.dumps({"gpus": [{"index": 0}, {"index": 1}]})
    db.run("insert into machines (id, pool, name, owner, profile, created_at, last_seen) values ('a1', 'p1', 'spark', 'm', ?, ?, ?)", gpus, t, t - 5)
    db.run("insert into machines (id, pool, name, owner, created_at, last_seen, drain) values ('a2', 'p1', 'old\"box', 'm', ?, ?, 0)", t, t - 3600)
    spec = json.dumps({"resources": {"gpus": 2, "cpus": 4}})
    rows = [("j1", "succeeded", None, t - 7200, t - 3600), ("j2", "running", "a1", t - 1800, None),
            ("j3", "queued", None, None, None), ("j4", "queued", None, None, None)]
    for job, status, machine, started, finished in rows:
        db.run("insert into jobs (id, pool, owner, spec, status, machine, gpus, created_at, started_at, finished_at) "
               "values (?, 'p1', 'm', ?, ?, ?, ?, ?, ?, ?)", job, spec, status, machine, "[0, 1]" if machine else None,
               t - 600, started, finished)
    db.run("insert into queues (pool, name, owner, created_at) values ('p1', 'prompts', 'm', ?)", t)
    db.run("insert into tasks (id, pool, queue, inputs, status, created_at) values ('t1', 'p1', 'prompts', '{}', 'queued', ?)", t)
    db.run("insert into artifacts (id, kind, size, files, manifest, created_at) values ('sha256:x', 'data', 3000000000, 1, '{}', ?)", t)
    for member in ("m", "n"):
        db.run("insert into artifact_owners (artifact, member, pool, created_at) values ('sha256:x', ?, 'p1', ?)", member, t)
    db.run("insert into chunks (sha, size, created_at) values ('c1', 3000000000, ?)", t)

    found = samples(metrics.render(db, "9.9", 1))
    assert found['pool_server_info{version="9.9"}'] == 1
    assert found['pool_jobs{pool="lab",state="queued"}'] == 2 and found['pool_jobs{pool="lab",state="running"}'] == 1
    assert found['pool_queue_depth{pool="lab"}'] == 2
    assert 590 < found['pool_queue_oldest_seconds{pool="lab"}'] < 700
    assert found['pool_tasks{pool="lab",queue="prompts",state="queued"}'] == 1
    assert found['pool_machines{pool="lab",state="online"}'] == 1 and found['pool_machines{pool="lab",state="offline"}'] == 1
    assert found['pool_machines_online{pool="lab"}'] == 1
    assert found['pool_gpus_online{pool="lab"}'] == 2 and found['pool_gpus_busy{pool="lab"}'] == 2
    assert 4 < found['pool_machine_heartbeat_age_seconds{pool="lab",machine="spark"}'] < 60
    assert found['pool_machine_heartbeat_age_seconds{pool="lab",machine="old\\"box"}'] >= 3600
    assert found['pool_gpu_hours_total{pool="lab"}'] == pytest.approx(3.0, abs=0.01)  # 2 GPUs x (1 h + 0.5 h)
    assert found['pool_gpu_hours_24h{pool="lab"}'] == pytest.approx(3.0, abs=0.01)
    assert found['pool_artifacts{pool="lab"}'] == 1 and found['pool_artifact_bytes{pool="lab"}'] == 3_000_000_000
    assert found["pool_chunk_bytes"] == 3_000_000_000 and found["pool_chunks"] == 1
