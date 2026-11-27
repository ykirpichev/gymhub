"""Schema migrations for the pool server's database, SQLite and Postgres alike.

The schema is a list of numbered migrations, applied in order when the server starts. The table
`schema_version` records each one applied (version, name, time). Each migration runs in its own
transaction together with its row in `schema_version`, so a failed migration leaves nothing behind
(DDL is transactional in both SQLite and Postgres) and the next start tries it again. On Postgres a
transaction-scoped advisory lock serialises servers that start at the same time.

Migration 1 is the schema as it stood before migrations existed (M2 to M6). Its statements all use
`if not exists`, so a database that older code created upgrades in place: migration 1 finds every
table already there and is recorded as applied.

To change the schema, append a migration with the next version. Never edit one that has shipped:
databases that already applied it will not run it again. A migration is SQL (statements separated by
`;`, with {REAL}, {BIG} and {ID} for the types that differ between the dialects) and/or a function
of the DB for anything SQL alone cannot do portably (see has_column).

Servers from before migrations accept only meta.schema_version = '2'. It is kept at latest + 1, so
those servers still open a database at migration 1 (the same schema) and refuse anything newer.
A server refuses a database whose migrations are newer than it knows.
"""

import time
from dataclasses import dataclass
from typing import Callable

TYPES = {
    "sqlite": {"REAL": "real", "BIG": "integer", "ID": "integer primary key autoincrement"},
    "postgres": {"REAL": "double precision", "BIG": "bigint", "ID": "bigint generated always as identity primary key"},
}
PG_LOCK = 4_500_045  # pg_advisory_xact_lock key for migrations (any constant; 45 is the issue)
LEGACY_VERSION = "2"  # meta.schema_version written by servers from before migrations


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str = ""
    run: Callable | None = None  # run(db), after the SQL, in the same transaction


def statements(sql: str, dialect: str) -> list[str]:
    for key, value in TYPES[dialect].items():
        sql = sql.replace("{" + key + "}", value)
    return [part.strip() for part in sql.split(";") if part.strip()]


def tables(db) -> set[str]:
    if db.dialect == "postgres":
        rows = db.all("select table_name as name from information_schema.tables where table_schema = current_schema()")
    else:
        rows = db.all("select name from sqlite_master where type='table'")
    return {row["name"] for row in rows}


def has_column(db, table: str, column: str) -> bool:
    if db.dialect == "postgres":
        return bool(db.one("select 1 as x from information_schema.columns where table_schema = current_schema() "
                           "and table_name=? and column_name=?", table, column))
    return any(row["name"] == column for row in db.all(f"pragma table_info({table})"))


BASELINE = """
create table if not exists meta (key text primary key, value text not null);
create table if not exists pools (
  id text primary key, name text unique not null, settings text not null default '{}', created_at {REAL} not null);
create table if not exists members (
  id text primary key, name text not null, is_admin integer not null default 0, token_hash text unique,
  default_pool text, created_at {REAL} not null);
create table if not exists pool_members (
  pool text not null, member text not null, role text not null, quota text not null default '{}',
  created_at {REAL} not null, primary key (pool, member));
create table if not exists codes (
  code_hash text primary key, kind text not null, pool text not null, role text, created_by text, name text,
  created_at {REAL} not null, expires_at {REAL} not null, used_at {REAL});
create table if not exists machines (
  id text primary key, pool text not null, name text unique not null, owner text not null, token_hash text unique,
  labels text not null default '{}', profile text not null default '{}', policy text not null default '{}',
  free text, free_at {REAL}, public_ip text, created_at {REAL} not null, last_seen {REAL},
  drain integer not null default 0);
create table if not exists chunks (sha text primary key, size {BIG} not null, created_at {REAL} not null);
create table if not exists artifacts (
  id text primary key, kind text not null, size {BIG} not null, files integer not null, manifest text not null,
  created_at {REAL} not null);
create table if not exists artifact_owners (
  artifact text not null, member text not null, pool text not null, pinned integer not null default 0,
  created_at {REAL} not null, primary key (artifact, member));
create index if not exists artifact_owners_member on artifact_owners(member, pool);
create table if not exists artifact_chunks (artifact text not null, chunk text not null, primary key (artifact, chunk));
create index if not exists artifact_chunks_chunk on artifact_chunks(chunk);
create table if not exists chunk_proofs (chunk text not null, member text not null, primary key (chunk, member));
create table if not exists chunk_locations (
  chunk text not null, machine text not null, seen_at {REAL} not null, primary key (chunk, machine));
create index if not exists chunk_locations_machine on chunk_locations(machine);
create table if not exists models (
  pool text not null, name text not null, tag text not null, artifact text not null, meta text not null,
  created_by text not null, created_at {REAL} not null, primary key (pool, name, tag));
create table if not exists jobs (
  id text primary key, pool text not null, owner text not null, name text, priority integer not null default 0,
  spec text not null, status text not null, machine text, gpus text, lease_expires {REAL},
  tries integer not null default 0, preemptions integer not null default 0, exit_code integer, error text,
  outputs text, placement text, created_at {REAL} not null, started_at {REAL}, finished_at {REAL},
  log_bytes {BIG} not null default 0);
create index if not exists jobs_status on jobs(status, created_at);
create index if not exists jobs_owner on jobs(owner, created_at);
create index if not exists jobs_machine on jobs(machine, status);
create index if not exists jobs_finished on jobs(pool, finished_at);
create table if not exists gang_members (
  gang text not null, rank integer not null, job text not null, primary key (gang, rank));
create index if not exists gang_members_job on gang_members(job);
create table if not exists peer_links (
  src text not null, dst text not null, mbps {REAL} not null, rtt_ms {REAL} not null,
  measured_at {REAL} not null, primary key (src, dst));
create table if not exists logs (
  id {ID}, job text not null, key text, text text not null);
create unique index if not exists logs_key on logs(job, key);
create index if not exists logs_job on logs(job, id);
create table if not exists refs (
  pool text not null, name text not null, version integer not null, artifact text not null, parent integer not null,
  meta text not null default '{}', created_by text not null, created_at {REAL} not null, primary key (pool, name, version));
create index if not exists refs_artifact on refs(artifact);
create table if not exists queues (
  pool text not null, name text not null, owner text not null, ref text, max_lag integer,
  settings text not null default '{}', created_at {REAL} not null, primary key (pool, name));
create table if not exists tasks (
  id text primary key, pool text not null, queue text not null, task_key text, inputs text not null,
  requires_version integer, status text not null, tries integer not null default 0, stale integer not null default 0,
  duplicates integer not null default 0, created_at {REAL} not null, started_at {REAL}, finished_at {REAL},
  computed_with integer, machine text, seconds {REAL});
create unique index if not exists tasks_key on tasks(pool, queue, task_key);
create index if not exists tasks_queue on tasks(pool, queue, status, created_at);
create table if not exists task_leases (
  task text not null, job text not null, machine text not null, started_at {REAL} not null, expires {REAL} not null,
  primary key (task, job));
create index if not exists task_leases_job on task_leases(job);
create table if not exists task_results (
  seq {ID}, pool text not null, queue text not null, task text not null, result text not null,
  computed_with integer, machine text, created_at {REAL} not null);
create index if not exists task_results_queue on task_results(pool, queue, seq);
create table if not exists task_votes (
  task text not null, machine text not null, job text not null, digest text not null, result text not null,
  computed_with integer, created_at {REAL} not null, primary key (task, machine));
create table if not exists actors (
  pool text not null, name text not null, owner text not null, spec text not null, min_replicas integer not null,
  max_replicas integer not null, queue text, per_replica integer not null default 1, status text not null,
  note text, created_at {REAL} not null, primary key (pool, name));
create table if not exists actor_jobs (
  job text primary key, pool text not null, actor text not null, replica integer not null, created_at {REAL} not null);
create index if not exists actor_jobs_actor on actor_jobs(pool, actor);
create table if not exists job_checkpoints (
  job text primary key, artifact text not null, attempt integer, saved_at {REAL} not null);
create table if not exists rounds (
  id text primary key, pool text not null, ref text not null, base_version integer not null, owner text not null,
  quorum integer not null, min_contrib integer not null, deadline {REAL} not null, grace {REAL} not null,
  settings text not null default '{}', status text not null, opened_at {REAL} not null, quorum_at {REAL},
  ready_at {REAL}, closed_at {REAL}, reducer text, result_version integer, note text);
create index if not exists rounds_ref on rounds(pool, ref, status);
create table if not exists contributions (
  id text primary key, pool text not null, ref text not null, round text, job text not null, machine text,
  artifact text not null, weight {REAL} not null, base_version integer not null, late integer not null default 0,
  created_at {REAL} not null);
create index if not exists contributions_round on contributions(round);
create index if not exists contributions_ref on contributions(pool, ref);
create table if not exists reputation (
  pool text not null, machine text not null, checks integer not null default 0, failures integer not null default 0,
  excluded integer not null default 0, updated_at {REAL} not null, primary key (pool, machine));
create table if not exists verifications (
  id {ID}, pool text not null, machine text not null, kind text not null, ok integer not null, detail text,
  reporter text, created_at {REAL} not null);
create index if not exists verifications_machine on verifications(pool, machine, id);
"""

M7 = """
create table if not exists gossip_mix (
  round text not null, contribution text not null, peer text not null, weight {REAL} not null,
  primary key (round, contribution, peer));
create table if not exists gang_plans (
  gang text primary key, pool text not null, plan text not null, planned_at {REAL} not null);
"""

TREE = """
alter table rounds add column tree_plan text;
create table if not exists round_partials (
  round text not null, island integer not null, members text not null, machines text not null, planned text,
  status text not null, given text, reducer text, reducer_machine text, free_at {REAL} not null, claimed_at {REAL},
  artifact text, weight {REAL}, dropped text not null default '[]', forwarded integer not null default 0,
  done_at {REAL}, note text, primary key (round, island));
create table if not exists held_artifacts (
  artifact text not null, machine text not null, pool text not null, created_at {REAL} not null,
  primary key (artifact, machine));
"""

MIGRATIONS: list[Migration] = [
    Migration(1, "baseline: the schema of M2 to M6", BASELINE),
    Migration(2, "M7: gossip mixing weights (pool/gossip.py), gang plans (pool/planner.py)", M7),
    Migration(3, "#156: island-first reduce trees for rounds (round_partials), peer-held artifacts", TREE),
]

VERSION_TABLE = "create table if not exists schema_version (version integer primary key, name text not null, applied_at {REAL} not null)"


class MigrationError(SystemExit):
    """The database cannot be brought to this server's schema (a SystemExit: the server does not start)."""


def latest(migrations: list[Migration] | None = None) -> int:
    return max(m.version for m in (migrations or MIGRATIONS))


def applied(db) -> list[int]:
    if "schema_version" not in tables(db):
        return []
    return [row["version"] for row in db.all("select version from schema_version order by version")]


def _locked(db) -> None:
    """Inside a transaction: wait for other servers migrating the same Postgres database."""
    if db.dialect == "postgres":
        db.all("select pg_advisory_xact_lock(?) as locked", PG_LOCK)


def migrate(db, migrations: list[Migration] | None = None) -> list[int]:
    """Apply the pending migrations in order; returns the versions applied now."""
    migrations = sorted(migrations or MIGRATIONS, key=lambda m: m.version)
    versions = [m.version for m in migrations]
    if versions != list(range(1, len(versions) + 1)):
        raise ValueError(f"migrations must be numbered 1, 2, 3, ... without gaps: {versions}")
    newest = versions[-1]
    done: list[int] = []
    with db.lock:
        with db.transaction():
            _locked(db)
            have = applied(db)
            if have and max(have) > newest:
                raise MigrationError(f"this database is at schema {max(have)}; this server knows up to {newest}: "
                                     "upgrade the server")
            if not have and "meta" in tables(db):
                legacy = db.one("select value from meta where key='schema_version'")
                if legacy and legacy["value"] != LEGACY_VERSION:
                    raise MigrationError(f"this database has schema {legacy['value']}, which no migration starts from")
            for statement in statements(VERSION_TABLE, db.dialect):
                db.run(statement)
        for migration in migrations:
            with db.transaction():
                _locked(db)
                if db.one("select 1 as x from schema_version where version=?", migration.version):
                    continue  # applied before, or by another server that started at the same time
                for statement in statements(migration.sql, db.dialect):
                    db.run(statement)
                if migration.run:
                    migration.run(db)
                db.run("insert into schema_version (version, name, applied_at) values (?, ?, ?)",
                       migration.version, migration.name, time.time())
                db.run("insert into meta (key, value) values ('schema_version', ?) "
                       "on conflict (key) do update set value=excluded.value", str(migration.version + 1))
                done.append(migration.version)
    return done
