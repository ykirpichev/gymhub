BEGIN TRANSACTION;
CREATE TABLE actor_jobs (
  job text primary key, pool text not null, actor text not null, replica integer not null, created_at real not null);
CREATE TABLE actors (
  pool text not null, name text not null, owner text not null, spec text not null, min_replicas integer not null,
  max_replicas integer not null, queue text, per_replica integer not null default 1, status text not null,
  note text, created_at real not null, primary key (pool, name));
CREATE TABLE artifact_chunks (artifact text not null, chunk text not null, primary key (artifact, chunk));
CREATE TABLE artifact_owners (
  artifact text not null, member text not null, pool text not null, pinned integer not null default 0,
  created_at real not null, primary key (artifact, member));
CREATE TABLE artifacts (
  id text primary key, kind text not null, size integer not null, files integer not null, manifest text not null,
  created_at real not null);
CREATE TABLE chunk_locations (
  chunk text not null, machine text not null, seen_at real not null, primary key (chunk, machine));
CREATE TABLE chunk_proofs (chunk text not null, member text not null, primary key (chunk, member));
CREATE TABLE chunks (sha text primary key, size integer not null, created_at real not null);
CREATE TABLE codes (
  code_hash text primary key, kind text not null, pool text not null, role text, created_by text, name text,
  created_at real not null, expires_at real not null, used_at real);
INSERT INTO "codes" VALUES('493bbb1b12fd278bfab91a51a7b423ae5ff114f4c48dd95587439564ed6fd251','invite','pbd9ca101','owner','',NULL,1.79558155945991373e+09,1.79618635945991396e+09,1.79558155950484728e+09);
INSERT INTO "codes" VALUES('26b89b60922920055bcd3137942892e3ac9e7f6a5a9d0d69b8e3bfe5df0b51a2','machine','pbd9ca101',NULL,'m53194cab','box1',1.79558155950951337e+09,1.79618635950951361e+09,1.79558155951348495e+09);
CREATE TABLE contributions (
  id text primary key, pool text not null, ref text not null, round text, job text not null, machine text,
  artifact text not null, weight real not null, base_version integer not null, late integer not null default 0,
  created_at real not null);
CREATE TABLE gang_members (
  gang text not null, rank integer not null, job text not null, primary key (gang, rank));
CREATE TABLE job_checkpoints (
  job text primary key, artifact text not null, attempt integer, saved_at real not null);
CREATE TABLE jobs (
  id text primary key, pool text not null, owner text not null, name text, priority integer not null default 0,
  spec text not null, status text not null, machine text, gpus text, lease_expires real,
  tries integer not null default 0, preemptions integer not null default 0, exit_code integer, error text,
  outputs text, placement text, created_at real not null, started_at real, finished_at real,
  log_bytes integer not null default 0);
INSERT INTO "jobs" VALUES('j006ec587','pbd9ca101','m53194cab','job',0,'{"name": "job", "image": "python:3.12", "command": ["echo", "hi"], "resources": {"gpus": 0, "gpu_memory_gb": 0, "cpus": 1, "memory_gb": 4, "labels": {}}, "priority": 0, "code": null, "inputs": [], "env": {}, "timeout_minutes": 60, "retries": 1, "network": "none"}','queued',NULL,NULL,NULL,0,0,NULL,NULL,NULL,NULL,1.79558155951777076e+09,NULL,NULL,0);
CREATE TABLE logs (
  id integer primary key autoincrement, job text not null, key text, text text not null);
CREATE TABLE machines (
  id text primary key, pool text not null, name text unique not null, owner text not null, token_hash text unique,
  labels text not null default '{}', profile text not null default '{}', policy text not null default '{}',
  free text, free_at real, public_ip text, created_at real not null, last_seen real,
  drain integer not null default 0);
INSERT INTO "machines" VALUES('aee6d8c19','pbd9ca101','box1','m53194cab','86cf544438a151148a02a722bb05026ce39ca9f7d56559eeb6b7e4e643ee0287','{}','{}','{}',NULL,NULL,NULL,1.79558155951360559e+09,1.79558155951360583e+09,0);
CREATE TABLE members (
  id text primary key, name text not null, is_admin integer not null default 0, token_hash text unique,
  default_pool text, created_at real not null);
INSERT INTO "members" VALUES('m53194cab','ann',1,'88af8d3a6f3223460b8ea1827466cdc129285bbc114a84f0159f2a02d65b46aa','pbd9ca101',1.79558155950490093e+09);
CREATE TABLE meta (key text primary key, value text not null);
INSERT INTO "meta" VALUES('schema_version','2');
CREATE TABLE models (
  pool text not null, name text not null, tag text not null, artifact text not null, meta text not null,
  created_by text not null, created_at real not null, primary key (pool, name, tag));
CREATE TABLE peer_links (
  src text not null, dst text not null, mbps real not null, rtt_ms real not null,
  measured_at real not null, primary key (src, dst));
CREATE TABLE pool_members (
  pool text not null, member text not null, role text not null, quota text not null default '{}',
  created_at real not null, primary key (pool, member));
INSERT INTO "pool_members" VALUES('pbd9ca101','m53194cab','admin','{}',1.79558155950492048e+09);
CREATE TABLE pools (
  id text primary key, name text unique not null, settings text not null default '{}', created_at real not null);
INSERT INTO "pools" VALUES('pbd9ca101','default','{}',1.79558155945984697e+09);
CREATE TABLE queues (
  pool text not null, name text not null, owner text not null, ref text, max_lag integer,
  settings text not null default '{}', created_at real not null, primary key (pool, name));
CREATE TABLE refs (
  pool text not null, name text not null, version integer not null, artifact text not null, parent integer not null,
  meta text not null default '{}', created_by text not null, created_at real not null, primary key (pool, name, version));
CREATE TABLE reputation (
  pool text not null, machine text not null, checks integer not null default 0, failures integer not null default 0,
  excluded integer not null default 0, updated_at real not null, primary key (pool, machine));
CREATE TABLE rounds (
  id text primary key, pool text not null, ref text not null, base_version integer not null, owner text not null,
  quorum integer not null, min_contrib integer not null, deadline real not null, grace real not null,
  settings text not null default '{}', status text not null, opened_at real not null, quorum_at real,
  ready_at real, closed_at real, reducer text, result_version integer, note text);
CREATE TABLE task_leases (
  task text not null, job text not null, machine text not null, started_at real not null, expires real not null,
  primary key (task, job));
CREATE TABLE task_results (
  seq integer primary key autoincrement, pool text not null, queue text not null, task text not null, result text not null,
  computed_with integer, machine text, created_at real not null);
CREATE TABLE task_votes (
  task text not null, machine text not null, job text not null, digest text not null, result text not null,
  computed_with integer, created_at real not null, primary key (task, machine));
CREATE TABLE tasks (
  id text primary key, pool text not null, queue text not null, task_key text, inputs text not null,
  requires_version integer, status text not null, tries integer not null default 0, stale integer not null default 0,
  duplicates integer not null default 0, created_at real not null, started_at real, finished_at real,
  computed_with integer, machine text, seconds real);
CREATE TABLE verifications (
  id integer primary key autoincrement, pool text not null, machine text not null, kind text not null, ok integer not null, detail text,
  reporter text, created_at real not null);
CREATE INDEX artifact_owners_member on artifact_owners(member, pool);
CREATE INDEX artifact_chunks_chunk on artifact_chunks(chunk);
CREATE INDEX chunk_locations_machine on chunk_locations(machine);
CREATE INDEX jobs_status on jobs(status, created_at);
CREATE INDEX jobs_owner on jobs(owner, created_at);
CREATE INDEX jobs_machine on jobs(machine, status);
CREATE INDEX jobs_finished on jobs(pool, finished_at);
CREATE INDEX gang_members_job on gang_members(job);
CREATE UNIQUE INDEX logs_key on logs(job, key);
CREATE INDEX logs_job on logs(job, id);
CREATE INDEX refs_artifact on refs(artifact);
CREATE UNIQUE INDEX tasks_key on tasks(pool, queue, task_key);
CREATE INDEX tasks_queue on tasks(pool, queue, status, created_at);
CREATE INDEX task_leases_job on task_leases(job);
CREATE INDEX task_results_queue on task_results(pool, queue, seq);
CREATE INDEX actor_jobs_actor on actor_jobs(pool, actor);
CREATE INDEX rounds_ref on rounds(pool, ref, status);
CREATE INDEX contributions_round on contributions(round);
CREATE INDEX contributions_ref on contributions(pool, ref);
CREATE INDEX verifications_machine on verifications(pool, machine, id);
DELETE FROM "sqlite_sequence";
COMMIT;
