"""Document store shared by the board (Cloud Run) and the runner (your machine).

Two backends with the same small interface:
- FirestoreStore: production. The board uses its service account, the runner uses your
  application-default credentials. Nothing else can reach the database.
- SqliteStore: local development and tests. Board and runner share one file.

Collections (flat, so both backends stay trivial):
  gyms/{gym}                  title, description, langs, order, runner, updated_at
  problems/{gym}__{slug}      statement, hints, starters, current code, metadata
  progress/{gym}              the gym's progress.json
  jobs/{id}                   test | bench | save | reset requests and their output
  runners/{id}                heartbeat and GPU info

Retention: a document with `expire_at` (a datetime) is scratch. Firestore deletes it some time after
that moment through a TTL policy on the collections in EXPIRING (deploy/gcp.sh ttl sets them);
SqliteStore has no TTL, so `delete_expired` does the same when the job sweep runs.
"""

import json
import os
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable

Doc = dict


def now() -> float:
    return time.time()


def new_id() -> str:
    return time.strftime("%Y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:8]


def problem_id(gym: str, slug: str) -> str:
    return f"{gym}__{slug}"


# Collections whose documents may carry expire_at. deploy/gcp.sh ttl enables Firestore's TTL policy on
# each of them (and on their "staging-" twins); tests/test_retention.py checks the two lists agree.
EXPIRING = ("jobs", "usage", "attempts", "throttle", "registrations")


def expires_in(seconds: float) -> datetime:
    """A value for `expire_at`. Firestore's TTL only acts on timestamps, so this is a real datetime
    (UTC), never a float like the other times in the store."""
    return datetime.now(timezone.utc) + timedelta(seconds=seconds)


def _encode(value):
    if isinstance(value, datetime):
        return {"__datetime__": value.astimezone(timezone.utc).isoformat(timespec="microseconds")}
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _decode(obj: dict):
    if len(obj) == 1 and "__datetime__" in obj:
        return datetime.fromisoformat(obj["__datetime__"])
    return obj


def _dumps(data) -> str:
    """JSON for SqliteStore. Datetimes come back as datetimes, as Firestore timestamps do."""
    return json.dumps(data, default=_encode)


def _loads(text: str):
    return json.loads(text, object_hook=_decode)


class Store:
    def get(self, collection: str, doc_id: str) -> Doc | None:
        raise NotImplementedError

    def set(self, collection: str, doc_id: str, data: Doc, merge: bool = False) -> None:
        raise NotImplementedError

    def delete(self, collection: str, doc_id: str) -> None:
        raise NotImplementedError

    def where(self, collection: str, field: str, value) -> list[tuple[str, Doc]]:
        """Documents whose `field` equals `value`."""
        raise NotImplementedError

    def where_contains(self, collection: str, field: str, value) -> list[tuple[str, Doc]]:
        """Documents whose list `field` holds `value`."""
        raise NotImplementedError

    def recent(self, collection: str, order_field: str, limit: int) -> list[tuple[str, Doc]]:
        """Newest first by `order_field`."""
        raise NotImplementedError

    def claim(self, collection: str, doc_id: str, expect: Doc, update: Doc) -> bool:
        """Atomically apply `update` if every field in `expect` matches. Return True on success."""
        raise NotImplementedError

    def transact(self, collection: str, doc_id: str, fn: Callable[[Doc | None], Doc]) -> Doc:
        """Read a document (None if absent), write back what `fn` returns, all in one transaction.

        `fn` may run more than once (Firestore retries on contention), so it must not have side effects.
        Whatever it raises aborts the write and propagates. Returns the document that was written.
        """
        raise NotImplementedError

    def delete_expired(self, collection: str) -> int:
        """Delete documents whose expire_at has passed; returns how many. Firestore's TTL policy does
        this by itself, so only backends without one do anything here."""
        return 0

    def watch(self, collection: str, field: str, value, callback: Callable[[list[tuple[str, Doc]]], None]):
        """Call `callback` with the matching documents whenever they may have changed. Returns a stop function."""
        raise NotImplementedError

    def watch_doc(self, collection: str, doc_id: str, callback: Callable[[Doc | None], None]):
        """Call `callback` with the document each time it changes (and once at the start). Returns a stop function."""
        raise NotImplementedError


class SqliteStore(Store):
    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as db:
            db.execute("create table if not exists docs (coll text, id text, data text, primary key (coll, id))")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.execute("pragma journal_mode=wal")
        return db

    def get(self, collection, doc_id):
        with self._connect() as db:
            row = db.execute("select data from docs where coll=? and id=?", (collection, doc_id)).fetchone()
        return _loads(row[0]) if row else None

    def set(self, collection, doc_id, data, merge=False):
        with self._lock, self._connect() as db:
            db.execute("begin immediate")
            if merge:
                row = db.execute("select data from docs where coll=? and id=?", (collection, doc_id)).fetchone()
                if row:
                    data = {**_loads(row[0]), **data}
            db.execute(
                "insert or replace into docs (coll, id, data) values (?, ?, ?)",
                (collection, doc_id, _dumps(data)),
            )
            db.execute("commit")

    def delete(self, collection, doc_id):
        with self._connect() as db:
            db.execute("delete from docs where coll=? and id=?", (collection, doc_id))

    def delete_expired(self, collection):
        cutoff = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        with self._lock, self._connect() as db:
            cursor = db.execute(
                "delete from docs where coll=? and json_extract(data, '$.expire_at.__datetime__') < ?", (collection, cutoff)
            )
            return cursor.rowcount

    def _all(self, collection) -> Iterable[tuple[str, Doc]]:
        with self._connect() as db:
            rows = db.execute("select id, data from docs where coll=?", (collection,)).fetchall()
        return [(row[0], _loads(row[1])) for row in rows]

    def where(self, collection, field, value):
        return [(doc_id, doc) for doc_id, doc in self._all(collection) if doc.get(field) == value]

    def where_contains(self, collection, field, value):
        return [(doc_id, doc) for doc_id, doc in self._all(collection)
                if isinstance(doc.get(field), list) and value in doc[field]]

    def recent(self, collection, order_field, limit):
        docs = sorted(self._all(collection), key=lambda item: item[1].get(order_field, 0), reverse=True)
        return docs[:limit]

    def claim(self, collection, doc_id, expect, update):
        with self._lock, self._connect() as db:
            db.execute("begin immediate")
            row = db.execute("select data from docs where coll=? and id=?", (collection, doc_id)).fetchone()
            if not row:
                db.execute("rollback")
                return False
            doc = _loads(row[0])
            if any(doc.get(key) != value for key, value in expect.items()):
                db.execute("rollback")
                return False
            doc.update(update)
            db.execute("update docs set data=? where coll=? and id=?", (_dumps(doc), collection, doc_id))
            db.execute("commit")
            return True

    def transact(self, collection, doc_id, fn):
        with self._lock, self._connect() as db:
            db.execute("begin immediate")
            try:
                row = db.execute("select data from docs where coll=? and id=?", (collection, doc_id)).fetchone()
                new = fn(_loads(row[0]) if row else None)
                db.execute(
                    "insert or replace into docs (coll, id, data) values (?, ?, ?)", (collection, doc_id, _dumps(new))
                )
            except BaseException:
                db.execute("rollback")
                raise
            db.execute("commit")
            return new

    def watch(self, collection, field, value, callback):
        stop = threading.Event()

        def loop():
            while not stop.is_set():
                try:
                    callback(self.where(collection, field, value))
                except Exception as exc:  # keep polling
                    print(f"watch error: {exc}", flush=True)
                stop.wait(0.5)

        threading.Thread(target=loop, daemon=True).start()
        return stop.set


def _poll_doc(store: "Store", collection: str, doc_id: str, callback, interval: float = 0.5):
    stop = threading.Event()

    def loop():
        last = object()
        while not stop.is_set():
            try:
                doc = store.get(collection, doc_id)
                encoded = json.dumps(doc, sort_keys=True, default=str)
                if encoded != last:
                    last = encoded
                    callback(doc)
            except Exception as exc:  # keep watching
                print(f"watch_doc error: {exc}", flush=True)
            stop.wait(interval)

    threading.Thread(target=loop, daemon=True).start()
    return stop.set


SqliteStore.watch_doc = lambda self, collection, doc_id, callback: _poll_doc(self, collection, doc_id, callback)


class FirestoreStore(Store):
    def __init__(self, project: str | None = None, database: str = "(default)"):
        from google.cloud import firestore

        self._firestore = firestore
        self.client = firestore.Client(project=project, database=database)

    def _ref(self, collection, doc_id):
        return self.client.collection(collection).document(doc_id)

    def get(self, collection, doc_id):
        snapshot = self._ref(collection, doc_id).get()
        return snapshot.to_dict() if snapshot.exists else None

    def set(self, collection, doc_id, data, merge=False):
        self._ref(collection, doc_id).set(data, merge=merge)

    def delete(self, collection, doc_id):
        self._ref(collection, doc_id).delete()

    def where(self, collection, field, value):
        from google.cloud.firestore_v1.base_query import FieldFilter

        query = self.client.collection(collection).where(filter=FieldFilter(field, "==", value))
        return [(snap.id, snap.to_dict()) for snap in query.stream()]

    def where_contains(self, collection, field, value):
        from google.cloud.firestore_v1.base_query import FieldFilter

        query = self.client.collection(collection).where(filter=FieldFilter(field, "array_contains", value))
        return [(snap.id, snap.to_dict()) for snap in query.stream()]

    def recent(self, collection, order_field, limit):
        query = (
            self.client.collection(collection)
            .order_by(order_field, direction=self._firestore.Query.DESCENDING)
            .limit(limit)
        )
        return [(snap.id, snap.to_dict()) for snap in query.stream()]

    def claim(self, collection, doc_id, expect, update):
        ref = self._ref(collection, doc_id)
        transaction = self.client.transaction()

        @self._firestore.transactional
        def apply(txn) -> bool:
            snapshot = ref.get(transaction=txn)
            if not snapshot.exists:
                return False
            doc = snapshot.to_dict()
            if any(doc.get(key) != value for key, value in expect.items()):
                return False
            txn.update(ref, update)
            return True

        return apply(transaction)

    def transact(self, collection, doc_id, fn):
        ref = self._ref(collection, doc_id)

        @self._firestore.transactional
        def apply(txn):
            snapshot = ref.get(transaction=txn)
            new = fn(snapshot.to_dict() if snapshot.exists else None)
            txn.set(ref, new)
            return new

        return apply(self.client.transaction())

    def watch(self, collection, field, value, callback):
        from google.cloud.firestore_v1.base_query import FieldFilter

        query = self.client.collection(collection).where(filter=FieldFilter(field, "==", value))

        def on_snapshot(snapshots, _changes, _read_time):
            callback([(snap.id, snap.to_dict()) for snap in snapshots])

        watch = query.on_snapshot(on_snapshot)
        return watch.unsubscribe

    def watch_doc(self, collection, doc_id, callback):
        def on_snapshot(snapshots, _changes, _read_time):
            snapshot = snapshots[0] if snapshots else None
            callback(snapshot.to_dict() if snapshot is not None and snapshot.exists else None)

        watch = self._ref(collection, doc_id).on_snapshot(on_snapshot)
        return watch.unsubscribe


class PrefixedStore(Store):
    """Every collection name gets a prefix, so staging (``staging-jobs``, ...) shares the production
    database without seeing its data: Firestore's free tier covers only the default database."""

    def __init__(self, inner: Store, prefix: str):
        self.inner, self.prefix = inner, prefix

    def get(self, collection, doc_id):
        return self.inner.get(self.prefix + collection, doc_id)

    def set(self, collection, doc_id, data, merge=False):
        return self.inner.set(self.prefix + collection, doc_id, data, merge=merge)

    def delete(self, collection, doc_id):
        return self.inner.delete(self.prefix + collection, doc_id)

    def where(self, collection, field, value):
        return self.inner.where(self.prefix + collection, field, value)

    def where_contains(self, collection, field, value):
        return self.inner.where_contains(self.prefix + collection, field, value)

    def recent(self, collection, order_field, limit):
        return self.inner.recent(self.prefix + collection, order_field, limit)

    def claim(self, collection, doc_id, expect, update):
        return self.inner.claim(self.prefix + collection, doc_id, expect, update)

    def transact(self, collection, doc_id, fn):
        return self.inner.transact(self.prefix + collection, doc_id, fn)

    def delete_expired(self, collection):
        return self.inner.delete_expired(self.prefix + collection)

    def watch(self, collection, field, value, callback):
        return self.inner.watch(self.prefix + collection, field, value, callback)

    def watch_doc(self, collection, doc_id, callback):
        return self.inner.watch_doc(self.prefix + collection, doc_id, callback)


def open_store() -> Store:
    """GYMHUB_STORE=sqlite:/path/to/file.db or firestore[:project]. Default: firestore.
    GYMHUB_STORE_PREFIX (for example "staging-") is put in front of every collection name."""
    spec = os.environ.get("GYMHUB_STORE", "firestore")
    if spec.startswith("sqlite:"):
        store: Store = SqliteStore(spec.removeprefix("sqlite:"))
    elif spec.startswith("firestore"):
        project = spec.partition(":")[2] or os.environ.get("GOOGLE_CLOUD_PROJECT") or None
        store = FirestoreStore(project=project)
    else:
        raise SystemExit(f"Unknown GYMHUB_STORE {spec!r}")
    prefix = os.environ.get("GYMHUB_STORE_PREFIX", "")
    if prefix and not all(ch.isalnum() or ch in "-_" for ch in prefix):
        raise SystemExit(f"GYMHUB_STORE_PREFIX may only hold letters, digits, - and _, not {prefix!r}")
    return PrefixedStore(store, prefix) if prefix else store
