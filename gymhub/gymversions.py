"""Versioned gyms: upload a bundle, validate it on a worker, publish it.

    upload  the board unpacks the bundle and checks the manifest and every problem.json, but never
            imports the author's code. It stores the bundle, records version N as "validating" and
            queues a validation job.
    validate  a worker that may run the uploader's code fetches the bundle, and on the pool checks
            that every grader loads, every starter FAILS its tests and every reference solution
            (problems/<slug>/solution/) PASSES them (gymhub/runner/validate.py).
    publish  only a version that passed: its exercises replace the gym's live ones. A version that
            failed is kept with its report and never reaches learners.

    gym_versions/{gym}__v{NNNN}   gym, version, sha, size, status, created_by, created_at, job,
                                  exercises, errors, checks, warnings, finished_at
    gyms/{gym}                    ... source "bundle", version, bundle {sha, key, version}
    blobs: bundles/<gym>/<sha>.tar.gz and catalogs/<gym>/<sha>.json (the public catalog)

Learners only ever see the published version. Jobs carry the bundle they grade with, so a job
queued before a new version is published still runs against the version the learner saw.

Shared by the board (uploads, worker API) and runners in direct mode (completing jobs).
"""

import json
import re
import tempfile
from pathlib import Path

from gymhub import packages
from gymhub.blobs import bundle_key
from gymhub.store import Store, new_id, now, problem_id

STATUSES = {"validating", "published", "failed"}
VALIDATE_TIMEOUT = 1800
ENRICH_KEYS = {"case_template", "benchable", "examples"}  # what a worker may add to the board's own catalog
MAX_EXAMPLES = 64_000  # bytes of example cases per exercise (gymkit/examples.py stays under 40 000)


class UploadRejected(ValueError):
    def __init__(self, message: str, errors: list[str] | None = None):
        super().__init__(message)
        self.errors = errors or []


def version_id(gym_id: str, version: int) -> str:
    return f"{gym_id}__v{version:04d}"


def catalog_key(gym_id: str, sha: str) -> str:
    return f"catalogs/{gym_id}/{sha}.json"


def keys_key(gym_id: str, sha: str) -> str:
    """Answer keys of the non-code exercises: published to answer_keys/, never to problems/."""
    return f"keys/{gym_id}/{sha}.json"


def versions(store: Store, gym_id: str) -> list[dict]:
    docs = [{**doc, "id": doc_id} for doc_id, doc in store.where("gym_versions", "gym", gym_id)]
    return sorted(docs, key=lambda doc: doc.get("version", 0), reverse=True)


def upload(store: Store, blobs, data: bytes, uid: str, email: str, may_publish=None, extra: dict | None = None) -> dict:
    """Check and store a bundle, record its version and queue its validation. Returns the version doc.

    `may_publish(existing_gym_doc_or_None, gym_id)` returns why this user may not publish to that gym id, or None.
    `extra` is kept on the version (e.g. forked_from) and copied to the gym when it is first published."""
    with tempfile.TemporaryDirectory(prefix="gym-upload-") as tmp:
        try:
            package = packages.unpack(data, Path(tmp))
        except packages.BundleError as exc:
            raise UploadRejected(str(exc)) from exc
        errors = packages.validate_upload(package)
        if errors:
            raise UploadRejected(f"{len(errors)} problem(s) in the bundle", errors)
        gym_id = package.id
        catalog = packages.public_catalog(package)
        keys = packages.answer_keys(package)
    existing = store.get("gyms", gym_id)
    if existing is not None and existing.get("source") != "bundle":
        raise UploadRejected(f"{gym_id!r} is a gym published by a machine's local files; choose another id")
    if may_publish is not None:
        reason = may_publish(existing, gym_id)
        if reason:
            raise UploadRejected(reason)
    elif existing is not None and existing.get("owner_uid") not in (None, uid):
        raise UploadRejected(f"{gym_id!r} belongs to someone else")
    sha = packages.bundle_hash(data)
    history = versions(store, gym_id)
    for doc in history:
        if doc.get("sha") == sha and doc.get("status") in {"validating", "published"}:
            return {**doc, "duplicate": True}
    number = max([doc.get("version", 0) for doc in history] + [(existing or {}).get("version", 0)]) + 1
    blobs.put(bundle_key(gym_id, sha), data)
    blobs.put(catalog_key(gym_id, sha), json.dumps(catalog).encode())
    blobs.put(keys_key(gym_id, sha), json.dumps(keys).encode())
    doc_id = version_id(gym_id, number)
    job_id = new_id()
    bundle = {"sha": sha, "key": bundle_key(gym_id, sha), "version": number}
    requires = packages.requirements(package)
    version = {
        "gym": gym_id, "version": number, "sha": sha, "size": len(data), "status": "validating",
        "title": package.manifest.get("title"), "exercises": len(catalog["problems"]),
        "created_by": email, "uid": uid, "created_at": now(), "job": job_id,
        "errors": [], "checks": [], "warnings": [], **(extra or {}),
    }
    store.set("gym_versions", doc_id, version)
    store.set(
        "jobs", job_id,
        {
            "gym": gym_id, "slug": "", "problem": None, "lang": "", "action": "validate", "code": None,
            "options": {}, "status": "queued", "created_at": now(), "created_by": email, "uid": uid,
            "isolated": True, "requires": requires, "bundle": bundle, "version_doc": doc_id, "output": "",
        },
    )
    return {**version, "id": doc_id}


def _clean_checks(checks) -> list[dict]:
    clean = []
    for check in (checks if isinstance(checks, list) else [])[:2000]:
        if isinstance(check, dict):
            clean.append({
                "slug": str(check.get("slug", ""))[:80], "lang": str(check.get("lang", ""))[:20],
                "what": str(check.get("what", ""))[:60], "ok": bool(check.get("ok")),
                "detail": str(check.get("detail", ""))[:500],
            })
    return clean


def _missing_checks(blobs, version: dict, checks: list[dict]) -> list[tuple[str, str, str]]:
    """What the package requires (every exercise, starter and reference) that the report does not show passing.

    The report comes from a machine, so the board computes the required set from the stored bundle itself.
    """
    try:
        with tempfile.TemporaryDirectory(prefix="gym-finish-") as tmp:
            package = packages.unpack(blobs.get(bundle_key(version["gym"], version["sha"])), Path(tmp))
            required = packages.required_checks(package)
    except Exception as exc:  # no bundle to compare with: nothing can be published from it
        return [("", "", f"the stored bundle could not be read ({type(exc).__name__})")]
    passing = {(c["slug"], c["lang"], c["what"]) for c in checks if c["ok"]}
    return sorted(required - passing)


def fail(store: Store, job: dict, reason: str) -> None:
    """A validation job ended without a verdict (cancelled, or its machine kept disappearing): the version failed."""
    doc_id = job.get("version_doc")
    version = store.get("gym_versions", doc_id) if doc_id else None
    if version is not None and version.get("status") == "validating":
        store.set("gym_versions", doc_id, {"status": "failed", "errors": [reason], "finished_at": now()}, merge=True)


def finish(store: Store, blobs, job: dict, final: dict) -> None:
    """A validation job ended: publish its version if every check passed, else record why not."""
    doc_id = job.get("version_doc")
    version = store.get("gym_versions", doc_id) if doc_id else None
    if version is None or version.get("status") != "validating":
        return
    result = final.get("result") if isinstance(final.get("result"), dict) else {}
    checks = _clean_checks(result.get("checks"))
    warnings = [str(item)[:300] for item in (result.get("warnings") or [])][:50]
    failed = [check for check in checks if not check["ok"]]
    errors = [f"{c['slug']} [{c['lang']}]: {c['what']}: {c['detail']}".strip(": ") for c in failed][:100]
    missing = _missing_checks(blobs, version, checks) if not failed else []
    errors += [f"no result for {slug} [{lang}]: {what}" if lang else f"no result for {slug}: {what}"
               for slug, lang, what in missing[:20]]
    passed = bool(final.get("ok")) and bool(result.get("ok")) and bool(checks) and not failed and not missing
    if not passed and not errors:
        tail = (final.get("output") or "").strip().splitlines()[-5:]
        errors = ["validation did not finish: " + (" / ".join(tail) or final.get("status", "error"))]
    fields = {"checks": checks, "warnings": warnings, "errors": errors, "finished_at": now()}
    if not passed:
        store.set("gym_versions", doc_id, {**fields, "status": "failed"}, merge=True)
        return
    gym_id = version["gym"]
    live = store.get("gyms", gym_id) or {}
    if live.get("version", 0) > version["version"]:
        store.set("gym_versions", doc_id, {**fields, "status": "failed",
                                           "errors": [f"v{live['version']} was published first"]}, merge=True)
        return
    catalog = json.loads(blobs.get(catalog_key(gym_id, version["sha"])))
    enrich = result.get("enrich") if isinstance(result.get("enrich"), dict) else {}
    try:
        keys = json.loads(blobs.get(keys_key(gym_id, version["sha"])))
    except Exception:  # versions uploaded before non-code kinds existed have none
        keys = {}
    _publish(store, gym_id, version, catalog, enrich)
    _publish_keys(store, gym_id, version, catalog, keys)
    store.set("gym_versions", doc_id, {**fields, "status": "published"}, merge=True)


def _publish(store: Store, gym_id: str, version: dict, catalog: dict, enrich: dict) -> None:
    bundle = {"sha": version["sha"], "key": bundle_key(gym_id, version["sha"]), "version": version["version"]}
    slugs = set()
    for problem in catalog["problems"]:
        slug = problem["slug"]
        extra = enrich.get(slug) if isinstance(enrich.get(slug), dict) else {}
        extra = {key: value for key, value in extra.items() if key in ENRICH_KEYS}
        if "case_template" in extra and len(json.dumps(extra["case_template"])) > 20_000:
            extra.pop("case_template")
        if "examples" in extra and not (isinstance(extra["examples"], str) and len(extra["examples"]) <= MAX_EXAMPLES):
            extra.pop("examples")  # a JSON string from gymkit/examples.py; anything else is dropped
        slugs.add(slug)
        store.set("problems", problem_id(gym_id, slug),
                  {**problem, **extra, "gym": gym_id, "version": version["version"], "updated_at": now()})
    for doc_id, _doc in store.where("problems", "gym", gym_id):
        if doc_id.split("__", 1)[1] not in slugs:
            store.delete("problems", doc_id)
    live = store.get("gyms", gym_id)
    fields = {
        **catalog["gym"], "id": gym_id, "problems": len(slugs), "source": "bundle", "version": version["version"],
        "bundle": bundle, "published_at": now(), "updated_at": now(),
    }
    if live is None:  # first publish: the uploader owns it, and it starts private
        fields.update({"owner_uid": version.get("uid"), "owner_email": version.get("created_by"), "visibility": "private",
                       "editors": [], "learners": [], "tags": [], "allow_fork": False})
        if version.get("forked_from"):
            fields["forked_from"] = version["forked_from"]
    store.set("gyms", gym_id, fields, merge=True)  # keep the owner's settings across versions


def _publish_keys(store: Store, gym_id: str, version: dict, catalog: dict, keys: dict) -> None:
    """answer_keys/{gym}__{slug}: kind, key, version. Read only by the board's grader."""
    kinds = {problem["slug"]: problem.get("kind", "code") for problem in catalog["problems"]}
    for slug, kind in kinds.items():
        if kind != "code":
            store.set("answer_keys", problem_id(gym_id, slug),
                      {"gym": gym_id, "slug": slug, "kind": kind, "key": keys.get(slug) or {}, "version": version["version"]})
    for doc_id, doc in store.where("answer_keys", "gym", gym_id):
        if kinds.get(doc.get("slug"), "code") == "code":
            store.delete("answer_keys", doc_id)


def is_bundle_gym(gym_doc: dict | None) -> bool:
    return bool(gym_doc) and gym_doc.get("source") == "bundle"


_SAFE = re.compile(r"^[0-9a-f]{64}$")


def bundle_for_job(job: dict) -> dict | None:
    bundle = job.get("bundle")
    if isinstance(bundle, dict) and _SAFE.match(str(bundle.get("sha", ""))):
        return bundle
    return None
