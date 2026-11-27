"""Where gym bundles are kept: a GCS bucket in the cloud, a local folder in development.

GYMHUB_BLOBS     gs://<bucket>[/<prefix>]  or  file:<dir>   (staging uses gs://<bucket>/staging)
                 Default: derived from GYMHUB_STORE. firestore:<project> uses gs://<project>-gym-bundles;
                 sqlite:<path> uses a "blobs" folder next to the database file.

Keys look like bundles/<gym>/<sha256>.tar.gz. Objects are immutable: a key is written once (and deleted only with its gym's owner's account).
GCS is reached through its JSON API with google-auth, so no extra client library is needed.
"""

import os
import urllib.parse
from pathlib import Path


class BlobMissing(KeyError):
    pass


class FileBlobs:
    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        if ".." in Path(key).parts or key.startswith("/"):
            raise ValueError(f"bad blob key {key!r}")
        return self.root / key

    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)

    def get(self, key: str) -> bytes:
        path = self._path(key)
        if not path.is_file():
            raise BlobMissing(key)
        return path.read_bytes()

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)


class GcsBlobs:
    API = "https://storage.googleapis.com"

    def __init__(self, bucket: str, prefix: str = ""):
        self.bucket = bucket
        self.prefix = prefix.strip("/") + "/" if prefix.strip("/") else ""
        self._session = None

    @property
    def session(self):
        if self._session is None:
            import google.auth
            from google.auth.transport.requests import AuthorizedSession

            credentials, _project = google.auth.default(scopes=["https://www.googleapis.com/auth/devstorage.read_write"])
            self._session = AuthorizedSession(credentials)
        return self._session

    def put(self, key: str, data: bytes) -> None:
        url = f"{self.API}/upload/storage/v1/b/{self.bucket}/o"
        reply = self.session.post(
            url, params={"uploadType": "media", "name": self.prefix + key}, data=data,
            headers={"Content-Type": "application/gzip"}, timeout=120,
        )
        if reply.status_code >= 300:
            raise RuntimeError(f"GCS upload of {key} failed: HTTP {reply.status_code} {reply.text[:300]}")

    def get(self, key: str) -> bytes:
        url = f"{self.API}/storage/v1/b/{self.bucket}/o/{urllib.parse.quote(self.prefix + key, safe='')}"
        reply = self.session.get(url, params={"alt": "media"}, timeout=120)
        if reply.status_code == 404:
            raise BlobMissing(key)
        if reply.status_code >= 300:
            raise RuntimeError(f"GCS download of {key} failed: HTTP {reply.status_code} {reply.text[:300]}")
        return reply.content

    def delete(self, key: str) -> None:
        url = f"{self.API}/storage/v1/b/{self.bucket}/o/{urllib.parse.quote(self.prefix + key, safe='')}"
        reply = self.session.delete(url, timeout=60)
        if reply.status_code >= 300 and reply.status_code != 404:
            raise RuntimeError(f"GCS delete of {key} failed: HTTP {reply.status_code} {reply.text[:300]}")


_opened: dict[str, object] = {}


def open_blobs(spec: str | None = None, store_spec: str | None = None):
    spec = spec or os.environ.get("GYMHUB_BLOBS")
    if not spec:
        store_spec = store_spec or os.environ.get("GYMHUB_STORE", "firestore")
        if store_spec.startswith("firestore"):
            project = store_spec.partition(":")[2] or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
            if not project:
                raise SystemExit("Set GYMHUB_BLOBS=gs://<bucket> (no project to derive the bucket name from)")
            spec = f"gs://{project}-gym-bundles"
            if os.environ.get("GYMHUB_STORE_PREFIX"):
                spec += "/" + os.environ["GYMHUB_STORE_PREFIX"].strip("-_")
        else:
            spec = "file:" + str(Path(store_spec.partition(":")[2] or ".cache/gymhub/board.db").parent / "blobs")
    if spec not in _opened:
        if spec.startswith("gs://"):
            bucket, _, prefix = spec[5:].strip("/").partition("/")
            _opened[spec] = GcsBlobs(bucket, prefix)  # one authorized session per process
        elif spec.startswith("file:"):
            _opened[spec] = FileBlobs(Path(spec[5:]))
    if spec in _opened:
        return _opened[spec]
    raise SystemExit(f"GYMHUB_BLOBS must be gs://<bucket> or file:<dir>, not {spec!r}")


def bundle_key(gym_id: str, sha: str) -> str:
    return f"bundles/{gym_id}/{sha}.tar.gz"
