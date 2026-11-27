"""Artifacts: immutable, content-addressed folders (#45 section 5).

An artifact is a manifest: every file with its size, mode and the sha256 of each 64 MiB chunk
(small files are inline). Its id is "sha256:" plus the hash of the canonical manifest JSON.
Identical chunks are stored and sent once, so a second upload of a model costs nothing, and a
new checkpoint reuses the chunks of files it did not change.

    manifest = {"v": 1, "dirs": ["empty"],
                "files": [{"path": "config.json", "size": 812, "mode": 420, "data": "<base64>"},
                          {"path": "model.safetensors", "size": 4e9, "mode": 420, "chunks": ["ab12...", ...]}]}

Used by the CLI (build_manifest), the server (ChunkStore, validate) and the agent (ChunkStore,
materialize); standard library only.
"""

import base64
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath

CHUNK = 64 * 1024 * 1024
INLINE_MAX = 64 * 1024
INLINE_TOTAL = 8 * 1024 * 1024
MANIFEST_MAX = 32 * 1024 * 1024
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache", ".pytest_cache", ".ipynb_checkpoints"}
HEX = set("0123456789abcdef")


class ArtifactError(ValueError):
    pass


def canonical(manifest: dict) -> bytes:
    return json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()


def manifest_id(manifest: dict) -> str:
    return "sha256:" + hashlib.sha256(canonical(manifest)).hexdigest()


def is_hex(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX


def _safe_path(path: str) -> PurePosixPath:
    if not isinstance(path, str) or not path or "\\" in path or "\0" in path:
        raise ArtifactError(f"unsafe path {path!r}")
    pure = PurePosixPath(path)
    if pure.is_absolute() or any(part in ("..", ".", "") for part in path.split("/")):
        raise ArtifactError(f"unsafe path {path!r}")
    return pure


def chunk_sizes(file: dict) -> list[int]:
    size, count = file["size"], len(file.get("chunks", []))
    return [min(CHUNK, size - index * CHUNK) for index in range(count)]


def chunks_of(manifest: dict) -> dict[str, int]:
    """Every distinct chunk of an artifact, with its size."""
    out: dict[str, int] = {}
    for file in manifest["files"]:
        for sha, size in zip(file.get("chunks", []), chunk_sizes(file)):
            out[sha] = size
    return out


def validate(manifest: dict) -> dict:
    """Check a manifest from a client: safe unique paths, sizes that match the chunk lists."""
    if not isinstance(manifest, dict) or manifest.get("v") != 1 or not isinstance(manifest.get("files"), list):
        raise ArtifactError("not a v1 manifest")
    if set(manifest) - {"v", "files", "dirs"}:
        raise ArtifactError("unknown manifest fields")
    if len(canonical(manifest)) > MANIFEST_MAX:
        raise ArtifactError("manifest too large")
    seen: set[str] = set()
    inline = 0
    for file in manifest["files"]:
        if not isinstance(file, dict) or set(file) - {"path", "size", "mode", "chunks", "data"}:
            raise ArtifactError("bad file entry")
        path = str(_safe_path(file.get("path", "")))
        if path in seen:
            raise ArtifactError(f"{path} listed twice")
        seen.add(path)
        size = file.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0 or file.get("mode") not in (0o644, 0o755):
            raise ArtifactError(f"{path}: bad size or mode")
        if "data" in file:
            try:
                data = base64.b64decode(file["data"], validate=True)
            except (ValueError, TypeError):
                raise ArtifactError(f"{path}: bad inline data") from None
            if len(data) != size or "chunks" in file:
                raise ArtifactError(f"{path}: inline data does not match its size")
            inline += size
        else:
            chunks = file.get("chunks")
            if not isinstance(chunks, list) or len(chunks) != -(-size // CHUNK) or not all(is_hex(c) for c in chunks):
                raise ArtifactError(f"{path}: chunk list does not match its size")
    if inline > INLINE_TOTAL:
        raise ArtifactError("too much inline data")
    dirs = manifest.get("dirs", [])
    if not isinstance(dirs, list):
        raise ArtifactError("bad dirs")
    for path in dirs:
        _safe_path(path)
    return manifest


def total_size(manifest: dict) -> int:
    return sum(file["size"] for file in manifest["files"])


# ---- building a manifest from a folder (client side) ---------------------------------------------


class HashCache:
    """Chunk hashes of files already hashed, keyed by path, size and mtime, so sending a 35 GB model
    folder a second time does not read it again."""

    def __init__(self, path: Path | None):
        self.path = path
        self.data: dict = {}
        if path and path.is_file():
            try:
                self.data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                self.data = {}
        self.dirty = False

    def key(self, path: Path) -> str:
        stat = path.stat()
        return f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"

    def get(self, path: Path):
        return self.data.get(self.key(path))

    def put(self, path: Path, chunks: list[str]) -> None:
        self.data[self.key(path)] = chunks
        self.dirty = True

    def save(self) -> None:
        if self.path and self.dirty:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data))
            tmp.replace(self.path)


def _hash_chunks(path: Path) -> list[str]:
    out = []
    with open(path, "rb") as handle:
        while True:
            block = handle.read(CHUNK)
            if not block:
                break
            out.append(hashlib.sha256(block).hexdigest())
    return out


def build_manifest(folder: Path, follow_links: bool = False, cache: HashCache | None = None,
                   max_bytes: int | None = None, skip_dirs: set[str] = SKIP_DIRS) -> tuple[dict, dict[str, tuple[Path, int, int]]]:
    """The manifest of `folder`, and where to read each chunk: {sha: (path, offset, length)}.

    follow_links=False (the agent packing job outputs): links are skipped, never followed, so a job
    cannot make the agent upload a host file. follow_links=True (your own folders, such as the
    Hugging Face cache, whose snapshots link into ../blobs): links are read through."""
    folder = Path(folder)
    if not folder.is_dir():
        raise ArtifactError(f"{folder} is not a folder")
    files, dirs, sources = [], [], {}
    inline_total = total = 0
    for root, subdirs, names in os.walk(folder, followlinks=follow_links):
        root_path = Path(root)
        subdirs[:] = sorted(d for d in subdirs if d not in skip_dirs and (follow_links or not (root_path / d).is_symlink()))
        relative_root = root_path.relative_to(folder).as_posix()
        if relative_root != "." and not names and not subdirs:
            dirs.append(relative_root)
        for name in sorted(names):
            path = root_path / name
            if path.is_symlink() and not follow_links:
                continue
            if not path.is_file():
                continue
            relative = path.relative_to(folder).as_posix()
            size = path.stat().st_size
            total += size
            if max_bytes is not None and total > max_bytes:
                raise ArtifactError(f"{folder} holds more than {max_bytes // 1024**2} MB")
            mode = 0o755 if os.access(path, os.X_OK) else 0o644
            if size <= INLINE_MAX and inline_total + size <= INLINE_TOTAL:
                inline_total += size
                files.append({"path": relative, "size": size, "mode": mode, "data": base64.b64encode(path.read_bytes()).decode()})
                continue
            chunks = cache.get(path) if cache else None
            if chunks is None or len(chunks) != -(-size // CHUNK):
                chunks = _hash_chunks(path)
                if cache:
                    cache.put(path, chunks)
            for index, sha in enumerate(chunks):
                sources.setdefault(sha, (path, index * CHUNK, min(CHUNK, size - index * CHUNK)))
            files.append({"path": relative, "size": size, "mode": mode, "chunks": chunks})
    if cache:
        cache.save()
    manifest = {"v": 1, "files": files, **({"dirs": dirs} if dirs else {})}
    return manifest, sources


def read_chunk(source: tuple[Path, int, int]) -> bytes:
    path, offset, length = source
    with open(path, "rb") as handle:
        handle.seek(offset)
        data = handle.read(length)
    if len(data) != length:
        raise ArtifactError(f"{path} changed while uploading")
    return data


# ---- storing chunks (server and agent) -----------------------------------------------------------


class ChunkStore:
    """Chunks as files: root/<aa>/<sha>. A write is hashed while it streams and kept only if it
    matches its name."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, sha: str) -> Path:
        if not is_hex(sha):
            raise ArtifactError(f"bad chunk id {sha!r}")
        return self.root / sha[:2] / sha

    def has(self, sha: str) -> bool:
        return self.path(sha).is_file()

    def writer(self, sha: str) -> "ChunkWriter":
        return ChunkWriter(self.path(sha), sha)

    def put(self, sha: str, data: bytes) -> None:
        writer = self.writer(sha)
        try:
            writer.write(data, CHUNK)
            writer.commit()
        except BaseException:
            writer.abort()
            raise

    def delete(self, sha: str) -> int:
        path = self.path(sha)
        try:
            size = path.stat().st_size
            path.unlink()
            return size
        except FileNotFoundError:
            return 0


class ChunkWriter:
    def __init__(self, target: Path, sha: str):
        self.target, self.sha_expected = target, sha
        target.parent.mkdir(parents=True, exist_ok=True)
        self.handle = tempfile.NamedTemporaryFile(dir=target.parent, delete=False, suffix=".part")
        self.sha = hashlib.sha256()
        self.size = 0

    def write(self, data: bytes, limit: int = CHUNK) -> None:
        self.size += len(data)
        if self.size > limit:
            raise ArtifactError("chunk larger than 64 MiB")
        self.sha.update(data)
        self.handle.write(data)

    def commit(self) -> int:
        self.handle.close()
        if self.sha.hexdigest() != self.sha_expected:
            os.unlink(self.handle.name)
            raise ArtifactError("the uploaded bytes do not match the chunk id")
        os.chmod(self.handle.name, 0o444)
        os.replace(self.handle.name, self.target)
        return self.size

    def abort(self) -> None:
        if not self.handle.closed:
            self.handle.close()
        try:
            os.unlink(self.handle.name)
        except FileNotFoundError:
            pass


def materialize(manifest: dict, chunks: ChunkStore, dest: Path) -> None:
    """Write an artifact's files into `dest` from stored chunks. A file of exactly one chunk is a
    hard link to it (no second copy of the bytes); larger files are assembled."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for directory in manifest.get("dirs", []):
        dest.joinpath(*_safe_path(directory).parts).mkdir(parents=True, exist_ok=True)
    for file in manifest["files"]:
        target = dest.joinpath(*_safe_path(file["path"]).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        if "data" in file:
            target.write_bytes(base64.b64decode(file["data"]))
            os.chmod(target, file["mode"])
            continue
        parts = file["chunks"]
        if len(parts) == 1 and file["mode"] == 0o644:
            try:
                os.link(chunks.path(parts[0]), target)
                continue
            except OSError:
                pass  # another filesystem: copy
        with open(target, "wb") as out:
            for sha in parts:
                with open(chunks.path(sha), "rb") as handle:
                    shutil.copyfileobj(handle, out, 1 << 20)
        os.chmod(target, file["mode"])
        if target.stat().st_size != file["size"]:
            raise ArtifactError(f"{file['path']}: assembled size does not match")


def manifest_of_bytes(files: dict[str, bytes]) -> tuple[dict, dict[str, bytes]]:
    """A manifest from in-memory files (tests): {path: bytes} -> (manifest, {sha: chunk bytes})."""
    with tempfile.TemporaryDirectory() as tmp:
        for name, data in files.items():
            path = Path(tmp) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        manifest, sources = build_manifest(Path(tmp))
        return manifest, {sha: read_chunk(source) for sha, source in sources.items()}
