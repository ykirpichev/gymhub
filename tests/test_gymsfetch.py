"""gymhub/gymsfetch.py: the gyms come from their own repository, pinned by gyms.lock."""

import io
import json
import tarfile

import pytest

from gymhub import gymsfetch


def tarball(files: dict[str, str], top: str = "owner-gyms-abc123") -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(f"{top}/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        link = tarfile.TarInfo(f"{top}/evil")
        link.type, link.linkname = tarfile.SYMTYPE, "/etc/passwd"
        tar.addfile(link)
    return buffer.getvalue()


@pytest.fixture
def lock(tmp_path):
    path = tmp_path / "gyms.lock"
    path.write_text(json.dumps({"repo": "owner/gyms", "ref": "abc123"}))
    return path


def test_fetch_extracts_the_pinned_ref_once(tmp_path, lock):
    calls = []

    def download(repo, ref):
        calls.append((repo, ref))
        return tarball({"hello/gym.json": "{}", "README.md": "gyms"})

    dest = tmp_path / "gyms"
    assert "fetched owner/gyms@abc123" in gymsfetch.fetch(dest=dest, lock=lock, download=download)
    assert (dest / "hello" / "gym.json").read_text() == "{}"
    assert not (dest / "evil").exists()  # links in the tarball are skipped
    assert gymsfetch.state(dest) == "abc123"
    gymsfetch.fetch(dest=dest, lock=lock, download=download)
    assert calls == [("owner/gyms", "abc123")]  # already there: no second download
    gymsfetch.fetch(ref="main", dest=dest, lock=lock, download=lambda repo, ref: tarball({"other/gym.json": "{}"}))
    assert (dest / "other").is_dir() and not (dest / "hello").exists()
    assert gymsfetch.state(dest) == "main"


def test_a_gyms_folder_it_did_not_write_is_left_alone(tmp_path, lock):
    dest = tmp_path / "gyms"
    (dest / "mine").mkdir(parents=True)
    assert gymsfetch.state(dest) == ""
    assert "left as it is" in gymsfetch.fetch(dest=dest, lock=lock, download=lambda *a: pytest.fail("downloaded"))
    assert (dest / "mine").is_dir()
    link = tmp_path / "linked"
    link.symlink_to(dest)
    assert gymsfetch.state(link) == ""


def test_ensure_only_fetches_when_missing_and_never_raises(tmp_path, lock, monkeypatch, capsys):
    dest = tmp_path / "gyms"

    def fail(repo, ref):
        raise gymsfetch.FetchError("offline")

    monkeypatch.setattr(gymsfetch, "_download", fail)
    assert gymsfetch.ensure(dest=dest, lock=lock) is None
    assert "offline" in capsys.readouterr().err
    assert not dest.exists()
    dest.mkdir()
    assert gymsfetch.ensure(dest=dest, lock=lock) is None  # there already


def test_the_repository_pins_the_gyms_repository():
    pin = gymsfetch.pinned()
    assert pin["repo"] == "ykirpichev/gymhub-gyms"
    assert len(pin["ref"]) == 40
