"""Home's "Start here" points at an exercise a machine that is up can grade: each gym on /api/home says
whether one can (`gradable`), so a learner without a GPU does not start on a CUDA kernel."""

import time

from gymhub.runner import catalog
from tests.test_board_auth import board, client, sign_in  # noqa: F401  (fixture)


def _gym(store, gym_id, order, **extra):
    store.set("gyms", gym_id, {"id": gym_id, "title": gym_id, "order": order, "visibility": "public", **extra})
    store.set("problems", f"{gym_id}__p", {"gym": gym_id, "slug": "p", "title": "P", "langs": ["python"], "order": 0})


def _runner(store, gpus, **extra):
    # A runner on the store itself (no worker token): the owner's own machine, as gymhub-board starts it.
    store.set("runners", "box", {"name": "box", "heartbeat_at": time.time(),
                                 "capabilities": {"arch": "x86_64", "gpus": gpus, "tools": ["g++"], **extra}})


def _gradable(c):
    return {gym["id"]: gym["gradable"] for gym in c.get("/api/home").json()["gyms"]}


def test_a_machine_without_a_gpu_grades_only_the_cpu_gyms(board):
    _gym(board.store, "gpu", 0, requires={"gpu": True})
    _gym(board.store, "algo", 1, requires={"gpu": False})
    _gym(board.store, "ml-coding", 2, requires={"gpu": False}, machine_ready=False)  # its runner lacks torch
    _runner(board.store, [])
    owner = client(board)
    sign_in(owner, "owner")
    assert _gradable(owner) == {"gpu": False, "algo": True, "ml-coding": False}


def test_a_gpu_machine_grades_every_gym(board):
    _gym(board.store, "gpu", 0, requires={"gpu": True})
    _gym(board.store, "algo", 1, requires={"gpu": False})
    _runner(board.store, [{"name": "RTX 4090"}])
    owner = client(board)
    sign_in(owner, "owner")
    assert _gradable(owner) == {"gpu": True, "algo": True}


def test_no_machine_up_grades_nothing(board):
    _gym(board.store, "algo", 1, requires={"gpu": False})
    owner = client(board)
    sign_in(owner, "owner")
    assert _gradable(owner) == {"algo": False}


def test_the_catalog_says_whether_its_machine_can_grade_it(tmp_path):
    import shutil

    from tests.test_local_board import HELLO

    root = tmp_path / "hello"
    shutil.copytree(HELLO, root)
    assert catalog.build(root)["gym"]["machine_ready"] is True
    grader = root / "problems" / "sum_list" / "problem.py"
    grader.write_text("import torch_that_is_not_installed  # noqa\n" + grader.read_text())
    assert catalog.build(root)["gym"]["machine_ready"] is False
