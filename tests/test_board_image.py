"""The board's container image copies only some files. Every gymhub module the board imports must be among them.

A missing file only shows up when Cloud Run fails to start the new revision, so check it here.
"""

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _image_files() -> set[str]:
    files: set[str] = set()
    for line in (REPO / "deploy" / "Dockerfile.board").read_text().splitlines():
        if not line.startswith("COPY "):
            continue
        *sources, _dest = line.split()[1:]
        for source in sources:
            path = REPO / source
            if path.is_dir():
                files.update(str(p.relative_to(REPO)) for p in path.rglob("*.py"))
            else:
                files.add(source)
    return files


def _gymhub_imports(path: Path) -> set[str]:
    modules = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("gymhub"):
            modules.add(node.module)
            modules.update(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names if alias.name.startswith("gymhub"))
    return modules


def _module_file(module: str) -> str | None:
    base = REPO / Path(*module.split("."))
    if (base / "__init__.py").is_file():
        return str((base / "__init__.py").relative_to(REPO))
    if base.with_suffix(".py").is_file():
        return str(base.with_suffix(".py").relative_to(REPO))
    return None  # a name imported from a module, not a module


def test_board_image_contains_every_imported_module():
    files = _image_files()
    seen, todo = set(), [f for f in files if f.endswith(".py")]
    missing = set()
    while todo:
        current = todo.pop()
        if current in seen:
            continue
        seen.add(current)
        for module in _gymhub_imports(REPO / current):
            target = _module_file(module)
            if target is None:
                continue
            if target not in files:
                missing.add(target)
            todo.append(target)
    assert not missing, f"deploy/Dockerfile.board does not copy {sorted(missing)}"


def test_deploy_script_stages_the_same_files():
    script = (REPO / "deploy" / "gcp.sh").read_text()
    staged = set(re.findall(r'"\$REPO/(gymhub/[\w./]+)"', script))
    for source in (f for f in _image_files() if f.startswith("gymhub/")):
        top = source.split("/")[1] if source.count("/") > 1 else source
        assert any(s == source or s.startswith(f"gymhub/{top}") for s in staged), f"gcp.sh does not stage {source}"


def test_board_image_contains_the_python_it_serves_to_browsers():
    app = (REPO / "gymhub" / "board" / "app.py").read_text()
    script = (REPO / "deploy" / "gcp.sh").read_text()
    files = _image_files()
    for name in ("compare.py", "examples.py"):
        assert f'"{name}"' in app  # BROWSER_PY
        assert f"gymkit/{name}" in files, f"deploy/Dockerfile.board does not copy gymkit/{name}"
        assert f'"$REPO/gymkit/{name}"' in script, f"gcp.sh does not stage gymkit/{name}"
