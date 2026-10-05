#!/usr/bin/env python3
"""Install a task repo's runtime and test dependencies into the current venv.

The contest sandbox has dependencies prebaked; data/sandbox/setup.py alone
misses test-only deps (dependency groups) and can pick wheels built for the
wrong Python. This installs from the repo's own lockfile when there is one,
else from its declared deps, using /wheels first and PyPI for the rest.

  <venv>/bin/python scripts/install_deps.py /path/to/repo
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

SKIP_GROUPS = {"docs", "docs-tests", "github-actions", "translations"}


def pip_install(reqs: list[str]) -> None:
    if not reqs:
        return
    base = [sys.executable, "-m", "pip", "install", "-q", "--find-links=/wheels", "--prefer-binary"]
    if subprocess.run(base + reqs).returncode == 0:
        return
    # One bad pin fails the whole call, so fall back to one at a time.
    for req in reqs:
        if subprocess.run(base + [req], capture_output=True).returncode != 0:
            name = re.split(r"[<>=!~;\[ ]", req, maxsplit=1)[0]
            if subprocess.run(base + [name], capture_output=True).returncode != 0:
                print(f"install_deps: could not install {req}", file=sys.stderr)


def from_uv_lock(repo: Path, groups: list[str]) -> list[str]:
    cmd = ["uv", "export", "--frozen", "--no-hashes", "--no-emit-project", "--all-extras",
           "--no-header", "--no-annotate"]
    for g in groups:
        cmd += ["--group", g]
    out = subprocess.run(cmd, cwd=repo, text=True, capture_output=True)
    if out.returncode != 0:
        print(f"install_deps: uv export failed: {out.stderr[-500:]}", file=sys.stderr)
        return []
    return [l.strip() for l in out.stdout.splitlines() if l.strip() and not l.startswith(("#", "-e", "."))]


def from_poetry_lock(repo: Path, wanted: set[str]) -> list[str]:
    lock = tomllib.loads((repo / "poetry.lock").read_text())
    pins = {norm(p["name"]): p["version"] for p in lock.get("package", [])}
    # Pin every locked package the repo asks for; pip resolves their deps.
    return [f"{n}=={pins[n]}" if n in pins else n for n in sorted(wanted)]


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def req_name(req: str) -> str:
    return norm(re.split(r"[<>=!~;\[ (]", req.strip(), maxsplit=1)[0])


def declared(repo: Path, pyproject: dict) -> tuple[list[str], list[str]]:
    """Requirement strings from pyproject (PEP 621, PEP 735, poetry) and requirements files."""
    reqs: list[str] = []
    project = pyproject.get("project", {})
    reqs += project.get("dependencies", [])
    for deps in project.get("optional-dependencies", {}).values():
        reqs += deps
    groups = pyproject.get("dependency-groups", {})
    group_names = [g for g in groups if g not in SKIP_GROUPS]
    for g in group_names:
        reqs += [d for d in groups[g] if isinstance(d, str)]
    poetry = pyproject.get("tool", {}).get("poetry", {})
    for key in ("dependencies", "dev-dependencies"):
        # Skip optional extras (e.g. rich's ipywidgets): not in /wheels, and ipython
        # being importable changes pygments lexer guessing in rich's tests.
        reqs += [n for n, spec in poetry.get(key, {}).items()
                 if n.lower() != "python" and not (isinstance(spec, dict) and spec.get("optional"))]
    for grp in poetry.get("group", {}).values():
        reqs += list(grp.get("dependencies", {}))
    for rf in list(repo.glob("requirements*.txt")) + list(repo.glob("test-requirements*.txt")):
        for line in rf.read_text(errors="replace").splitlines():
            line = line.split("#", 1)[0].strip()
            if line and not line.startswith("-"):
                reqs.append(line)
    return reqs, group_names


def main() -> None:
    repo = Path(sys.argv[1])
    pyproject = tomllib.loads((repo / "pyproject.toml").read_text()) if (repo / "pyproject.toml").exists() else {}
    reqs, groups = declared(repo, pyproject)
    own = norm(pyproject.get("project", {}).get("name", "") or pyproject.get("tool", {}).get("poetry", {}).get("name", ""))
    if (repo / "uv.lock").exists():
        locked = from_uv_lock(repo, groups)
        reqs = locked or reqs
        print(f"install_deps: {len(reqs)} reqs from uv.lock (groups {groups})")
    elif (repo / "poetry.lock").exists():
        reqs = from_poetry_lock(repo, {req_name(r) for r in reqs})
        print(f"install_deps: {len(reqs)} reqs pinned from poetry.lock")
    else:
        print(f"install_deps: {len(reqs)} declared reqs")
    reqs = [r for r in dict.fromkeys(r.strip() for r in reqs) if r and req_name(r) != own]
    pip_install(reqs)
    # Reinstall the repo itself in editable mode, with its deps: this applies the repo's own
    # version bounds (setup.py-era requests needs urllib3; old fastapi needs an old starlette),
    # since setup.py always picks the newest wheel.
    editable = [sys.executable, "-m", "pip", "install", "-q", "--find-links=/wheels", "--no-build-isolation", "-e", str(repo)]
    if subprocess.run(editable, capture_output=True).returncode != 0:
        subprocess.run(editable[:4] + ["--no-deps"] + editable[4:], capture_output=True)


if __name__ == "__main__":
    main()
