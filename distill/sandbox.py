"""One task's Docker sandbox, shaped like the contest's (HARNESS_README 4.1-4.2).

The agent's commands run in a container with no network, 4 GiB and 2 CPUs, that has
only the task repo at /workspace and its venv at /venv. It cannot see the contest
data (which holds the gold patches), other rollouts, or any cache. Setup and grading
run in separate containers that do see the data, and grading starts from a fresh copy
of the set-up repo, like the harness's Container B.

The repo and venv live in host folders bind-mounted at the same paths everywhere, so
the rollout reads and edits files on the host, and only run_command goes through docker.
Setup is the same for every rollout of a task, so it is built once into
<runs>/_setup/<task> and cloned for each rollout (copy-on-write on APFS and btrfs/xfs).
"""

from __future__ import annotations

import fcntl
import json
import platform
import shutil
import subprocess
from pathlib import Path

IMAGE = "gemma-distill"
PLATFORM = "linux/amd64"  # the contest wheels are x86-64
COMMAND_TIMEOUT = 120
CLIP = 5000


def clip(text: str) -> str:
    return text if len(text) <= CLIP else text[:CLIP] + "\n...[truncated]"


def clone_tree(src: Path, dst: Path) -> None:
    """Copy a directory tree, sharing blocks where the filesystem can (instant, no extra space)."""
    flags = ["-cR"] if platform.system() == "Darwin" else ["-a", "--reflink=auto"]
    if subprocess.run(["cp", *flags, str(src), str(dst)], capture_output=True).returncode != 0:
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst, symlinks=True)


class Sandbox:
    def __init__(self, task_id: str, name: str, workdir: Path, code: Path, data: Path):
        self.task_id, self.name, self.workdir = task_id, name, workdir.resolve()
        self.code, self.data = code.resolve(), data.resolve()
        self.agent_dir, self.grade_dir = self.workdir / "agent", self.workdir / "grade"
        self.repo = self.agent_dir / "repo"
        self.cache = self.workdir.parent / "_setup" / task_id

    def _dirs(self, base: Path) -> list[str]:
        (base / "repo").mkdir(parents=True, exist_ok=True)
        (base / "venv").mkdir(parents=True, exist_ok=True)
        return ["-v", f"{base / 'repo'}:/workspace", "-v", f"{base / 'venv'}:/venv"]

    def _trusted(self, base: Path, network: bool, *argv: str) -> subprocess.CompletedProcess:
        """Run a setup or grading step: sees code and data, never the agent."""
        mounts = []
        for sub in ("scripts", "distill", "submission"):
            mounts += ["-v", f"{self.code / sub}:/g/{sub}:ro"]
        cmd = ["docker", "run", "--rm", "--platform", PLATFORM, *([] if network else ["--network", "none"]),
               "-e", "GEMMA_ROOT=/g", "-e", "GEMMA_DATA=/data", "-v", f"{self.data}:/data:ro",
               *mounts, *self._dirs(base), IMAGE, "python", "/g/distill/in_container.py", *argv]
        return subprocess.run(cmd, text=True, capture_output=True)

    def _build_cache(self) -> None:
        """Build the task's pristine repo and venv once; concurrent rollouts wait on the lock."""
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache.with_suffix(".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if (self.cache / "ready").exists():
                return
            shutil.rmtree(self.cache, ignore_errors=True)
            res = self._trusted(self.cache, True, "setup", "--task", self.task_id)
            if res.returncode != 0:
                shutil.rmtree(self.cache, ignore_errors=True)
                raise RuntimeError(f"setup failed: {res.stderr[-1500:]}")
            (self.cache / "ready").touch()

    def setup(self) -> None:
        """Give the agent a fresh copy of the set-up repo, and keep another for grading."""
        self._build_cache()
        for dest in (self.agent_dir, self.grade_dir):
            dest.mkdir(parents=True, exist_ok=True)
            for sub in ("repo", "venv"):
                shutil.rmtree(dest / sub, ignore_errors=True)
                clone_tree(self.cache / sub, dest / sub)
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)
        subprocess.run(["docker", "run", "-d", "--name", self.name, "--label", "gemma-distill=agent",
                        "--platform", PLATFORM, "--network", "none", "--memory", "4g", "--cpus", "2",
                        "-w", "/workspace", "-e", "VIRTUAL_ENV=/venv", "-e", "TEST_TMPDIR=/tmp",
                        "-e", "PATH=/venv/bin:/usr/local/bin:/usr/bin:/bin",
                        *self._dirs(self.agent_dir), IMAGE, "sleep", "infinity"],
                       check=True, capture_output=True)

    def run(self, command: str) -> str:
        """run_command, with the harness's result shape."""
        proc = subprocess.run(["docker", "exec", "-w", "/workspace", self.name,
                               "timeout", "-s", "KILL", str(COMMAND_TIMEOUT), "bash", "-lc", command],
                              text=True, capture_output=True, timeout=COMMAND_TIMEOUT + 60)
        if proc.returncode == 137:
            return json.dumps({"status": "error", "error_type": "TimeoutExceeded"})
        return json.dumps({"status": "ok" if proc.returncode == 0 else "error", "exit_code": proc.returncode,
                           "stdout": clip(proc.stdout), "stderr": clip(proc.stderr)})

    @staticmethod
    def _summary(res: subprocess.CompletedProcess) -> dict:
        lines = [l for l in res.stdout.splitlines() if l.startswith("SUMMARY ")]
        return json.loads(lines[-1][8:]) if lines else {"applied": False, "code": 1, "failing": [], "passed": 0}

    def _gold_baseline(self) -> dict:
        """Grade the maintainer patch once per task. Some hidden tests need the network and fail
        offline even with the gold fix; an agent patch is as good as gold if it fails no others."""
        path = self.cache / "gold.json"
        with open(self.cache.with_suffix(".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not path.exists():
                scratch = self.workdir / "gold"
                scratch.mkdir(parents=True, exist_ok=True)
                for sub in ("repo", "venv"):
                    clone_tree(self.cache / sub, scratch / sub)
                path.write_text(json.dumps(self._summary(self._trusted(scratch, False, "grade", "--task", self.task_id, "--gold"))))
                shutil.rmtree(scratch, ignore_errors=True)
        return json.loads(path.read_text())

    def grade(self, patch: str) -> tuple[bool, str]:
        """Apply the patch to the pristine copy and run the hidden tests, offline."""
        (self.workdir / "agent.patch").write_text(patch)
        shutil.copy(self.workdir / "agent.patch", self.grade_dir / "repo" / ".agent.patch")
        res = self._trusted(self.grade_dir, False, "grade", "--task", self.task_id, "--patch", "/workspace/.agent.patch")
        log = (res.stdout + res.stderr)[-3000:]
        if res.returncode == 0:
            return True, log
        mine = self._summary(res)
        if not mine["applied"]:
            return False, log
        gold = self._gold_baseline()
        as_good = (gold["code"] != 0 and gold["passed"] > 0 and set(mine["failing"]) <= set(gold["failing"])
                   and mine["passed"] >= gold["passed"])
        if as_good:
            log += f"\nAS GOOD AS GOLD: fails only tests the gold patch also fails offline ({len(gold['failing'])})"
        return as_good, log

    def close(self) -> None:
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)
