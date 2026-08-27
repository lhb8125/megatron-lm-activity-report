"""Exact-path Git commits used by the publication transaction."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable


class RepositoryError(RuntimeError):
    pass


class GitRepository:
    def __init__(self, root: Path, *, branch: str):
        self.root = root
        self.branch = branch

    def commit_and_push(self, paths: Iterable[Path], *, message: str) -> str | None:
        relative = [str(path.resolve().relative_to(self.root.resolve())) for path in paths]
        if not relative:
            return None
        self._run(["git", "add", "--", *relative])
        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet", "--", *relative],
            cwd=self.root,
            check=False,
        )
        if staged.returncode == 0:
            return None
        if staged.returncode != 1:
            raise RepositoryError("git diff --cached failed")
        self._run(["git", "commit", "-m", message, "--", *relative])
        commit = self._run(["git", "rev-parse", "HEAD"]).strip()
        self._run(["git", "push", "origin", f"HEAD:{self.branch}"])
        return commit

    def _run(self, command: list[str]) -> str:
        completed = subprocess.run(
            command,
            cwd=self.root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if completed.returncode != 0:
            raise RepositoryError(
                f"{' '.join(command[:2])} failed with {completed.returncode}: "
                f"{completed.stdout[-3000:]}"
            )
        return completed.stdout
