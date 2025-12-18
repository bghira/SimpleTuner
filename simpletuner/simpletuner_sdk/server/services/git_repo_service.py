"""Lightweight git utilities for config-scoped operations."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from urllib.parse import urlparse

from fastapi import status


class GitRepoError(Exception):
    """Domain error raised when git operations fail."""

    def __init__(self, message: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


@dataclass(frozen=True)
class GitStatus:
    """Summary of repository state scoped to the configs directory."""

    git_available: bool
    repo_present: bool
    repo_root: Optional[str] = None
    branch: Optional[str] = None
    remote: Optional[str] = None
    dirty_paths: Optional[List[str]] = None
    ahead: Optional[int] = None
    behind: Optional[int] = None
    user_name: Optional[str] = None
    user_email: Optional[str] = None
    identity_configured: bool = False


class GitRepoService:
    """Low-level git operations scoped to a configs directory."""

    def __init__(self):
        self._lock = threading.Lock()

    # ---------------------------
    # Helpers
    # ---------------------------
    @staticmethod
    def _ensure_git_installed() -> None:
        if shutil.which("git") is None:
            raise GitRepoError("git is not installed or not on PATH", status.HTTP_503_SERVICE_UNAVAILABLE)

    @staticmethod
    def _normalize_dir(config_dir: Path | str) -> Path:
        path = Path(config_dir).expanduser()
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return path.resolve()

    @staticmethod
    def _run_git(args: Sequence[str], cwd: Path, check: bool = True, timeout: int = 15) -> subprocess.CompletedProcess:
        cmd = ["git", *args]
        try:
            result = subprocess.run(
                cmd,
                cwd=str(cwd),
                check=check,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result
        except subprocess.CalledProcessError as exc:  # pragma: no cover - passthrough
            message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
            raise GitRepoError(message)
        except FileNotFoundError as exc:  # pragma: no cover - git missing
            raise GitRepoError(str(exc), status.HTTP_503_SERVICE_UNAVAILABLE) from exc

    @staticmethod
    def _validate_remote_url(remote: Optional[str]) -> Optional[str]:
        if remote is None:
            return None
        cleaned = remote.strip()
        if not cleaned:
            return None
        if cleaned.startswith("git@"):
            return cleaned
        parsed = urlparse(cleaned)
        if parsed.scheme in {"https", "http", "ssh", "git"} and parsed.netloc:
            return cleaned
        raise GitRepoError(
            "Remote URL must use ssh://, https://, git://, or git@host:path format.",
            status.HTTP_400_BAD_REQUEST,
        )

    @staticmethod
    def _validate_branch_name(name: str) -> str:
        cleaned = (name or "").strip()
        if not cleaned:
            raise GitRepoError("Branch name is required", status.HTTP_400_BAD_REQUEST)
        invalid_substrings = ("..", "~", "^", ":", "?", "*", "[", "\\")
        if any(token in cleaned for token in invalid_substrings):
            raise GitRepoError("Branch name contains invalid characters", status.HTTP_400_BAD_REQUEST)
        if cleaned.startswith("-") or cleaned.endswith("/") or cleaned.endswith(".lock") or "//" in cleaned:
            raise GitRepoError("Branch name is not allowed", status.HTTP_400_BAD_REQUEST)
        if any(ch.isspace() for ch in cleaned):
            raise GitRepoError("Branch name cannot contain whitespace", status.HTTP_400_BAD_REQUEST)
        return cleaned

    @staticmethod
    def _validate_remote_name(remote: str) -> str:
        cleaned = (remote or "").strip()
        if not cleaned:
            raise GitRepoError("Remote name is required", status.HTTP_400_BAD_REQUEST)
        if cleaned.startswith("-") or re.search(r"\s", cleaned):
            raise GitRepoError("Remote name is invalid", status.HTTP_400_BAD_REQUEST)
        return cleaned

    @staticmethod
    def _validate_commit_message(message: str) -> str:
        cleaned = (message or "").strip()
        if not cleaned:
            raise GitRepoError("Commit message is required", status.HTTP_400_BAD_REQUEST)
        if "\n" in cleaned or "\r" in cleaned or re.search(r"[\x00-\x08\x0b-\x1f\x7f]", cleaned):
            raise GitRepoError("Commit message cannot contain control characters or newlines", status.HTTP_400_BAD_REQUEST)
        return cleaned

    @staticmethod
    def _validate_email(email: str) -> str:
        cleaned = (email or "").strip()
        if not cleaned:
            raise GitRepoError("Email is required", status.HTTP_400_BAD_REQUEST)
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", cleaned):
            raise GitRepoError("Invalid email address", status.HTTP_400_BAD_REQUEST)
        return cleaned

    def _set_remote_locked(self, root: Path, remote: Optional[str]) -> None:
        validated = self._validate_remote_url(remote)
        if validated is None:
            existing = self._run_git(["remote"], cwd=root, check=False).stdout.splitlines()
            if "origin" in existing:
                self._run_git(["remote", "remove", "origin"], cwd=root, check=False)
            return
        existing = self._run_git(["remote"], cwd=root, check=False).stdout.splitlines()
        if "origin" in existing:
            self._run_git(["remote", "set-url", "origin", validated], cwd=root)
        else:
            self._run_git(["remote", "add", "origin", validated], cwd=root)

    def _repo_root(self, config_dir: Path) -> Optional[Path]:
        try:
            result = self._run_git(["rev-parse", "--show-toplevel"], cwd=config_dir)
        except GitRepoError:
            return None
        root = Path(result.stdout.strip() or "").resolve()
        return root if root.exists() else None

    def _ensure_repo(self, config_dir: Path) -> Path:
        root = self._repo_root(config_dir)
        if root is None:
            raise GitRepoError("Not a git repository. Initialize first.", status.HTTP_404_NOT_FOUND)
        try:
            config_dir.resolve().relative_to(root)
        except ValueError as exc:
            raise GitRepoError(
                "Configs directory is outside the git repository; git actions are disabled.",
                status.HTTP_400_BAD_REQUEST,
            ) from exc
        return root

    def _read_identity(self, root: Path) -> tuple[Optional[str], Optional[str]]:
        name = self._run_git(["config", "--get", "user.name"], cwd=root, check=False).stdout.strip() or None
        email = self._run_git(["config", "--get", "user.email"], cwd=root, check=False).stdout.strip() or None
        return name, email

    def _ensure_identity(self, root: Path) -> None:
        name, email = self._read_identity(root)
        if name and email:
            return

        default_name = os.environ.get("GIT_AUTHOR_NAME") or os.environ.get("GIT_COMMITTER_NAME") or "SimpleTuner"
        default_email = (
            os.environ.get("GIT_AUTHOR_EMAIL") or os.environ.get("GIT_COMMITTER_EMAIL") or "simpletuner@example.com"
        )
        if not name:
            self._run_git(["config", "user.name", default_name], cwd=root)
        if not email:
            self._run_git(["config", "user.email", default_email], cwd=root)

    def _relativize(self, root: Path, target: Path) -> str:
        try:
            return target.resolve().relative_to(root).as_posix()
        except ValueError as exc:
            raise GitRepoError(
                f"Path '{target}' is outside the repository root '{root}'",
                status.HTTP_400_BAD_REQUEST,
            ) from exc

    def _parse_dirty(self, repo_root: Path, config_dir: Path) -> List[str]:
        result = self._run_git(["status", "--porcelain"], cwd=repo_root, check=False)
        dirty: List[str] = []
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            # Lines look like: "XY path"
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            path_part = parts[1].strip()
            candidate = (repo_root / path_part).resolve()
            try:
                candidate.relative_to(config_dir)
            except ValueError:
                continue
            dirty.append(self._relativize(config_dir, candidate))
        return dirty

    # ---------------------------
    # Public API
    # ---------------------------
    def discover_repo(self, config_dir: Path | str) -> GitStatus:
        """Return repo status for the given configs directory."""
        self._ensure_git_installed()
        config_path = self._normalize_dir(config_dir)
        root = self._repo_root(config_path)
        user_name = None
        user_email = None
        identity_configured = False
        if not root:
            return GitStatus(
                git_available=True,
                repo_present=False,
                repo_root=None,
                branch=None,
                remote=None,
                dirty_paths=[],
                ahead=None,
                behind=None,
                user_name=None,
                user_email=None,
                identity_configured=False,
            )

        try:
            config_path.relative_to(root)
        except ValueError:
            raise GitRepoError(
                "Configs directory is outside the git repository; git actions are disabled.",
                status.HTTP_400_BAD_REQUEST,
            )

        branch = None
        try:
            branch = self._run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=root).stdout.strip() or None
        except GitRepoError:
            branch = None

        remote = None
        try:
            remote = self._run_git(["config", "--get", "remote.origin.url"], cwd=root, check=False).stdout.strip() or None
        except GitRepoError:
            remote = None

        ahead = behind = None
        try:
            counts = self._run_git(["rev-list", "--left-right", "--count", "@{upstream}...HEAD"], cwd=root, check=False)
            tokens = (counts.stdout.strip() or "").split()
            if len(tokens) == 2:
                behind = int(tokens[0])
                ahead = int(tokens[1])
        except Exception:
            ahead = behind = None

        dirty_paths = self._parse_dirty(root, config_path)
        try:
            user_name, user_email = self._read_identity(root)
            identity_configured = bool(user_name and user_email)
        except Exception:
            user_name = None
            user_email = None
            identity_configured = False

        return GitStatus(
            git_available=True,
            repo_present=True,
            repo_root=str(root),
            branch=branch,
            remote=remote,
            dirty_paths=dirty_paths,
            ahead=ahead,
            behind=behind,
            user_name=user_name,
            user_email=user_email,
            identity_configured=identity_configured,
        )

    def init_repo(self, config_dir: Path | str, remote: Optional[str] = None, branch: Optional[str] = None) -> GitStatus:
        """Initialize a repository rooted at the configs directory."""
        with self._lock:
            self._ensure_git_installed()
            config_path = self._normalize_dir(config_dir)
            self._run_git(["init"], cwd=config_path)
            if branch:
                validated_branch = self._validate_branch_name(branch)
                self._run_git(["checkout", "-B", validated_branch], cwd=config_path, check=False)
            if remote is not None:
                self._set_remote_locked(config_path, remote)
            repo_status = self.discover_repo(config_path)
        return repo_status

    def set_remote(self, config_dir: Path | str, remote: Optional[str]) -> GitStatus:
        """Set or clear the origin remote."""
        with self._lock:
            root = self._ensure_repo(self._normalize_dir(config_dir))
            self._set_remote_locked(root, remote)
            repo_status = self.discover_repo(root)
        return repo_status

    def create_or_switch_branch(self, config_dir: Path | str, name: str, create: bool = False) -> GitStatus:
        with self._lock:
            config_path = self._normalize_dir(config_dir)
            root = self._ensure_repo(config_path)
            dirty_paths = self._parse_dirty(root, config_path)
            if dirty_paths:
                raise GitRepoError(
                    f"Cannot switch branches with uncommitted changes in configs directory: {', '.join(dirty_paths[:3])}{'...' if len(dirty_paths) > 3 else ''}",
                    status.HTTP_409_CONFLICT,
                )
            branch_name = self._validate_branch_name(name)
            args = ["checkout", "-B" if create else "", branch_name]
            args = [part for part in args if part]
            self._run_git(args, cwd=root)
            repo_status = self.discover_repo(root)
        return repo_status

    def fetch(self, config_dir: Path | str, remote: Optional[str] = None) -> GitStatus:
        with self._lock:
            root = self._ensure_repo(self._normalize_dir(config_dir))
            args = ["fetch"]
            if remote:
                args.append(self._validate_remote_name(remote))
            self._run_git(args, cwd=root)
            repo_status = self.discover_repo(root)
        return repo_status

    def pull(
        self,
        config_dir: Path | str,
        remote: Optional[str] = None,
        branch: Optional[str] = None,
        autostash: bool = True,
    ) -> GitStatus:
        with self._lock:
            root = self._ensure_repo(self._normalize_dir(config_dir))
            args = ["pull"]
            if autostash:
                args.append("--autostash")
            if remote:
                args.append(self._validate_remote_name(remote))
            if branch:
                args.append(self._validate_branch_name(branch))
            try:
                self._run_git(args, cwd=root, timeout=60)
            except GitRepoError as exc:
                if "conflict" in exc.message.lower() or "merge" in exc.message.lower():
                    raise GitRepoError(
                        f"Pull failed due to conflicts. Resolve manually or stash changes: {exc.message}",
                        status.HTTP_409_CONFLICT,
                    ) from exc
                raise
            repo_status = self.discover_repo(root)
        return repo_status

    def push(self, config_dir: Path | str, remote: Optional[str] = None, branch: Optional[str] = None) -> GitStatus:
        with self._lock:
            root = self._ensure_repo(self._normalize_dir(config_dir))
            args = ["push"]
            if remote:
                args.append(self._validate_remote_name(remote))
            if branch:
                args.append(self._validate_branch_name(branch))
            self._run_git(args, cwd=root, timeout=60)
            repo_status = self.discover_repo(root)
        return repo_status

    def log(self, config_dir: Path | str, path: Path | str, skip: int = 0, limit: int = 20) -> List[Dict[str, object]]:
        with self._lock:
            root = self._ensure_repo(self._normalize_dir(config_dir))
            target = Path(path)
            rel = self._relativize(root, target)
            log_format = "%H|%h|%ct|%an|%s"
            args = ["log", f"--skip={max(0, int(skip))}", f"-n{max(1, int(limit))}", f"--format={log_format}", "--", rel]
            result = self._run_git(args, cwd=root, check=False)
        commits: List[Dict[str, object]] = []
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            parts = line.split("|", 4)
            if len(parts) != 5:
                continue
            full, short, timestamp, author, subject = parts
            try:
                ts = int(timestamp)
            except ValueError:
                ts = None
            commits.append(
                {
                    "commit": full,
                    "abbrev": short,
                    "timestamp": ts,
                    "author": author,
                    "subject": subject,
                }
            )
        return commits

    def diff(
        self,
        config_dir: Path | str,
        path: Path | str,
        commit: Optional[str] = None,
        max_bytes: int = 100_000,
        max_lines: int = 500,
    ) -> Dict[str, object]:
        with self._lock:
            root = self._ensure_repo(self._normalize_dir(config_dir))
            target = Path(path)
            rel = self._relativize(root, target)
            args = ["diff"]
            if commit:
                if re.search(r"\s", commit):
                    raise GitRepoError("Commit reference contains whitespace", status.HTTP_400_BAD_REQUEST)
                args.append(commit)
            args.extend(["--", rel])
            result = self._run_git(args, cwd=root, check=False)
        content = result.stdout or ""
        truncated = False
        if len(content.encode("utf-8")) > max_bytes:
            content = content.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")
            truncated = True
        lines = content.splitlines()
        if len(lines) > max_lines:
            content = "\n".join(lines[:max_lines])
            truncated = True
        return {"path": rel, "diff": content, "truncated": truncated}

    def stage_and_commit(
        self,
        config_dir: Path | str,
        paths: Sequence[Path | str],
        message: str,
        include_untracked: bool = False,
    ) -> Dict[str, str]:
        if not paths:
            raise GitRepoError("No paths provided to commit", status.HTTP_400_BAD_REQUEST)
        cleaned_message = self._validate_commit_message(message)

        with self._lock:
            root = self._ensure_repo(self._normalize_dir(config_dir))
            rel_paths = [self._relativize(root, Path(p)) for p in paths]

            self._ensure_identity(root)

            add_args = ["add"]
            if not include_untracked:
                add_args.append("-u")
            add_args.extend(rel_paths)
            self._run_git(add_args, cwd=root)

            diff_cached = self._run_git(["diff", "--cached", "--name-only"], cwd=root)
            if not diff_cached.stdout.strip():
                raise GitRepoError("No changes staged for commit", status.HTTP_400_BAD_REQUEST)

            self._run_git(["commit", "-m", cleaned_message], cwd=root)
        return {"message": "Committed changes", "commit_message": cleaned_message}

    def set_identity(self, config_dir: Path | str, name: str, email: str) -> Dict[str, str]:
        cleaned_name = (name or "").strip()
        cleaned_email = self._validate_email(email)
        if not cleaned_name:
            raise GitRepoError("Both user name and email are required", status.HTTP_400_BAD_REQUEST)
        with self._lock:
            root = self._ensure_repo(self._normalize_dir(config_dir))
            self._run_git(["config", "user.name", cleaned_name], cwd=root)
            self._run_git(["config", "user.email", cleaned_email], cwd=root)
        return {"message": "Updated git identity", "user_name": cleaned_name, "user_email": cleaned_email}

    def restore_path(self, config_dir: Path | str, path: Path | str, commit: Optional[str] = None) -> Dict[str, str]:
        with self._lock:
            root = self._ensure_repo(self._normalize_dir(config_dir))
            rel = self._relativize(root, Path(path))
            args = ["checkout"]
            if commit:
                if re.search(r"\s", commit):
                    raise GitRepoError("Commit reference contains whitespace", status.HTTP_400_BAD_REQUEST)
                args.append(commit)
            args.extend(["--", rel])
            self._run_git(args, cwd=root)
        return {"message": f"Restored {rel}"}


GIT_REPO_SERVICE = GitRepoService()
