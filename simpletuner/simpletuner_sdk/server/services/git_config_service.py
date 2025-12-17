"""Git-aware helpers layered on top of ConfigStore semantics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from fastapi import status

from simpletuner.simpletuner_sdk.server.services.config_store import ConfigStore
from simpletuner.simpletuner_sdk.server.services.git_repo_service import GIT_REPO_SERVICE, GitRepoError, GitStatus
from simpletuner.simpletuner_sdk.server.services.webui_state import WebUIStateStore
from simpletuner.simpletuner_sdk.server.utils.paths import resolve_config_path


class GitConfigError(Exception):
    """Domain error raised when config-aware git operations fail."""

    def __init__(self, message: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


@dataclass(frozen=True)
class SnapshotPreferences:
    """User preferences controlling git behaviour on save."""

    auto_commit: bool = False
    require_clean: bool = False
    include_untracked: bool = False
    push_on_snapshot: bool = False
    default_message: Optional[str] = None


class GitConfigService:
    """Map config names to file paths and perform scoped git operations."""

    def __init__(self):
        self._store_cache: Dict[str, ConfigStore] = {}

    # ---------------------------
    # Helpers
    # ---------------------------
    def _get_store(self, config_type: str = "model") -> ConfigStore:
        """Retrieve a ConfigStore using persisted defaults when available."""
        cache_key = config_type
        if cache_key in self._store_cache:
            return self._store_cache[cache_key]
        try:
            defaults = WebUIStateStore().load_defaults()
            if defaults.configs_dir:
                store = ConfigStore(config_dir=Path(defaults.configs_dir).expanduser(), config_type=config_type)
                self._store_cache[cache_key] = store
                return store
        except Exception:
            pass
        store = ConfigStore(config_type=config_type)
        self._store_cache[cache_key] = store
        return store

    def _resolve_path(self, name_or_path: str, config_type: str = "model") -> Path:
        """Resolve a config name or path to an absolute file path."""
        store = self._get_store(config_type)
        candidate = Path(name_or_path).expanduser()

        if candidate.is_absolute() or candidate.parts[:1] == (".",):
            resolved = resolve_config_path(str(candidate), config_dir=store.config_dir, check_cwd_first=True)
            if resolved:
                return resolved.resolve()
            raise GitConfigError(f"Config path not found: {name_or_path}", status.HTTP_404_NOT_FOUND)

        # Fallback to ConfigStore path resolution
        target = store._get_config_path(name_or_path)  # noqa: SLF001 - deliberate reuse
        if not target.exists():
            raise GitConfigError(f"Config '{name_or_path}' not found", status.HTTP_404_NOT_FOUND)
        return target.resolve()

    def _ensure_tracked_location(self, path: Path, config_dir: Path) -> None:
        try:
            path.relative_to(config_dir.resolve())
        except ValueError as exc:
            raise GitConfigError(
                "Config path is outside the configs directory; git snapshot skipped",
                status.HTTP_400_BAD_REQUEST,
            ) from exc

    def _status(self, config_type: str = "model") -> Tuple[GitStatus, Path]:
        store = self._get_store(config_type)
        repo_status = GIT_REPO_SERVICE.discover_repo(store.config_dir)
        return repo_status, Path(store.config_dir).expanduser().resolve()

    # ---------------------------
    # Public API
    # ---------------------------
    def is_git_ready(self, config_type: str = "model") -> GitStatus:
        """Return git readiness information for the configs directory."""
        repo_status, _ = self._status(config_type)
        return repo_status

    def history_for_config(self, name_or_path: str, config_type: str = "model", skip: int = 0, limit: int = 20):
        path = self._resolve_path(name_or_path, config_type)
        repo_status, _ = self._status(config_type)
        if not repo_status.repo_present:
            raise GitConfigError("Repository not initialized", status.HTTP_404_NOT_FOUND)
        return GIT_REPO_SERVICE.log(path.parent, path, skip=skip, limit=limit)

    def diff_for_config(
        self,
        name_or_path: str,
        config_type: str = "model",
        commit: Optional[str] = None,
        max_bytes: int = 100_000,
        max_lines: int = 500,
    ):
        path = self._resolve_path(name_or_path, config_type)
        repo_status, _ = self._status(config_type)
        if not repo_status.repo_present:
            raise GitConfigError("Repository not initialized", status.HTTP_404_NOT_FOUND)
        return GIT_REPO_SERVICE.diff(path.parent, path, commit=commit, max_bytes=max_bytes, max_lines=max_lines)

    def snapshot_configs(
        self,
        names_or_paths: Sequence[str],
        message: str,
        config_type: str = "model",
        include_untracked: bool = False,
        push_on_snapshot: bool = False,
    ) -> Dict[str, object]:
        if not names_or_paths:
            raise GitConfigError("No configuration paths provided", status.HTTP_400_BAD_REQUEST)

        paths = [self._resolve_path(item, config_type) for item in names_or_paths]
        store = self._get_store(config_type)
        for path in paths:
            self._ensure_tracked_location(path, Path(store.config_dir))

        result = GIT_REPO_SERVICE.stage_and_commit(store.config_dir, paths, message, include_untracked=include_untracked)

        if push_on_snapshot:
            try:
                GIT_REPO_SERVICE.push(store.config_dir, None, None)
            except GitRepoError:
                # Surface push errors but keep commit success
                result["push_error"] = "Push failed; verify remote and credentials."

        return result

    def restore_config(self, name_or_path: str, config_type: str = "model", commit: Optional[str] = None):
        path = self._resolve_path(name_or_path, config_type)
        store = self._get_store(config_type)
        self._ensure_tracked_location(path, Path(store.config_dir))
        return GIT_REPO_SERVICE.restore_path(store.config_dir, path, commit=commit)

    def snapshot_on_save(
        self,
        name: str,
        config_type: str,
        prefs: SnapshotPreferences,
        message: Optional[str] = None,
    ) -> Optional[Dict[str, object]]:
        repo_status, _ = self._status(config_type)
        if not repo_status.repo_present:
            return None

        if prefs.require_clean and repo_status.dirty_paths:
            raise GitConfigError(
                "Working tree has uncommitted changes; clean or commit before saving.",
                status.HTTP_409_CONFLICT,
            )

        if not prefs.auto_commit:
            return None

        final_message = (message or prefs.default_message or "").strip() or f"env:{name} update"
        return self.snapshot_configs(
            names_or_paths=[name],
            message=final_message,
            config_type=config_type,
            include_untracked=prefs.include_untracked,
            push_on_snapshot=prefs.push_on_snapshot,
        )


GIT_CONFIG_SERVICE = GitConfigService()
