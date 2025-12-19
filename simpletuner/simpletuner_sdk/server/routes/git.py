"""Git integration routes for configuration versioning."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

from simpletuner.simpletuner_sdk.server.services.git_config_service import GIT_CONFIG_SERVICE, GitConfigError
from simpletuner.simpletuner_sdk.server.services.git_repo_service import GIT_REPO_SERVICE, GitRepoError

router = APIRouter(prefix="/api/git", tags=["git"])


class GitInitRequest(BaseModel):
    remote: Optional[str] = None
    branch: Optional[str] = None


class GitRemoteRequest(BaseModel):
    remote: Optional[str] = None


class GitBranchRequest(BaseModel):
    name: str
    create: bool = False


class GitSnapshotRequest(BaseModel):
    paths: List[str]
    message: Optional[str] = None
    include_untracked: bool = False
    push: bool = False
    config_type: str = "model"


class GitRevertRequest(BaseModel):
    path: str
    commit: Optional[str] = None
    config_type: str = "model"


class GitRemoteActionRequest(BaseModel):
    allow_remote: bool = False
    remote: Optional[str] = None
    branch: Optional[str] = None


class GitIdentityRequest(BaseModel):
    name: str
    email: str


def _config_dir(config_type: str = "model") -> str:
    store = GIT_CONFIG_SERVICE._get_store(config_type)  # noqa: SLF001 - shared path resolution
    return str(store.config_dir)


def _handle_repo_call(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except GitRepoError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)


def _handle_config_call(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except GitConfigError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except GitRepoError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)


@router.get("/status")
async def git_status(config_type: str = "model") -> Dict[str, Any]:
    """Return git status scoped to the configs directory."""
    status_obj = _handle_repo_call(GIT_REPO_SERVICE.discover_repo, _config_dir(config_type))
    return {
        "git_available": status_obj.git_available,
        "repo_present": status_obj.repo_present,
        "repo_root": status_obj.repo_root,
        "branch": status_obj.branch,
        "remote": status_obj.remote,
        "dirty_paths": status_obj.dirty_paths or [],
        "ahead": status_obj.ahead,
        "behind": status_obj.behind,
        "user_name": status_obj.user_name,
        "user_email": status_obj.user_email,
        "identity_configured": status_obj.identity_configured,
        "config_dir": _config_dir(config_type),
    }


@router.post("/init")
async def git_init(payload: GitInitRequest, config_type: str = "model") -> Dict[str, Any]:
    _handle_repo_call(
        GIT_REPO_SERVICE.init_repo,
        _config_dir(config_type),
        remote=payload.remote,
        branch=payload.branch,
    )
    return await git_status(config_type)


@router.post("/remote")
async def git_remote(payload: GitRemoteRequest, config_type: str = "model") -> Dict[str, Any]:
    _handle_repo_call(GIT_REPO_SERVICE.set_remote, _config_dir(config_type), payload.remote)
    return await git_status(config_type)


@router.post("/branch")
async def git_branch(payload: GitBranchRequest, config_type: str = "model") -> Dict[str, Any]:
    _handle_repo_call(
        GIT_REPO_SERVICE.create_or_switch_branch,
        _config_dir(config_type),
        payload.name,
        payload.create,
    )
    return await git_status(config_type)


@router.post("/identity")
async def git_identity(payload: GitIdentityRequest, config_type: str = "model") -> Dict[str, Any]:
    _handle_repo_call(GIT_REPO_SERVICE.set_identity, _config_dir(config_type), payload.name, payload.email)
    return await git_status(config_type)


@router.get("/history")
async def git_history(
    path: str = Query(..., description="Config name or path"),
    config_type: str = "model",
    skip: int = 0,
    limit: int = 20,
) -> Dict[str, Any]:
    commits = _handle_config_call(GIT_CONFIG_SERVICE.history_for_config, path, config_type, skip, limit)
    return {"commits": commits}


@router.get("/diff")
async def git_diff(
    path: str = Query(..., description="Config name or path"),
    commit: Optional[str] = None,
    config_type: str = "model",
    max_bytes: int = 100_000,
    max_lines: int = 500,
) -> Dict[str, Any]:
    diff_payload = _handle_config_call(GIT_CONFIG_SERVICE.diff_for_config, path, config_type, commit, max_bytes, max_lines)
    return diff_payload


@router.post("/snapshot")
async def git_snapshot(payload: GitSnapshotRequest, config_type: Optional[str] = None) -> Dict[str, Any]:
    target_config_type = config_type or payload.config_type or "model"
    message = (payload.message or "").strip() or "Snapshot from SimpleTuner"
    result = _handle_config_call(
        GIT_CONFIG_SERVICE.snapshot_configs,
        payload.paths,
        message,
        target_config_type,
        payload.include_untracked,
        payload.push,
    )
    return result


@router.post("/revert")
async def git_revert(payload: GitRevertRequest, config_type: Optional[str] = None) -> Dict[str, Any]:
    target_config_type = config_type or payload.config_type or "model"
    return _handle_config_call(
        GIT_CONFIG_SERVICE.restore_config,
        payload.path,
        target_config_type,
        payload.commit,
    )


@router.post("/push")
async def git_push(payload: GitRemoteActionRequest, config_type: str = "model") -> Dict[str, Any]:
    if not payload.allow_remote:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Remote actions require explicit opt-in.",
        )
    _handle_repo_call(GIT_REPO_SERVICE.push, _config_dir(config_type), payload.remote, payload.branch)
    return await git_status(config_type)


@router.post("/pull")
async def git_pull(payload: GitRemoteActionRequest, config_type: str = "model") -> Dict[str, Any]:
    if not payload.allow_remote:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Remote actions require explicit opt-in.",
        )
    _handle_repo_call(GIT_REPO_SERVICE.pull, _config_dir(config_type), payload.remote, payload.branch)
    return await git_status(config_type)
