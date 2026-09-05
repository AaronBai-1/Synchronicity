"""Shared FastAPI dependencies: DB session + workspace scoping.

Workspace scoping is header-based for now: X-Workspace-Id selects the tenant, defaulting
to a single-tenant "default" workspace so local dev needs no setup. The production plan
(docs/plan.md "Platform") replaces this with Clerk auth — the Clerk org id becomes the
workspace id and a signature-verified JWT replaces the bare header — without changing
any route signature, since everything downstream only sees a Workspace row.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header
from sqlalchemy.orm import Session

from synchro_api.db import get_session
from synchro_api.models import Workspace

DEFAULT_WORKSPACE_ID = "default"

SessionDep = Annotated[Session, Depends(get_session)]


def get_workspace(
    session: SessionDep,
    x_workspace_id: Annotated[str | None, Header()] = None,
) -> Workspace:
    """Resolve (and lazily create) the request's workspace.

    Creation-on-first-sight is a dev convenience; with Clerk, workspaces are provisioned
    by an org webhook and an unknown id becomes a 401 instead.
    """
    workspace_id = x_workspace_id or DEFAULT_WORKSPACE_ID
    workspace = session.get(Workspace, workspace_id)
    if workspace is None:
        workspace = Workspace(id=workspace_id, name=workspace_id)
        session.add(workspace)
        session.flush()
    return workspace


WorkspaceDep = Annotated[Workspace, Depends(get_workspace)]
