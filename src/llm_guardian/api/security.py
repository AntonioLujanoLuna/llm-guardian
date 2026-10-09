"""Optional service-token enforcement for the ``/v1`` routes.

The service is gateway-local, but a stray local client should not be able to drive it when the
operator has set a token. This is not authentication of end users -- user authorization is the
gateway's and ``policy.py``'s job -- it only keeps the guardian's own surface closed.
"""

from __future__ import annotations

import hmac
from typing import Annotated

from fastapi import Header, HTTPException, status

from ..deps import SettingsDep


def require_service_token(
    settings: SettingsDep,
    x_guardian_token: Annotated[str | None, Header()] = None,
) -> None:
    """Enforce ``server.service_token`` when one is configured; no-op when it is not."""
    expected = settings.server.service_token
    if expected is None:
        return
    presented = x_guardian_token or ""
    if not hmac.compare_digest(presented, expected.get_secret_value()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or invalid X-Guardian-Token",
        )
