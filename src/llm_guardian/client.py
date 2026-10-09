"""A typed client for the guardian's HTTP API, for gateways and executors that run out of process.

In-process integrations should use the FastAPI dependencies in ``deps.py`` instead; this is for the
other case, where the gateway is a separate service. Every method returns the service's own wire
models, so a caller gets validation and attribute access rather than dictionaries, and a non-2xx
answer raises :class:`GuardianError` carrying the status and the server's detail.

Requires the ``proxy`` extra (``httpx``)::

    from llm_guardian.client import GuardianClient

    with GuardianClient("http://guardian:8090", token="...") as guardian:
        verdict = guardian.scan_text(tool_output, provenance="tool_result", conversation_id=cid)
        if verdict.blocked:
            ...
        decision = guardian.authorize(principal="ci-agent", tool="run_tests", evidence=...)

:class:`AsyncGuardianClient` has the same methods as coroutines.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import httpx

from .models import (
    Authorization,
    EvidenceRef,
    GuardMode,
    HealthReport,
    IncidentCreateRequest,
    IncidentRecord,
    IncidentStatus,
    MetricsReport,
    PolicyAction,
    PolicyCheckRequest,
    PolicyCheckResponse,
    ReviewRequest,
    ScanRequest,
    ScanResponse,
    SegmentInput,
)
from .provenance import Provenance

__all__ = ["AsyncGuardianClient", "GuardianClient", "GuardianError"]


class GuardianError(RuntimeError):
    """A non-2xx answer from the guardian."""

    def __init__(self, status_code: int, detail: Any) -> None:
        super().__init__(f"guardian answered {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


# -- request builders shared by both clients ----------------------------------------------------
def _scan_body(
    segments: Sequence[SegmentInput],
    *,
    conversation_id: str | None,
    tenant: str | None,
    request_id: str | None,
    mode: GuardMode | None,
) -> dict[str, Any]:
    request = ScanRequest(
        request_id=request_id,
        conversation_id=conversation_id,
        tenant=tenant,
        segments=tuple(segments),
        mode=mode,
    )
    return request.model_dump(mode="json", exclude_none=True)


def _authorize_body(
    *,
    principal: str,
    tool: str,
    tool_arguments: dict[str, Any] | None,
    evidence: Sequence[EvidenceRef],
    tenant: str | None,
    requested_capabilities: Sequence[str],
) -> dict[str, Any]:
    request = PolicyCheckRequest(
        principal=principal,
        tenant=tenant,
        action=PolicyAction.TOOL_CALL,
        tool=tool,
        tool_arguments=dict(tool_arguments or {}),
        evidence=tuple(evidence),
        requested_capabilities=tuple(requested_capabilities),
    )
    return request.model_dump(mode="json", exclude_none=True)


def _list_params(
    status: IncidentStatus | None,
    tenant: str | None,
    provenance: Provenance | str | None,
    limit: int,
    offset: int,
) -> dict[str, Any]:
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    if status is not None:
        params["status"] = IncidentStatus(status).value
    if tenant is not None:
        params["tenant"] = tenant
    if provenance is not None:
        params["provenance"] = Provenance(provenance).value
    return params


def _checked(response: httpx.Response) -> Any:
    if response.is_success:
        return response.json()
    try:
        detail: Any = response.json()
    except ValueError:
        detail = response.text
    if isinstance(detail, dict) and "detail" in detail:
        detail = detail["detail"]
    raise GuardianError(response.status_code, detail)


def _headers(token: str | None) -> dict[str, str]:
    return {"X-Guardian-Token": token} if token else {}


# -- the clients --------------------------------------------------------------------------------
class GuardianClient:
    """Blocking client. Pass ``transport`` (e.g. an ``httpx.ASGITransport``) to test in-process."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        timeout: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=_headers(token),
            timeout=timeout,
            transport=transport,
        )

    def __enter__(self) -> GuardianClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def scan(
        self,
        segments: Sequence[SegmentInput],
        *,
        conversation_id: str | None = None,
        tenant: str | None = None,
        request_id: str | None = None,
        mode: GuardMode | None = None,
    ) -> ScanResponse:
        body = _scan_body(
            segments,
            conversation_id=conversation_id,
            tenant=tenant,
            request_id=request_id,
            mode=mode,
        )
        return ScanResponse.model_validate(_checked(self._http.post("/v1/scan", json=body)))

    def scan_text(
        self, text: str, *, provenance: Provenance | str = Provenance.USER, **kwargs: Any
    ) -> ScanResponse:
        segment = SegmentInput(text=text, provenance=Provenance(provenance))
        return self.scan((segment,), **kwargs)

    def authorize(
        self,
        *,
        principal: str,
        tool: str,
        tool_arguments: dict[str, Any] | None = None,
        evidence: Sequence[EvidenceRef] = (),
        tenant: str | None = None,
        requested_capabilities: Sequence[str] = (),
    ) -> Authorization:
        body = _authorize_body(
            principal=principal,
            tool=tool,
            tool_arguments=tool_arguments,
            evidence=evidence,
            tenant=tenant,
            requested_capabilities=requested_capabilities,
        )
        return Authorization.model_validate(
            _checked(self._http.post("/v1/agent/authorize", json=body))
        )

    def check_policy(self, request: PolicyCheckRequest) -> PolicyCheckResponse:
        body = request.model_dump(mode="json", exclude_none=True)
        return PolicyCheckResponse.model_validate(
            _checked(self._http.post("/v1/policy/check", json=body))
        )

    def create_incident(self, request: IncidentCreateRequest) -> IncidentRecord:
        body = request.model_dump(mode="json", exclude_none=True)
        return IncidentRecord.model_validate(_checked(self._http.post("/v1/incidents", json=body)))

    def review(self, case_id: str, request: ReviewRequest) -> IncidentRecord:
        body = request.model_dump(mode="json", exclude_none=True)
        response = self._http.post(f"/v1/incidents/{case_id}/review", json=body)
        return IncidentRecord.model_validate(_checked(response))

    def incidents(
        self,
        *,
        status: IncidentStatus | None = None,
        tenant: str | None = None,
        provenance: Provenance | str | None = None,
        page_size: int = 100,
    ) -> Iterator[IncidentRecord]:
        """Every matching incident, oldest first, fetched a page at a time."""
        offset = 0
        while True:
            params = _list_params(status, tenant, provenance, page_size, offset)
            page = _checked(self._http.get("/v1/incidents", params=params))
            yield from (IncidentRecord.model_validate(item) for item in page)
            if len(page) < page_size:
                return
            offset += page_size

    def health(self) -> HealthReport:
        return HealthReport.model_validate(_checked(self._http.get("/v1/health")))

    def metrics(self) -> MetricsReport:
        return MetricsReport.model_validate(_checked(self._http.get("/v1/metrics")))


class AsyncGuardianClient:
    """The same API as :class:`GuardianClient`, as coroutines."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        timeout: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=_headers(token),
            timeout=timeout,
            transport=transport,
        )

    async def __aenter__(self) -> AsyncGuardianClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def scan(
        self,
        segments: Sequence[SegmentInput],
        *,
        conversation_id: str | None = None,
        tenant: str | None = None,
        request_id: str | None = None,
        mode: GuardMode | None = None,
    ) -> ScanResponse:
        body = _scan_body(
            segments,
            conversation_id=conversation_id,
            tenant=tenant,
            request_id=request_id,
            mode=mode,
        )
        return ScanResponse.model_validate(_checked(await self._http.post("/v1/scan", json=body)))

    async def scan_text(
        self, text: str, *, provenance: Provenance | str = Provenance.USER, **kwargs: Any
    ) -> ScanResponse:
        segment = SegmentInput(text=text, provenance=Provenance(provenance))
        return await self.scan((segment,), **kwargs)

    async def authorize(
        self,
        *,
        principal: str,
        tool: str,
        tool_arguments: dict[str, Any] | None = None,
        evidence: Sequence[EvidenceRef] = (),
        tenant: str | None = None,
        requested_capabilities: Sequence[str] = (),
    ) -> Authorization:
        body = _authorize_body(
            principal=principal,
            tool=tool,
            tool_arguments=tool_arguments,
            evidence=evidence,
            tenant=tenant,
            requested_capabilities=requested_capabilities,
        )
        response = await self._http.post("/v1/agent/authorize", json=body)
        return Authorization.model_validate(_checked(response))

    async def review(self, case_id: str, request: ReviewRequest) -> IncidentRecord:
        body = request.model_dump(mode="json", exclude_none=True)
        response = await self._http.post(f"/v1/incidents/{case_id}/review", json=body)
        return IncidentRecord.model_validate(_checked(response))

    async def health(self) -> HealthReport:
        return HealthReport.model_validate(_checked(await self._http.get("/v1/health")))
