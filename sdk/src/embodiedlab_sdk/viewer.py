"""Private Viewer transport contracts implemented by SDK Worker runtimes."""

from __future__ import annotations

import asyncio
import base64
import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .models import Environment


@dataclass(frozen=True)
class ViewerHTTPResponse:
    status_code: int
    content: bytes = b""
    media_type: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)


class ViewerWebSocket(Protocol):
    async def receive(self) -> str | bytes | None: ...

    async def send(self, message: str | bytes) -> None: ...

    async def close(self) -> None: ...


class ViewerTransport(Protocol):
    """Private SDK Worker channel; it must not expose an endpoint to callers."""

    async def fetch_http(
        self,
        environment: Environment,
        path: str,
        query: str,
        method: str,
    ) -> ViewerHTTPResponse: ...

    async def connect_websocket(
        self, environment: Environment, path: str, query: str, subprotocols: list[str]
    ) -> ViewerWebSocket: ...


@runtime_checkable
class LocalViewerTransport(Protocol):
    """Worker-local Viewer access. Implementations must only target loopback/container IPC."""

    async def fetch_viewer_http(
        self, environment_id: str, path: str, query: str, method: str
    ) -> ViewerHTTPResponse: ...

    async def connect_viewer_websocket(
        self, environment_id: str, path: str, query: str, subprotocols: list[str]
    ) -> ViewerWebSocket: ...


class ViewerTransportUnavailable(RuntimeError):
    """The Controller cannot reach a ready private Worker Viewer channel."""


_UNSAFE_VIEWER_RESPONSE_HEADERS = frozenset(
    {
        "connection",
        "content-encoding",
        "content-length",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)


def sanitize_viewer_response_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Drop transport headers invalid after relaying or decoding a response body."""

    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in _UNSAFE_VIEWER_RESPONSE_HEADERS
    }


@dataclass(frozen=True)
class ViewerTunnelLimits:
    max_http_body_bytes: int = 16 * 1024 * 1024
    max_websocket_frame_bytes: int = 16 * 1024 * 1024
    max_streams_per_worker: int = 16
    request_timeout_seconds: float = 15.0


def _wire_data(value: str | bytes, *, limit: int) -> dict[str, str]:
    if isinstance(value, str):
        if len(value.encode("utf-8")) > limit:
            raise ViewerTransportUnavailable("viewer_frame_too_large")
        return {"encoding": "text", "data": value}
    if len(value) > limit:
        raise ViewerTransportUnavailable("viewer_frame_too_large")
    return {"encoding": "base64", "data": base64.b64encode(value).decode("ascii")}


def _from_wire_data(value: Mapping[str, Any], *, limit: int) -> str | bytes:
    encoding = value.get("encoding")
    data = value.get("data")
    if not isinstance(data, str):
        raise ViewerTransportUnavailable("viewer_protocol_invalid")
    if encoding == "text":
        if len(data.encode("utf-8")) > limit:
            raise ViewerTransportUnavailable("viewer_frame_too_large")
        return data
    if encoding == "base64":
        try:
            decoded = base64.b64decode(data, validate=True)
        except ValueError as exc:
            raise ViewerTransportUnavailable("viewer_protocol_invalid") from exc
        if len(decoded) > limit:
            raise ViewerTransportUnavailable("viewer_frame_too_large")
        return decoded
    raise ViewerTransportUnavailable("viewer_protocol_invalid")


class TunnelViewerWebSocket:
    """Controller-side Viewer socket backed by a Worker-initiated tunnel."""

    def __init__(self, connection: WorkerViewerTunnel, stream_id: str) -> None:
        self._connection = connection
        self._stream_id = stream_id
        self._incoming: asyncio.Queue[str | bytes | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        self._opened: asyncio.Future[str | None] = loop.create_future()
        self._closed = False
        self.subprotocol: str | None = None

    async def wait_open(self) -> None:
        self.subprotocol = await asyncio.wait_for(
            self._opened, timeout=self._connection.limits.request_timeout_seconds
        )

    async def receive(self) -> str | bytes | None:
        return await self._incoming.get()

    async def send(self, message: str | bytes) -> None:
        if self._closed:
            return
        payload = _wire_data(message, limit=self._connection.limits.max_websocket_frame_bytes)
        await self._connection.send({"type": "websocket_frame", "id": self._stream_id, **payload})

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._connection.close_stream(self._stream_id)

    def mark_open(self, subprotocol: str | None) -> None:
        if not self._opened.done():
            self._opened.set_result(subprotocol)

    def mark_failed(self) -> None:
        if not self._opened.done():
            self._opened.set_exception(ViewerTransportUnavailable("viewer_tunnel_unavailable"))
        self._incoming.put_nowait(None)

    def deliver(self, value: str | bytes | None) -> None:
        self._incoming.put_nowait(value)


class WorkerViewerTunnel:
    """One authenticated, Worker-initiated multiplexed Viewer connection."""

    def __init__(self, worker_id: str, websocket: Any, limits: ViewerTunnelLimits) -> None:
        self.worker_id = worker_id
        self.websocket = websocket
        self.limits = limits
        self._send_lock = asyncio.Lock()
        self._pending_http: dict[str, asyncio.Future[ViewerHTTPResponse]] = {}
        self._streams: dict[str, TunnelViewerWebSocket] = {}
        self._closed = False

    async def send(self, payload: Mapping[str, Any]) -> None:
        if self._closed:
            raise ViewerTransportUnavailable("viewer_tunnel_unavailable")
        try:
            async with self._send_lock:
                await self.websocket.send_json(dict(payload))
        except Exception as exc:
            self.close()
            raise ViewerTransportUnavailable("viewer_tunnel_unavailable") from exc

    async def fetch_http(
        self, environment: Environment, path: str, query: str, method: str
    ) -> ViewerHTTPResponse:
        request_id = secrets.token_urlsafe(18)
        future: asyncio.Future[ViewerHTTPResponse] = asyncio.get_running_loop().create_future()
        self._pending_http[request_id] = future
        try:
            await self.send(
                {
                    "type": "http_request",
                    "id": request_id,
                    "environment_id": environment.id,
                    "path": path,
                    "query": query,
                    "method": method,
                }
            )
            return await asyncio.wait_for(future, timeout=self.limits.request_timeout_seconds)
        finally:
            self._pending_http.pop(request_id, None)

    async def connect_websocket(
        self, environment: Environment, path: str, query: str, subprotocols: list[str]
    ) -> ViewerWebSocket:
        if len(self._streams) >= self.limits.max_streams_per_worker:
            raise ViewerTransportUnavailable("viewer_tunnel_stream_limit")
        stream_id = secrets.token_urlsafe(18)
        stream = TunnelViewerWebSocket(self, stream_id)
        self._streams[stream_id] = stream
        try:
            await self.send(
                {
                    "type": "websocket_open",
                    "id": stream_id,
                    "environment_id": environment.id,
                    "path": path,
                    "query": query,
                    "subprotocols": subprotocols,
                }
            )
            await stream.wait_open()
            return stream
        except Exception:
            self._streams.pop(stream_id, None)
            await stream.close()
            raise

    async def close_stream(self, stream_id: str) -> None:
        self._streams.pop(stream_id, None)
        if not self._closed:
            try:
                await self.send({"type": "websocket_close", "id": stream_id})
            except ViewerTransportUnavailable:
                pass

    def receive(self, payload: Mapping[str, Any]) -> None:
        kind = payload.get("type")
        request_id = payload.get("id")
        if not isinstance(request_id, str):
            return
        if kind == "http_response":
            future = self._pending_http.get(request_id)
            if future is None or future.done():
                return
            try:
                content = _from_wire_data(payload, limit=self.limits.max_http_body_bytes)
                if not isinstance(content, bytes):
                    raise ViewerTransportUnavailable("viewer_protocol_invalid")
                status_code = payload.get("status_code")
                if not isinstance(status_code, int) or not 100 <= status_code <= 599:
                    raise ViewerTransportUnavailable("viewer_protocol_invalid")
                headers = payload.get("headers", {})
                if not isinstance(headers, dict) or not all(
                    isinstance(key, str) and isinstance(item, str)
                    for key, item in headers.items()
                ):
                    raise ViewerTransportUnavailable("viewer_protocol_invalid")
                media_type = payload.get("media_type")
                if media_type is not None and not isinstance(media_type, str):
                    raise ViewerTransportUnavailable("viewer_protocol_invalid")
                future.set_result(
                    ViewerHTTPResponse(
                        status_code=status_code,
                        content=content,
                        media_type=media_type,
                        headers=headers,
                    )
                )
            except ViewerTransportUnavailable as exc:
                future.set_exception(exc)
            return
        stream = self._streams.get(request_id)
        if stream is None:
            return
        if kind == "websocket_opened":
            selected = payload.get("subprotocol")
            stream.mark_open(selected if isinstance(selected, str) else None)
        elif kind == "websocket_frame":
            try:
                frame = _from_wire_data(payload, limit=self.limits.max_websocket_frame_bytes)
                stream.deliver(frame)
            except ViewerTransportUnavailable:
                stream.mark_failed()
        elif kind == "websocket_close":
            self._streams.pop(request_id, None)
            stream.deliver(None)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for future in self._pending_http.values():
            if not future.done():
                future.set_exception(ViewerTransportUnavailable("viewer_tunnel_unavailable"))
        self._pending_http.clear()
        for stream in self._streams.values():
            stream.mark_failed()
        self._streams.clear()


class WorkerViewerTunnelRegistry:
    """Routes SDK Controller Viewer requests only through registered Worker tunnels."""

    def __init__(self, limits: ViewerTunnelLimits | None = None) -> None:
        self.limits = limits or ViewerTunnelLimits()
        self._connections: dict[str, WorkerViewerTunnel] = {}
        self._lock = asyncio.Lock()

    async def register(self, worker_id: str, websocket: Any) -> WorkerViewerTunnel:
        connection = WorkerViewerTunnel(worker_id, websocket, self.limits)
        async with self._lock:
            previous = self._connections.get(worker_id)
            self._connections[worker_id] = connection
        if previous is not None:
            previous.close()
        return connection

    async def unregister(self, connection: WorkerViewerTunnel) -> None:
        connection.close()
        async with self._lock:
            if self._connections.get(connection.worker_id) is connection:
                self._connections.pop(connection.worker_id, None)

    async def _connection_for(self, environment: Environment) -> WorkerViewerTunnel:
        if not environment.worker_id:
            raise ViewerTransportUnavailable("viewer_tunnel_unavailable")
        async with self._lock:
            connection = self._connections.get(environment.worker_id)
        if connection is None:
            raise ViewerTransportUnavailable("viewer_tunnel_unavailable")
        return connection

    async def fetch_http(
        self, environment: Environment, path: str, query: str, method: str
    ) -> ViewerHTTPResponse:
        return await (await self._connection_for(environment)).fetch_http(
            environment, path, query, method
        )

    async def connect_websocket(
        self, environment: Environment, path: str, query: str, subprotocols: list[str]
    ) -> ViewerWebSocket:
        return await (await self._connection_for(environment)).connect_websocket(
            environment, path, query, subprotocols
        )
