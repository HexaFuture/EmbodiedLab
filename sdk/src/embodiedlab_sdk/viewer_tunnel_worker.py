"""Worker-initiated private Viewer tunnel client.

The Worker owns all local Viser addressing. The Controller only sends an
environment ID and opaque HTTP/WebSocket traffic, never an endpoint or port.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .managed_runtime import ManagedRuntimeClient, ManagedWorkerIdentity
from .viewer import (
    LocalViewerTransport,
    ViewerTransportUnavailable,
    ViewerTunnelLimits,
    _from_wire_data,
    _wire_data,
    sanitize_viewer_response_headers,
)


class WorkerViewerTunnelClient:
    def __init__(
        self,
        controller_url: str,
        worker_id: str,
        api_key: str,
        transport: LocalViewerTransport,
        *,
        limits: ViewerTunnelLimits | None = None,
        tunnel_url: str | None = None,
    ) -> None:
        self.controller_url = controller_url.rstrip("/")
        self.worker_id = worker_id
        self.api_key = api_key
        self.transport = transport
        self.limits = limits or ViewerTunnelLimits()
        self._direct_tunnel_url = tunnel_url
        self._stopped = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._websocket: Any | None = None
        self._send_lock: asyncio.Lock | None = None
        self._streams: dict[str, Any] = {}

    def start(self) -> threading.Thread:
        thread = threading.Thread(target=self.run_forever, name="embodiedlab-viewer-tunnel", daemon=True)
        thread.start()
        return thread

    def stop(self) -> None:
        self._stopped.set()
        if self._loop is not None and self._websocket is not None:
            asyncio.run_coroutine_threadsafe(self._websocket.close(), self._loop)

    def run_forever(self) -> None:
        while not self._stopped.is_set():
            try:
                self.run_once()
            except Exception:
                if self._stopped.wait(1.0):
                    return

    def run_once(self) -> None:
        asyncio.run(self._run_once())

    def _tunnel_url(self) -> str:
        if self._direct_tunnel_url:
            parsed_direct = urlsplit(self._direct_tunnel_url)
            if parsed_direct.scheme not in {"ws", "wss"} or not parsed_direct.netloc:
                raise ValueError("tunnel_url must be an absolute WS(S) URL")
            return self._direct_tunnel_url
        parsed = urlsplit(self.controller_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("controller_url must be an absolute HTTP(S) URL")
        scheme = "wss" if parsed.scheme == "https" else "ws"
        path = parsed.path.rstrip("/") + f"/internal/v1/workers/{self.worker_id}/viewer-tunnel"
        return urlunsplit((scheme, parsed.netloc, path, "", ""))

    async def _run_once(self) -> None:
        try:
            from websockets.asyncio.client import connect
        except ImportError as exc:  # pragma: no cover - packaging gate
            raise RuntimeError("websockets is required for Worker Viewer tunnels") from exc
        self._loop = asyncio.get_running_loop()
        try:
            async with connect(
                self._tunnel_url(),
                additional_headers={"Authorization": f"Bearer {self.api_key}"},
                compression=None,
                max_size=self.limits.max_websocket_frame_bytes,
                ping_interval=20,
                ping_timeout=60,
            ) as websocket:
                self._websocket = websocket
                self._send_lock = asyncio.Lock()
                async for raw in websocket:
                    if not isinstance(raw, str):
                        continue
                    try:
                        payload = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict):
                        await self._dispatch(payload)
        finally:
            streams = list(self._streams.values())
            self._streams.clear()
            for stream in streams:
                try:
                    await stream.close()
                except Exception:
                    pass
            self._websocket = None
            self._send_lock = None
            self._loop = None

    async def _send(self, payload: dict[str, Any]) -> None:
        if self._websocket is None or self._send_lock is None:
            raise ViewerTransportUnavailable("viewer_tunnel_unavailable")
        async with self._send_lock:
            await self._websocket.send(json.dumps(payload, separators=(",", ":")))

    async def _dispatch(self, payload: dict[str, Any]) -> None:
        kind = payload.get("type")
        request_id = payload.get("id")
        environment_id = payload.get("environment_id")
        if not isinstance(request_id, str):
            return
        if kind == "http_request" and isinstance(environment_id, str):
            asyncio.create_task(self._handle_http(request_id, environment_id, payload))
        elif kind == "websocket_open" and isinstance(environment_id, str):
            asyncio.create_task(self._handle_websocket_open(request_id, environment_id, payload))
        elif kind == "websocket_frame":
            stream = self._streams.get(request_id)
            if stream is None:
                return
            try:
                frame = _from_wire_data(payload, limit=self.limits.max_websocket_frame_bytes)
                await stream.send(frame)
            except Exception:
                await self._close_stream(request_id)
        elif kind == "websocket_close":
            await self._close_stream(request_id)

    async def _handle_http(
        self, request_id: str, environment_id: str, payload: dict[str, Any]
    ) -> None:
        try:
            path, query, method = self._request_target(payload)
            response = await self.transport.fetch_viewer_http(environment_id, path, query, method)
            encoded = _wire_data(response.content, limit=self.limits.max_http_body_bytes)
            await self._send(
                {
                    "type": "http_response",
                    "id": request_id,
                    "status_code": response.status_code,
                    "media_type": response.media_type,
                    "headers": sanitize_viewer_response_headers(response.headers),
                    **encoded,
                }
            )
        except Exception:
            await self._send(
                {
                    "type": "http_response",
                    "id": request_id,
                    "status_code": 503,
                    "media_type": "text/plain",
                    "headers": {"cache-control": "no-store"},
                    **_wire_data(b"viewer unavailable", limit=self.limits.max_http_body_bytes),
                }
            )

    async def _handle_websocket_open(
        self, request_id: str, environment_id: str, payload: dict[str, Any]
    ) -> None:
        if len(self._streams) >= self.limits.max_streams_per_worker:
            await self._send({"type": "websocket_close", "id": request_id})
            return
        try:
            path, query, _method = self._request_target(payload)
            subprotocols = payload.get("subprotocols", [])
            if not isinstance(subprotocols, list) or not all(
                isinstance(value, str) and value for value in subprotocols
            ):
                raise ViewerTransportUnavailable("viewer_protocol_invalid")
            stream = await self.transport.connect_viewer_websocket(
                environment_id, path, query, subprotocols
            )
            self._streams[request_id] = stream
            await self._send(
                {
                    "type": "websocket_opened",
                    "id": request_id,
                    "subprotocol": getattr(stream, "subprotocol", None),
                }
            )
            asyncio.create_task(self._pump_websocket(request_id, stream))
        except Exception:
            await self._send({"type": "websocket_close", "id": request_id})

    def _request_target(self, payload: dict[str, Any]) -> tuple[str, str, str]:
        path = payload.get("path")
        query = payload.get("query")
        method = payload.get("method", "GET")
        if (
            not isinstance(path, str)
            or not isinstance(query, str)
            or not isinstance(method, str)
            or any(part in {".", ".."} for part in path.split("/"))
            or method not in {"GET", "HEAD"}
        ):
            raise ViewerTransportUnavailable("viewer_protocol_invalid")
        return path, query, method

    async def _pump_websocket(self, request_id: str, stream: Any) -> None:
        try:
            while True:
                message = await stream.receive()
                if message is None:
                    return
                await self._send(
                    {
                        "type": "websocket_frame",
                        "id": request_id,
                        **_wire_data(message, limit=self.limits.max_websocket_frame_bytes),
                    }
                )
        except Exception:
            pass
        finally:
            await self._close_stream(request_id)

    async def _close_stream(self, request_id: str) -> None:
        stream = self._streams.pop(request_id, None)
        if stream is not None:
            try:
                await stream.close()
            except Exception:
                pass
        try:
            await self._send({"type": "websocket_close", "id": request_id})
        except Exception:
            pass


class ManagedViewerTunnelService:
    """Refresh Sim-issued credentials before each outbound tunnel connection."""

    def __init__(
        self,
        runtime_client: ManagedRuntimeClient,
        identity: ManagedWorkerIdentity,
        gateway_url: str,
        transport: LocalViewerTransport,
        *,
        limits: ViewerTunnelLimits | None = None,
        reconnect_delay_seconds: float = 1.0,
    ) -> None:
        self.runtime_client = runtime_client
        self.identity = identity
        self.gateway_url = gateway_url.rstrip("/")
        self.transport = transport
        self.limits = limits
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self._stopped = threading.Event()
        self._active: WorkerViewerTunnelClient | None = None

    def start(self) -> threading.Thread:
        thread = threading.Thread(
            target=self.run_forever,
            name="embodiedlab-managed-viewer-tunnel",
            daemon=True,
        )
        thread.start()
        return thread

    def stop(self) -> None:
        self._stopped.set()
        if self._active is not None:
            self._active.stop()

    def run_forever(self) -> None:
        while not self._stopped.is_set():
            tunnel: WorkerViewerTunnelClient | None = None
            try:
                credentials = self.runtime_client.viewer_credentials(self.identity)
                tunnel = WorkerViewerTunnelClient(
                    self.gateway_url,
                    self.identity.worker_id,
                    credentials.ticket,
                    self.transport,
                    limits=self.limits,
                    tunnel_url=_gateway_tunnel_url(self.gateway_url, credentials.tunnel_path),
                )
                self._active = tunnel
                tunnel.run_once()
            except Exception:
                pass
            finally:
                if self._active is tunnel:
                    self._active = None
            if self._stopped.wait(self.reconnect_delay_seconds):
                return


def _gateway_tunnel_url(gateway_url: str, tunnel_path: str) -> str:
    parsed = urlsplit(gateway_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("gateway_url must be an absolute HTTP(S) URL")
    if not tunnel_path.startswith("/internal/v1/tunnels/"):
        raise ValueError("invalid managed Viewer tunnel path")
    return urlunsplit(
        (
            "wss" if parsed.scheme == "https" else "ws",
            parsed.netloc,
            parsed.path.rstrip("/") + tunnel_path,
            "",
            "",
        )
    )
