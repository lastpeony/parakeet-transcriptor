"""Web-side bridge to the GPU inference process.

Inference itself runs in a separate process (see inference_proc.py) so it can
never block the event loop. This module just holds the per-connection result
queues and the manager handle, and exposes thin helpers used by the routes.
"""
import asyncio

from parakeet_service.inference_proc import InferenceManager

# Per-connection result queues, drained by each WebSocket consumer.
connection_queues: dict[str, asyncio.Queue] = {}

# Set by the app lifespan once the inference process is up.
manager: InferenceManager | None = None


def submit_chunk(connection_id: str, path: str) -> None:
    """Hand a VAD-flushed audio chunk to the inference process (non-blocking)."""
    if manager is not None:
        manager.submit_stream(connection_id, path)


def pending_requests() -> int:
    """Approximate depth of the inference request queue (-1 if unavailable)."""
    return manager.pending() if manager is not None else 0
