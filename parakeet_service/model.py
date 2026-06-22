from contextlib import asynccontextmanager
import logging
import torch, asyncio

from .config import NUM_THREADS, logger

from parakeet_service import batchworker as bw
from parakeet_service.batchworker import connection_queues, pending_requests
from parakeet_service.inference_proc import InferenceManager

# Dedicated logger with an explicit level: the shared `parakeet_service` logger
# inherits the root WARNING level (see config.py), which would swallow these
# INFO heartbeats.
hb_logger = logging.getLogger("heartbeat")
hb_logger.setLevel(logging.INFO)

try:
    import psutil
    _proc = psutil.Process()
except Exception:
    _proc = None


def _rss_mb() -> str:
    if _proc is None:
        return "n/a (install psutil)"
    return f"{_proc.memory_info().rss / (1024 * 1024):.1f} MB"


async def _memory_heartbeat(interval_s: float = 30.0):
    """Periodically log active-connection count, queue depth, and process RSS.

    A steadily climbing 'active' count or RSS while no clients are connected is
    the direct signature of the connection leak.
    """
    while True:
        await asyncio.sleep(interval_s)
        hb_logger.info(
            "HEARTBEAT | active_connections=%d | inference_queue=%d | rss=%s",
            len(connection_queues), pending_requests(), _rss_mb(),
        )


@asynccontextmanager
async def lifespan(app):
    """Start the GPU inference process; the web process never touches CUDA."""
    # CPU threading for VAD/audio preprocessing (this process is CPU-only).
    torch.set_num_threads(NUM_THREADS)
    logger.info("CPU threading: %d threads", torch.get_num_threads())

    loop = asyncio.get_running_loop()
    bw.manager = InferenceManager()
    logger.info("Starting inference process (loading model)...")
    bw.manager.start(loop, connection_queues)
    app.state.inference = bw.manager
    logger.info("Inference process ready")

    app.state.heartbeat = asyncio.create_task(_memory_heartbeat(), name="memory_heartbeat")
    hb_logger.info("memory_heartbeat scheduled")

    try:
        yield
    finally:
        app.state.heartbeat.cancel()
        try:
            await app.state.heartbeat
        except asyncio.CancelledError:
            pass

        logger.info("Stopping inference process")
        bw.manager.stop()
        bw.manager = None
