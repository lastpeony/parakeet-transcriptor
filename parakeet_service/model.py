from contextlib import asynccontextmanager
import contextlib
import gc
import torch, asyncio
import nemo.collections.asr as nemo_asr
from omegaconf import open_dict

from .config import MODEL_NAME, NEMO_MODEL_PATH, MODEL_PRECISION, DEVICE, NUM_THREADS, logger

from parakeet_service.batchworker import batch_worker, transcription_queue, connection_queues

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
        logger.info(
            "HEARTBEAT | active_connections=%d | transcription_queue=%d | rss=%s",
            len(connection_queues), transcription_queue.qsize(), _rss_mb(),
        )


def _to_builtin(obj):
    """torch/NumPy → pure-Python (JSON-safe)."""
    import numpy as np
    import torch as th

    if isinstance(obj, (th.Tensor, np.ndarray)):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    return obj


@asynccontextmanager
async def lifespan(app):
    """Load model once per process; free GPU on shutdown."""
    dtype = torch.float16 if MODEL_PRECISION == "fp16" else torch.float32

    if NEMO_MODEL_PATH:
        logger.info("Loading model from local file: %s", NEMO_MODEL_PATH)
        with torch.inference_mode():
            model = nemo_asr.models.ASRModel.restore_from(
                NEMO_MODEL_PATH,
                map_location=DEVICE,
            ).to(dtype=dtype)
    else:
        logger.info("Downloading %s from pretrained hub...", MODEL_NAME)
        with torch.inference_mode():
            model = nemo_asr.models.ASRModel.from_pretrained(
                MODEL_NAME,
                map_location=DEVICE,
            ).to(dtype=dtype)

    logger.info("Loaded model with %s weights on %s", MODEL_PRECISION.upper(), DEVICE)
        
    # Aggressive cleanup
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Memory cleanup complete")

    # Configure CPU threading for VAD (after CUDA init)
    torch.set_num_threads(NUM_THREADS)
    logger.info("CPU threading: %d threads", torch.get_num_threads())

    app.state.asr_model = model
    logger.info("Model ready on %s", next(model.parameters()).device)

    app.state.worker = asyncio.create_task(batch_worker(model), name="batch_worker")
    logger.info("batch_worker scheduled")

    app.state.heartbeat = asyncio.create_task(_memory_heartbeat(), name="memory_heartbeat")
    logger.info("memory_heartbeat scheduled")

    try:
        yield
    finally:
        app.state.heartbeat.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await app.state.heartbeat

        app.state.worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await app.state.worker

        logger.info("Releasing GPU memory and shutting down worker")
        del app.state.asr_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # free cache but keep driver


def reset_fast_path(model):
    """Restore low-latency decoding flags."""
    with open_dict(model.cfg.decoding):
        if getattr(model.cfg.decoding, "compute_timestamps", False):
            model.cfg.decoding.compute_timestamps = False
        if getattr(model.cfg.decoding, "preserve_alignments", False):
            model.cfg.decoding.preserve_alignments = False
    model.change_decoding_strategy(model.cfg.decoding)
