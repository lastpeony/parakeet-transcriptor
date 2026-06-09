"""GPU inference in a dedicated process.

The web server runs on a single asyncio event loop. Running ``model.transcribe``
there — even via a thread pool — can still freeze the loop, because CPython's
GIL is process-wide: while NeMo's Python-level decoding holds the GIL, the loop
thread cannot run, so WebSocket keepalive pongs are delayed and clients drop
with "keepalive ping timeout".

A separate *process* has its own interpreter and its own GIL, so inference can
never block the web server's loop regardless of core count. This module owns
that process (the sole holder of the GPU model) plus the web-side manager that
talks to it over multiprocessing queues.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import pathlib
import queue
import threading
import time
import uuid
import wave
from typing import Optional

logger = logging.getLogger("inference")
logger.setLevel(logging.INFO)


def _wav_duration_s(path: str) -> float:
    try:
        with wave.open(path, "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0


# --------------------------------------------------------------------------- #
# Worker process (separate interpreter, owns the GPU model)
# --------------------------------------------------------------------------- #
def inference_worker(request_q: "mp.Queue", result_q: "mp.Queue", ready_evt) -> None:
    """Process entry point. Loads the model, then serves requests forever.

    Request tuples:
        ("stream", conn_id, path)          -> ("stream", conn_id, text)
        ("job",  req_id, paths, timestamps)-> ("result", req_id, payload | {"error": str})
        ("cfg",  req_id)                   -> ("result", req_id, yaml_str)
        None  (sentinel)                   -> shut down
    """
    # Heavy imports happen only in the child so the parent never touches CUDA.
    import torch
    import nemo.collections.asr as nemo_asr
    from omegaconf import open_dict

    from parakeet_service.config import (
        MODEL_NAME, NEMO_MODEL_PATH, MODEL_PRECISION, DEVICE, NUM_THREADS,
        BATCH_SIZE, VAD_MIN_CHUNK_MS,
    )

    min_chunk_s = VAD_MIN_CHUNK_MS / 1000.0
    batch_ms = 15.0
    max_batch = max(1, BATCH_SIZE)

    def _to_builtin(obj):
        import numpy as np
        if isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.tolist()
        if isinstance(obj, (list, tuple)):
            return [_to_builtin(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _to_builtin(v) for k, v in obj.items()}
        return obj

    def _reset_fast_path(model):
        with open_dict(model.cfg.decoding):
            if getattr(model.cfg.decoding, "compute_timestamps", False):
                model.cfg.decoding.compute_timestamps = False
            if getattr(model.cfg.decoding, "preserve_alignments", False):
                model.cfg.decoding.preserve_alignments = False
        model.change_decoding_strategy(model.cfg.decoding)

    try:
        dtype = torch.float16 if MODEL_PRECISION == "fp16" else torch.float32
        if NEMO_MODEL_PATH:
            logger.info("Loading model from local file: %s", NEMO_MODEL_PATH)
            with torch.inference_mode():
                model = nemo_asr.models.ASRModel.restore_from(
                    NEMO_MODEL_PATH, map_location=DEVICE,
                ).to(dtype=dtype)
        else:
            logger.info("Downloading %s from pretrained hub...", MODEL_NAME)
            with torch.inference_mode():
                model = nemo_asr.models.ASRModel.from_pretrained(
                    MODEL_NAME, map_location=DEVICE,
                ).to(dtype=dtype)
        torch.set_num_threads(NUM_THREADS)
        logger.info("inference process: model ready on %s (%s)",
                    next(model.parameters()).device, MODEL_PRECISION.upper())
    except Exception:
        logger.exception("inference process: model load FAILED")
        ready_evt.set()  # unblock parent so it can detect the dead process
        return

    ready_evt.set()

    def _run_stream_batch(batch):
        # batch: list of (conn_id, path); discard chunks too short to transcribe.
        good = []
        for conn_id, path in batch:
            if _wav_duration_s(path) < min_chunk_s:
                pathlib.Path(path).unlink(missing_ok=True)
                continue
            good.append((conn_id, path))
        if not good:
            return
        paths = [p for _, p in good]
        try:
            with torch.inference_mode():
                outs = model.transcribe(paths, batch_size=len(paths), verbose=False)
        except Exception as exc:
            logger.exception("stream ASR failed: %s", exc)
            outs = None
        if outs is not None:
            for (conn_id, _), res in zip(good, outs):
                text = getattr(res, "text", str(res)).strip()
                if not text:
                    continue
                logger.info("[%s] %s", conn_id[:8], text)
                result_q.put(("stream", conn_id, text))
        for _, path in good:
            pathlib.Path(path).unlink(missing_ok=True)

    def _run_job(req_id, paths, timestamps):
        try:
            with torch.inference_mode():
                outs = model.transcribe(paths, batch_size=2, timestamps=timestamps)
            if (not timestamps
                    and getattr(model.cfg.decoding, "compute_timestamps", False)):
                _reset_fast_path(model)
            if isinstance(outs, tuple):
                outs = outs[0]
            payload = []
            for h in outs:
                payload.append({
                    "text": getattr(h, "text", str(h)),
                    "timestamp": _to_builtin(getattr(h, "timestamp", {})) if timestamps else {},
                })
            result_q.put(("result", req_id, payload))
        except Exception as exc:
            logger.exception("job ASR failed: %s", exc)
            result_q.put(("result", req_id, {"error": str(exc)}))

    # A non-stream request pulled mid-batch is parked here and handled next loop.
    stash: list = []
    while True:
        req = stash.pop(0) if stash else request_q.get()
        if req is None:
            break
        kind = req[0]

        if kind == "stream":
            batch = [(req[1], req[2])]
            deadline = time.monotonic() + batch_ms / 1000.0
            while len(batch) < max_batch:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                try:
                    nxt = request_q.get(timeout=timeout)
                except queue.Empty:
                    break
                if nxt is None:
                    stash.append(None)
                    break
                if nxt[0] == "stream":
                    batch.append((nxt[1], nxt[2]))
                else:
                    stash.append(nxt)
                    break
            _run_stream_batch(batch)

        elif kind == "job":
            _run_job(req[1], req[2], req[3])

        elif kind == "cfg":
            try:
                from omegaconf import OmegaConf
                result_q.put(("result", req[1], OmegaConf.to_yaml(model.cfg, resolve=True)))
            except Exception as exc:
                result_q.put(("result", req[1], {"error": str(exc)}))

    logger.info("inference process: shutting down")


# --------------------------------------------------------------------------- #
# Web-side manager (runs in the FastAPI process)
# --------------------------------------------------------------------------- #
class InferenceManager:
    """Owns the inference subprocess and bridges it to the asyncio loop.

    Streaming results are pushed straight into per-connection queues. One-shot
    jobs (HTTP) are correlated back to an asyncio.Future by request id.
    """

    def __init__(self, ready_timeout_s: float = 600.0):
        self._ready_timeout_s = ready_timeout_s
        self._ctx = mp.get_context("spawn")  # required for CUDA in the child
        self._request_q: "mp.Queue" = self._ctx.Queue()
        self._result_q: "mp.Queue" = self._ctx.Queue()
        self._ready_evt = self._ctx.Event()
        self._proc: Optional[mp.Process] = None
        self._router: Optional[threading.Thread] = None
        self._loop = None
        self._connection_queues = None
        self._pending: dict = {}

    def start(self, loop, connection_queues: dict) -> None:
        """Spawn the worker, wait for the model to load, start the router thread.

        Called once at startup; blocking here is fine (no clients yet).
        """
        self._loop = loop
        self._connection_queues = connection_queues
        # Not daemonic: NeMo's transcribe may spawn DataLoader worker children,
        # which daemonic processes are forbidden from doing. We stop it
        # explicitly in the app lifespan instead.
        self._proc = self._ctx.Process(
            target=inference_worker,
            args=(self._request_q, self._result_q, self._ready_evt),
            name="inference_worker",
            daemon=False,
        )
        self._proc.start()

        if not self._ready_evt.wait(timeout=self._ready_timeout_s):
            raise RuntimeError("inference process did not become ready in time")
        if not self._proc.is_alive():
            raise RuntimeError("inference process exited during model load")

        self._router = threading.Thread(
            target=self._route_results, name="inference-router", daemon=True,
        )
        self._router.start()

    def _route_results(self) -> None:
        while True:
            try:
                msg = self._result_q.get()
            except (EOFError, OSError):
                break
            if msg is None or msg[0] == "__stop__":
                break
            tag = msg[0]
            if tag == "stream":
                _, conn_id, text = msg
                self._loop.call_soon_threadsafe(self._deliver_stream, conn_id, text)
            elif tag == "result":
                _, req_id, payload = msg
                self._loop.call_soon_threadsafe(self._deliver_job, req_id, payload)

    def _deliver_stream(self, conn_id: str, text: str) -> None:
        q = self._connection_queues.get(conn_id)
        if q is not None:
            q.put_nowait(text)
        else:
            logger.warning("connection %s gone, discarding: %s", conn_id[:8], text)

    def _deliver_job(self, req_id: str, payload) -> None:
        fut = self._pending.pop(req_id, None)
        if fut is not None and not fut.done():
            fut.set_result(payload)

    # ---- public API (called from the event loop) ---------------------------
    def submit_stream(self, conn_id: str, path: str) -> None:
        self._request_q.put(("stream", conn_id, path))

    async def transcribe_job(self, paths: list[str], timestamps: bool):
        req_id = uuid.uuid4().hex
        fut = self._loop.create_future()
        self._pending[req_id] = fut
        self._request_q.put(("job", req_id, paths, timestamps))
        return await fut

    async def get_cfg(self) -> str:
        req_id = uuid.uuid4().hex
        fut = self._loop.create_future()
        self._pending[req_id] = fut
        self._request_q.put(("cfg", req_id))
        return await fut

    def pending(self) -> int:
        try:
            return self._request_q.qsize()
        except (NotImplementedError, OSError):
            return -1

    def stop(self) -> None:
        if self._proc is None:
            return
        try:
            self._request_q.put(None)  # ask the worker to exit its loop
        except Exception:
            pass
        self._proc.join(timeout=10)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=5)
        try:
            self._result_q.put(("__stop__",))  # unblock the router thread
        except Exception:
            pass
        if self._router is not None:
            self._router.join(timeout=5)
