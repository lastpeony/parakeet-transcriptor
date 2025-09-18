import asyncio, contextlib, logging, tempfile, pathlib, time, torch
from typing import Union, List, Tuple
from parakeet_service import model as mdl

logger = logging.getLogger("batcher")
logger.setLevel(logging.DEBUG)


transcription_queue: asyncio.Queue[Tuple[str, Union[str, bytes]]] = asyncio.Queue()


connection_queues: dict[str, asyncio.Queue] = {}


def _as_path(data: Union[str, bytes]) -> str:

    if isinstance(data, str):
        return data
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(data)
        return f.name

async def batch_worker(model, batch_ms: float = 15.0, max_batch: int = 4):

    logger.info("worker started (batch ≤%d, window %.0f ms)", max_batch, batch_ms)
    logger.info("worker started with model id=%s", id(model))

    while True:
        connection_id, chunk = await transcription_queue.get()
        file_path = _as_path(chunk)

        batch: List[Tuple[str, str]] = [(connection_id, file_path)]

        deadline = time.monotonic() + batch_ms / 1000
        while len(batch) < max_batch:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                break
            try:
                nxt_connection_id, nxt_chunk = await asyncio.wait_for(transcription_queue.get(), timeout)
                nxt_file_path = _as_path(nxt_chunk)
                batch.append((nxt_connection_id, nxt_file_path))
            except asyncio.TimeoutError:
                break

        logger.debug("processing %d-file batch", len(batch))

        file_paths_for_model = [fp for _, fp in batch]
        try:
            with torch.inference_mode():
                outs = model.transcribe(file_paths_for_model, batch_size=len(file_paths_for_model))
        except Exception as exc:
            logger.exception("ASR failed: %s", exc)
            for _ in batch:
                transcription_queue.task_done()
            continue

        # --- Distribute results to connection queues ---
        for (conn_id, _), result in zip(batch, outs):
            text = getattr(result, "text", str(result))


            if conn_id in connection_queues:

                await connection_queues[conn_id].put(text)
            else:
                logger.warning(f"Connection ID {conn_id} not found. Client likely disconnected. Discarding result.")

            transcription_queue.task_done()

        for _, fp_to_delete in batch:
            with contextlib.suppress(FileNotFoundError):
                pathlib.Path(fp_to_delete).unlink(missing_ok=True)