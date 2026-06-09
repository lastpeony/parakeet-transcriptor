from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from parakeet_service.streaming_vad import StreamingVAD
# Make sure this import path matches your project structure
from parakeet_service.batchworker import transcription_queue, connection_queues
import asyncio
import contextlib
import logging
import uuid

logger = logging.getLogger("stream")
logger.setLevel(logging.DEBUG)

router = APIRouter()

@router.websocket("/ws")
async def ws_asr(ws: WebSocket):

    await ws.accept()

    connection_id = str(uuid.uuid4())

    connection_queues[connection_id] = asyncio.Queue()

    vad = await StreamingVAD.create_async()

    logger.info(
        "OPEN  %s | active=%d | transcription_queue=%d",
        connection_id[:8], len(connection_queues), transcription_queue.qsize(),
    )

    async def producer():

        try:
            while True:
                frame = await ws.receive_bytes()
                for chunk in await vad.feed_async(frame):
                    tagged_chunk = (connection_id, chunk)
                    await transcription_queue.put(tagged_chunk)
                    await ws.send_json({"status": "queued"})
        except WebSocketDisconnect:
            logger.info("DISCONNECT %s | producer saw client disconnect", connection_id[:8])

    async def consumer():

        my_queue = connection_queues[connection_id]
        while True:
            result_text = await my_queue.get()
            await ws.send_json({"text": result_text})
            my_queue.task_done()

    producer_task = asyncio.create_task(producer())
    consumer_task = asyncio.create_task(consumer())
    try:
        # Wait for whichever task finishes first (producer returns on
        # WebSocketDisconnect; consumer only ends if a send fails). Cancel the
        # other so neither coroutine is left awaiting forever — otherwise the
        # consumer blocks on my_queue.get() and the finally cleanup never runs.
        done, pending = await asyncio.wait(
            {producer_task, consumer_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            which = "consumer" if task is consumer_task else "producer"
            logger.info("CANCEL %s | cancelling stuck %s task", connection_id[:8], which)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        # Surface any non-cancellation error from the completed task(s).
        for task in done:
            task.result()
    except WebSocketDisconnect:
        logger.info("DISCONNECT %s | outer handler", connection_id[:8])
    finally:
        removed = connection_queues.pop(connection_id, None)
        logger.info(
            "CLOSE %s | cleaned_up=%s | active=%d | transcription_queue=%d",
            connection_id[:8], removed is not None,
            len(connection_queues), transcription_queue.qsize(),
        )