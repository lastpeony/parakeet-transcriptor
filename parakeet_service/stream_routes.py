from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from parakeet_service.streaming_vad import StreamingVAD
# Make sure this import path matches your project structure
from parakeet_service.batchworker import transcription_queue, connection_queues
import asyncio
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

    vad = StreamingVAD()

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
            # Client is gone, but in the gather() design the consumer is still
            # parked on my_queue.get(), so gather() never returns and the finally
            # cleanup below never runs. Watch for the ABSENCE of a matching CLOSE
            # line and a heartbeat 'active_connections' that never drops.
            logger.info(
                "DISCONNECT %s | producer saw client disconnect (cleanup pending on gather)",
                connection_id[:8],
            )

    async def consumer():

        my_queue = connection_queues[connection_id]
        while True:
            result_text = await my_queue.get()
            await ws.send_json({"text": result_text})
            my_queue.task_done()

    try:
        await asyncio.gather(producer(), consumer())
    finally:

        connection_queues.pop(connection_id, None)
        logger.info(
            "CLOSE %s | cleaned_up | active=%d | transcription_queue=%d",
            connection_id[:8], len(connection_queues), transcription_queue.qsize(),
        )