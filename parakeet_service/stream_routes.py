from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from parakeet_service.streaming_vad import StreamingVAD
# Make sure this import path matches your project structure
from parakeet_service.batchworker import transcription_queue, connection_queues
import asyncio
import uuid

router = APIRouter()

@router.websocket("/ws")
async def ws_asr(ws: WebSocket):

    await ws.accept()

    connection_id = str(uuid.uuid4())

    connection_queues[connection_id] = asyncio.Queue()

    vad = StreamingVAD()

    async def producer():

        try:
            while True:
                frame = await ws.receive_bytes()
                for chunk in vad.feed(frame):
                    tagged_chunk = (connection_id, chunk)
                    await transcription_queue.put(tagged_chunk)
                    await ws.send_json({"status": "queued"})
        except WebSocketDisconnect:
            pass

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
        print(f"Connection {connection_id} closed and cleaned up.")