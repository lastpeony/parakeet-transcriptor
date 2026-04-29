from __future__ import annotations
import io, wave, tempfile, numpy as np, torch
from typing import List
from torch.hub import load as torch_hub_load
from concurrent.futures import ThreadPoolExecutor
import asyncio
from parakeet_service.config import VAD_THRESHOLD

# Thread pool for CPU-bound VAD operations
_vad_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="vad")

# Load VADIterator class once (shared), but each instance gets its own model
_, _vad_utils = torch_hub_load("snakers4/silero-vad", "silero_vad")
(_, _, _, VADIterator, _) = _vad_utils


def _load_vad_model():
    """Load a fresh VAD model instance (CPU only, ~2MB)."""
    model, _ = torch_hub_load("snakers4/silero-vad", "silero_vad")
    return model

SAMPLE_RATE              = 16_000         # model is trained for 16 kHz
WINDOW_SAMPLES           = 512            # 32 ms frame
THRESHOLD                = VAD_THRESHOLD  # voice prob threshold — set via VAD_THRESHOLD env var
MIN_SILENCE_MS           = 150            # flush after ≥150 ms quiet
SPEECH_PAD_MS            = 120            # keep 120 ms context before/after
MAX_SPEECH_MS            = 8_000          # hard stop at 8 s
PERIODIC_FLUSH_MS        = 6_000          # flush mid-speech if no silence event by 6 s

# Helper: float32 → int16 PCM bytes
def _f32_to_pcm16(frames: np.ndarray) -> bytes:
    return np.clip(frames * 32768, -32768, 32767).astype(np.int16).tobytes()

class StreamingVAD:
    """
    Feed successive 20–40 ms PCM frames (16 kHz, int16 mono).
    Emits temp-file *paths* when a full utterance is detected.
    """

    def __init__(self):
        # Each instance gets its own VAD model (thread-safe)
        self._vad_model = _load_vad_model()
        self.vad = VADIterator(
            self._vad_model,
            sampling_rate=SAMPLE_RATE,
            threshold=THRESHOLD,
            min_silence_duration_ms=MIN_SILENCE_MS,
            speech_pad_ms=SPEECH_PAD_MS,
        )
        self.buffer = bytearray()
        self.speech_ms = 0


    def _flush(self) -> List[str]:
        if not self.buffer:
            return []
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        with wave.open(tmp, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(self.buffer)
        self.buffer.clear()
        self.speech_ms = 0
        self.vad.reset_states()
        return [tmp.name]

    def feed(self, frame_bytes: bytes) -> List[str]:
        out: List[str] = []

        pcm_f32 = np.frombuffer(frame_bytes, np.int16).astype("float32") / 32768
        for start in range(0, len(pcm_f32), WINDOW_SAMPLES):
            window = pcm_f32[start:start + WINDOW_SAMPLES]
            if len(window) < WINDOW_SAMPLES:
                break  # wait for full 32 ms window

            voice_event = self.vad(window, return_seconds=False)
            self.buffer.extend(_f32_to_pcm16(window))
            self.speech_ms += 32

            # Flush on trailing-silence event or periodic guard
            if voice_event and voice_event.get("end"):
                out.extend(self._flush())
            elif self.speech_ms >= PERIODIC_FLUSH_MS:
                out.extend(self._flush())

        return out

    async def feed_async(self, frame_bytes: bytes) -> List[str]:
        """Async wrapper that runs VAD in thread pool to avoid blocking event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_vad_executor, self.feed, frame_bytes)
