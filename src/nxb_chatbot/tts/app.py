import asyncio
import json
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from piper import PiperVoice


VOICE_MODEL = Path(
    "/app/src/nxb_chatbot/tts/voices/en_US-lessac-medium.onnx"
)

voice: PiperVoice | None = None

# Start with one synthesis at a time.
# You can revisit concurrency after testing Piper safely.
synthesis_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global voice

    if not VOICE_MODEL.exists():
        raise RuntimeError(
            f"Voice model not found: {VOICE_MODEL}"
        )

    voice = await asyncio.to_thread(
        PiperVoice.load,
        str(VOICE_MODEL),
    )

    yield

    voice = None


app = FastAPI(
    title="Streaming Piper TTS",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok" if voice is not None else "not_ready",
        "voice_loaded": voice is not None,
        "voice_model": str(VOICE_MODEL),
        "voice_exists": VOICE_MODEL.exists(),
    }


def get_next_chunk(iterator: Iterator[Any]) -> Any | None:
    try:
        return next(iterator)
    except StopIteration:
        return None


@app.websocket("/ws/speech")
async def stream_speech(websocket: WebSocket) -> None:
    await websocket.accept()

    try:
        request_text = await websocket.receive_text()

        try:
            payload = json.loads(request_text)
        except json.JSONDecodeError:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": "Invalid JSON payload.",
                }
            )
            await websocket.close(code=1003)
            return

        text = str(payload.get("text", "")).strip()

        if not text:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": "Text cannot be empty.",
                }
            )
            await websocket.close(code=1008)
            return

        if voice is None:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": "Piper voice is not loaded.",
                }
            )
            await websocket.close(code=1011)
            return

        await websocket.send_json(
            {
                "type": "started",
            }
        )

        async with synthesis_lock:
            chunk_iterator = voice.synthesize(text)
            first_chunk = True

            while True:
                chunk = await asyncio.to_thread(
                    get_next_chunk,
                    chunk_iterator,
                )

                if chunk is None:
                    break

                if first_chunk:
                    await websocket.send_json(
                        {
                            "type": "format",
                            "sample_rate": chunk.sample_rate,
                            "sample_width": chunk.sample_width,
                            "channels": chunk.sample_channels,
                        }
                    )
                    first_chunk = False

                await websocket.send_bytes(
                    chunk.audio_int16_bytes
                )

        await websocket.send_json(
            {
                "type": "done",
            }
        )

        await websocket.close(code=1000)

    except WebSocketDisconnect:
        return

    except Exception as exc:
        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": str(exc),
                }
            )
            await websocket.close(code=1011)
        except Exception:
            pass