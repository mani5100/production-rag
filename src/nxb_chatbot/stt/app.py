import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from faster_whisper import WhisperModel


MODEL_NAME = os.getenv(
    "WHISPER_MODEL",
    "small.en",
)

DEVICE = os.getenv(
    "WHISPER_DEVICE",
    "cpu",
)

COMPUTE_TYPE = os.getenv(
    "WHISPER_COMPUTE_TYPE",
    "int8",
)

model: WhisperModel | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    model = WhisperModel(
        MODEL_NAME,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
    )

    yield

    model = None


app = FastAPI(
    title="Local Faster-Whisper STT",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok" if model is not None else "not_ready",
        "model_loaded": model is not None,
        "model_name": MODEL_NAME,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
    }


@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...),
) -> dict[str, str]:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Speech-to-text model is not ready.",
        )

    suffix = Path(file.filename or "audio.webm").suffix or ".webm"

    temporary_file = tempfile.NamedTemporaryFile(
        suffix=suffix,
        delete=False,
    )

    temporary_path = Path(temporary_file.name)

    try:
        audio_bytes = await file.read()

        if not audio_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded audio is empty.",
            )

        temporary_file.write(audio_bytes)
        temporary_file.close()

        segments, _ = model.transcribe(
            str(temporary_path),
            language="en",
            beam_size=5,
            vad_filter=True,
        )

        transcript = " ".join(
            segment.text.strip()
            for segment in segments
            if segment.text.strip()
        ).strip()

        if not transcript:
            raise HTTPException(
                status_code=422,
                detail="No speech could be detected.",
            )

        return {
            "text": transcript,
        }

    finally:
        temporary_file.close()
        temporary_path.unlink(missing_ok=True)