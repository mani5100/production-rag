import json
import os
from io import BytesIO
import wave

os.environ.pop("DATABASE_URL", None)

import chainlit as cl
import httpx


API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://localhost:8000/api/v1",
)

STT_BASE_URL = os.getenv(
    "STT_BASE_URL",
    "http://localhost:6000",
)

TTS_WEBSOCKET_URL = os.getenv(
    "TTS_WEBSOCKET_URL",
    "ws://localhost:5000/ws/speech",
)


@cl.on_chat_start
async def on_chat_start() -> None:
    cl.user_session.set("backend_session_id", None)
    cl.user_session.set("audio_buffer", None)
    cl.user_session.set("audio_mime_type", None)

    await cl.Message(
        content=(
            "Hello! I am your Production RAG Assistant.\n\n"
            "You can type a question or use the microphone."
        )
    ).send()


async def process_user_text(user_text: str) -> None:
    """Send typed or transcribed text to the RAG backend."""

    session_id = cl.user_session.get(
        "backend_session_id"
    )

    payload = {
        "message": user_text,
        "session_id": session_id,
        "retrieval_filters": None,
    }

    response_message = cl.Message(content="")
    await response_message.send()

    received_token = False
    full_answer = ""

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,
                read=180.0,
                write=30.0,
                pool=10.0,
            )
        ) as client:
            async with client.stream(
                "POST",
                f"{API_BASE_URL}/chat/stream",
                json=payload,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    event = json.loads(line)
                    event_type = event.get("type")

                    if event_type == "session":
                        backend_session_id = event.get(
                            "session_id"
                        )

                        if backend_session_id:
                            cl.user_session.set(
                                "backend_session_id",
                                backend_session_id,
                            )

                    elif event_type == "token":
                        content = event.get("content", "")

                        if content:
                            received_token = True
                            full_answer += content

                            await response_message.stream_token(
                                content
                            )

                    elif event_type == "done":
                        backend_session_id = event.get(
                            "session_id"
                        )

                        if backend_session_id:
                            cl.user_session.set(
                                "backend_session_id",
                                backend_session_id,
                            )

                        if event.get("web_search_used"):
                            await response_message.stream_token(
                                "\n\n🌐 _Web search was used._"
                            )

                    elif event_type == "error":
                        error_message = event.get(
                            "message",
                            "Unknown backend error",
                        )

                        await response_message.stream_token(
                            f"\n\nBackend error: `{error_message}`"
                        )

        if not received_token:
            await response_message.stream_token(
                "The workflow completed but returned no streamed answer."
            )

        await response_message.update()

        if full_answer.strip():
            tts_element = cl.CustomElement(
                name="StreamingTTS",
                props={
                    "text": full_answer,
                    "websocketUrl": TTS_WEBSOCKET_URL,
                },
                display="inline",
            )

            await cl.Message(
                content="",
                elements=[tts_element],
            ).send()

    except httpx.ConnectError:
        response_message.content = (
            "I could not connect to the FastAPI backend."
        )
        await response_message.update()

    except httpx.TimeoutException:
        response_message.content = (
            "The FastAPI stream timed out."
        )
        await response_message.update()

    except httpx.HTTPStatusError as exc:
        response_message.content = (
            f"FastAPI returned status "
            f"{exc.response.status_code}: "
            f"{exc.response.text}"
        )
        await response_message.update()

    except json.JSONDecodeError:
        response_message.content = (
            "FastAPI returned an invalid streaming event."
        )
        await response_message.update()

    except Exception as exc:
        response_message.content = (
            f"Unexpected streaming error: `{exc}`"
        )
        await response_message.update()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Handle normal typed messages."""

    user_text = message.content.strip()

    if not user_text:
        return

    await process_user_text(user_text)

@cl.on_audio_start
async def on_audio_start() -> bool:
    """Initialize raw PCM audio collection."""

    cl.user_session.set("audio_buffer", BytesIO())
    cl.user_session.set("audio_sample_rate", 24000)
    cl.user_session.set("audio_channels", 1)
    cl.user_session.set("audio_sample_width", 2)

    return True

@cl.on_audio_chunk
async def on_audio_chunk(
    chunk: cl.InputAudioChunk,
) -> None:
    """Collect raw PCM chunks sent by Chainlit."""

    audio_buffer = cl.user_session.get("audio_buffer")

    if audio_buffer is None:
        audio_buffer = BytesIO()
        cl.user_session.set("audio_buffer", audio_buffer)

    audio_buffer.write(chunk.data)


def pcm_to_wav(
    pcm_bytes: bytes,
    sample_rate: int,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    """Wrap raw PCM bytes in a valid WAV container."""

    wav_buffer = BytesIO()

    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)

    return wav_buffer.getvalue()


@cl.on_audio_end
async def on_audio_end() -> None:
    """Convert recorded PCM to WAV, then send it to STT."""

    audio_buffer = cl.user_session.get("audio_buffer")

    sample_rate = cl.user_session.get(
        "audio_sample_rate"
    ) or 24000

    channels = cl.user_session.get(
        "audio_channels"
    ) or 1

    sample_width = cl.user_session.get(
        "audio_sample_width"
    ) or 2

    cl.user_session.set("audio_buffer", None)

    if audio_buffer is None:
        await cl.Message(
            content="No microphone audio was captured."
        ).send()
        return

    pcm_bytes = audio_buffer.getvalue()
    audio_buffer.close()

    if not pcm_bytes:
        await cl.Message(
            content="The microphone recording was empty."
        ).send()
        return

    wav_bytes = pcm_to_wav(
        pcm_bytes=pcm_bytes,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
    )

    status_message = cl.Message(
        content="Transcribing your audio..."
    )
    await status_message.send()

    files = {
        "file": (
            "recording.wav",
            wav_bytes,
            "audio/wav",
        )
    }

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,
                read=180.0,
                write=60.0,
                pool=10.0,
            )
        ) as client:
            response = await client.post(
                f"{STT_BASE_URL}/v1/audio/transcriptions",
                files=files,
            )

            response.raise_for_status()
            result = response.json()

        transcript = result.get("text", "").strip()

        if not transcript:
            status_message.content = (
                "No speech was detected in the recording."
            )
            await status_message.update()
            return

        status_message.content = (
            f"🎤 **You said:** {transcript}"
        )
        await status_message.update()

        await process_user_text(transcript)

    except httpx.HTTPStatusError as exc:
        status_message.content = (
            f"STT returned status {exc.response.status_code}:\n\n"
            f"`{exc.response.text}`"
        )
        await status_message.update()

    except httpx.ConnectError:
        status_message.content = (
            "Could not connect to the STT service on port 6000."
        )
        await status_message.update()

    except httpx.TimeoutException:
        status_message.content = (
            "Speech transcription timed out."
        )
        await status_message.update()

    except Exception as exc:
        status_message.content = (
            f"Speech transcription failed: `{exc}`"
        )
        await status_message.update()

def get_audio_extension(mime_type: str) -> str:
    """Map browser MIME types to appropriate file extensions."""

    normalized = mime_type.lower()

    if "webm" in normalized:
        return ".webm"

    if "wav" in normalized:
        return ".wav"

    if "ogg" in normalized:
        return ".ogg"

    if "mp4" in normalized or "m4a" in normalized:
        return ".m4a"

    if "mpeg" in normalized or "mp3" in normalized:
        return ".mp3"

    return ".webm"