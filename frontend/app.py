import os

os.environ.pop("DATABASE_URL", None)

import asyncio
import json
import chainlit as cl
import httpx


API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://localhost:8000/api/v1",
)

@cl.on_chat_start
async def on_chat_start() -> None:
    cl.user_session.set("backend_session_id", None)

    await cl.Message(
        content=(
            "Hello! I am your Production RAG Assistant.\n\n"
            "Ask me a question about the available documents."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Stream the FastAPI response directly into Chainlit."""

    session_id = cl.user_session.get(
        "backend_session_id"
    )

    payload = {
        "message": message.content,
        "session_id": session_id,
        "retrieval_filters": None,
    }

    response_message = cl.Message(content="")
    await response_message.send()

    received_token = False

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
                        cl.user_session.set(
                            "backend_session_id",
                            event["session_id"],
                        )

                    elif event_type == "token":
                        content = event.get("content", "")

                        if content:
                            received_token = True
                            await response_message.stream_token(
                                content
                            )

                    elif event_type == "done":
                        cl.user_session.set(
                            "backend_session_id",
                            event["session_id"],
                        )

                        if event.get("web_search_used"):
                            await response_message.stream_token(
                                "\n\n🌐 _Web search was used._"
                            )

                    elif event_type == "error":
                        error_message = event.get(
                            "message",
                            "Unknown streaming error",
                        )

                        await response_message.stream_token(
                            f"\n\nBackend error: `{error_message}`"
                        )

        if not received_token:
            await response_message.stream_token(
                "The workflow completed but returned "
                "no streamed answer."
            )

        await response_message.update()

    except httpx.ConnectError:
        response_message.content = (
            "I could not connect to FastAPI. "
            "Make sure it is running on port 8000."
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
            f"{exc.response.status_code}."
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