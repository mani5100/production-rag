from typing import Annotated

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    retrieved_docs: list[dict]
    retrieval_filters: dict | None
    standalone_query: str | None
    guardrail_passed: bool | None
    web_search_used: bool | None
    route_intent: str | None
    meal_intent: str | None
    meal_data: dict | None