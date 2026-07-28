from typing import Literal

from pydantic import BaseModel

class GuardrailResult(BaseModel):
    passed: bool
    reason: str
    intent: Literal[
        "meal_subscription",
        "meal_status_check",
        "general_query",
    ] = "general_query"