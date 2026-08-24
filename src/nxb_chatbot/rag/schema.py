from typing import Literal

from pydantic import BaseModel, Field


class GuardrailResult(BaseModel):
    passed: bool
    reason: str
    intent: Literal[
        "meal_subscription",
        "meal_status_check",
        "mis_request",
        "mis_status_check",
        "employee_request",
        "employee_request_status",
        "conversational",
        "general_query",
    ] = "general_query"


class EmployeeRequestDecision(BaseModel):
    """
    Structured decision produced by the LLM on every turn of the
    Leave / Work From Home workflow.
    """

    action: Literal[
        "ask_question",
        "request_confirmation",
        "send_request",
        "cancel_request",
        "show_submitted",
    ]

    request_type: Literal["Leave", "Work From Home"] | None = None

    employee_name: str | None = None
    employee_id: str | None = None

    # ISO format: YYYY-MM-DD
    start_date: str | None = None
    end_date: str | None = None

    reason: str | None = None

    # The LLM writes all user-facing responses/questions.
    response: str = Field(
        description=(
            "Natural response to show the employee. When asking a question, "
            "ask only one useful question at a time."
        )
    )


class EmployeeConfirmationDecision(BaseModel):
    action: Literal[
        "confirmed",
        "rejected",
        "correction",
        "unclear",
    ]

    response: str = Field(
        description="Natural response to show the employee."
    )


class GMAcknowledgementResult(BaseModel):
    """
    Context-aware acknowledgement generated from the GM's reply.
    """

    acknowledgement: str = Field(
        description=(
            "Professional email acknowledgement addressed to the General Manager."
        )
    )

    reply_summary: str = Field(
        description=(
            "A short explanation for the employee describing what the GM said."
        )
    )


class AcknowledgementConfirmationDecision(BaseModel):
    action: Literal[
        "send",
        "skip",
        "regenerate",
        "unclear",
    ]

    response: str
    
class GradeDocuments(BaseModel):
    """
    CRAG grading verdict: whether retrieved_docs actually answer the question.
    """

    verdict: Literal["relevant", "irrelevant"]
    reason: str = Field(
        description=(
            "Short, specific explanation for the verdict. Used to inform "
            "the query rewrite if the verdict is irrelevant."
        )
    )