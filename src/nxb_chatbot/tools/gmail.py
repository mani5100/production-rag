import logging
from datetime import datetime, timezone
from uuid import uuid4

from google.oauth2.credentials import Credentials
from langchain_community.tools.gmail.get_thread import GmailGetThread
from langchain_community.tools.gmail.search import GmailSearch
from langchain_community.tools.gmail.send_message import GmailSendMessage
from langchain_community.tools.gmail.utils import build_resource_service
from langchain_core.tools import tool

from nxb_chatbot.core.config import settings

logger = logging.getLogger(__name__)

GMAIL_SCOPES = ["https://mail.google.com/"]
MEAL_EMAIL_SUBJECT = "Meal Subscription Request — NXB Chatbot"


def _get_credentials() -> Credentials:
    return Credentials(
        token=None,
        refresh_token=settings.GMAIL_REFRESH_TOKEN,
        token_uri=settings.GMAIL_TOKEN_URI,
        client_id=settings.GMAIL_CLIENT_ID,
        client_secret=settings.GMAIL_CLIENT_SECRET,
        scopes=GMAIL_SCOPES,
    )


def _get_api_resource():
    return build_resource_service(credentials=_get_credentials())


# ---------------------------------------------------------------------------
# Raw LangChain tool instances (used internally by @tool functions below)
# ---------------------------------------------------------------------------

def _send() -> GmailSendMessage:
    return GmailSendMessage(api_resource=_get_api_resource())


def _search() -> GmailSearch:
    return GmailSearch(api_resource=_get_api_resource())


def _thread() -> GmailGetThread:
    return GmailGetThread(api_resource=_get_api_resource())


# ---------------------------------------------------------------------------
# @tool decorated functions — these are what nodes and the LLM call
# ---------------------------------------------------------------------------

@tool
def send_meal_subscription_email(
    name: str,
    employee_id: str,
    preference: str,
) -> str:
    """
    Sends a meal subscription request email to the meals department.

    A unique tracking reference is added to the subject so the exact sent
    message can be found without matching an older meal request.

    Args:
        name: Full name of the employee.
        employee_id: Employee ID, for example NXB-0042.
        preference: Lunch, Dinner, Both, or Roti Only.

    Returns:
        Confirmation string containing the Gmail thread_id.
    """
    request_reference = uuid4().hex

    subject = f"{MEAL_EMAIL_SUBJECT} [{request_reference}]"

    body = (
        f"Dear Meals Coordinator,<br><br>"
        f"An employee has submitted a meal subscription request via the "
        f"NXB internal chatbot.<br><br>"
        f"&nbsp;&nbsp;Full Name"
        f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: {name}<br>"
        f"&nbsp;&nbsp;Employee ID"
        f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: {employee_id}<br>"
        f"&nbsp;&nbsp;Subscription Type : {preference}<br>"
        f"&nbsp;&nbsp;Request Reference&nbsp;: {request_reference}<br>"
        f"&nbsp;&nbsp;Request Time"
        f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}<br><br>"
        f"Please process this request and reply to this email to confirm "
        f"activation.<br><br>"
        f"Regards,<br>NXB Chatbot System"
    )

    _send().invoke(
        {
            "to": [settings.MEAL_DEPARTMENT_EMAIL],
            "subject": subject,
            "message": body,
        }
    )

    logger.info(
        "Meal subscription email sent for %s (%s), reference=%s",
        name,
        employee_id,
        request_reference,
    )

    thread_id: str | None = None

    try:
        results = _search().invoke(
            {
                # The unique reference prevents an older email from matching.
                "query": f'in:sent subject:"{request_reference}"',
                "resource": "messages",
                "max_results": 1,
            }
        )

        if isinstance(results, list) and results:
            thread_id = results[0].get("threadId")

    except Exception as exc:
        logger.warning(
            "Could not retrieve thread_id for request %s: %s",
            request_reference,
            exc,
        )

    return (
        f"Email sent successfully. "
        f"thread_id={thread_id}; "
        f"request_reference={request_reference}"
    )


@tool
def check_meal_reply(thread_id: str) -> str:
    """
    Checks only the Gmail thread associated with the current meal request.

    Args:
        thread_id: Gmail thread ID of the current subscription email.

    Returns:
        Reply body when a reply exists, otherwise NO_REPLY.
    """
    if not thread_id or thread_id.lower() == "none":
        logger.warning("Meal reply check skipped because thread_id is missing.")
        return "TRACKING_UNAVAILABLE"

    try:
        thread_data = _thread().invoke({"thread_id": thread_id})
        thread_msgs = (
            thread_data.get("messages", [])
            if isinstance(thread_data, dict)
            else thread_data
        )

        if not isinstance(thread_msgs, list) or len(thread_msgs) <= 1:
            return "NO_REPLY"

        # The first message is the outgoing subscription request.
        # Inspect later messages only.
        for message in reversed(thread_msgs[1:]):
            sender = str(
                message.get("from")
                or message.get("sender")
                or message.get("From")
                or ""
            ).lower()

            # Some Gmail tool responses may not expose the sender field.
            # When it is available, require the configured department sender.
            if sender and settings.MEAL_DEPARTMENT_EMAIL.lower() not in sender:
                continue

            body = message.get("body") or message.get("snippet", "")

            if body:
                logger.info("Reply found in meal thread %s", thread_id)
                return body

        return "NO_REPLY"

    except Exception as exc:
        logger.warning(
            "Meal thread lookup failed for thread_id=%s: %s",
            thread_id,
            exc,
        )
        return "TRACKING_UNAVAILABLE"


@tool
def send_meal_acknowledgment(name: str, employee_id: str) -> str:
    """
    Sends an acknowledgment reply to the meals department after they respond.
    Only call this after the user confirms they want to acknowledge.

    Args:
        name: Full name of the employee.
        employee_id: Employee ID.

    Returns:
        Confirmation that the acknowledgment was sent.
    """
    body = (
        f"Dear Meals Coordinator,<br><br>"
        f"Thank you for your response regarding the meal subscription request "
        f"for {name} (ID: {employee_id}).<br><br>"
        f"We acknowledge receipt of your reply and will act accordingly.<br><br>"
        f"Regards,<br>NXB Chatbot System"
    )

    _send().invoke({
        "to": [settings.MEAL_DEPARTMENT_EMAIL],
        "subject": f"Re: {MEAL_EMAIL_SUBJECT}",
        "message": body,
    })

    logger.info(f"Acknowledgment sent for {name} ({employee_id})")
    return "Acknowledgment email sent successfully."