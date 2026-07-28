import logging
from datetime import datetime, timezone

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
    Call this after collecting the employee's name, ID, and meal preference.

    Args:
        name: Full name of the employee.
        employee_id: Employee ID (e.g. NXB-0042).
        preference: Chosen meal plan — Lunch, Dinner, Both, or Roti Only.

    Returns:
        Confirmation string with thread_id for reply tracking.
    """
    body = (
        f"Dear Meals Coordinator,<br><br>"
        f"An employee has submitted a meal subscription request via the NXB internal chatbot.<br><br>"
        f"&nbsp;&nbsp;Full Name&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: {name}<br>"
        f"&nbsp;&nbsp;Employee ID&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: {employee_id}<br>"
        f"&nbsp;&nbsp;Subscription Type : {preference}<br>"
        f"&nbsp;&nbsp;Request Time&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}<br><br>"
        f"Please process this request and reply to this email to confirm activation.<br><br>"
        f"Regards,<br>NXB Chatbot System"
    )

    _send().invoke({
        "to": [settings.MEAL_DEPARTMENT_EMAIL],
        "subject": MEAL_EMAIL_SUBJECT,
        "message": body,
    })

    logger.info(f"Meal subscription email sent for {name} ({employee_id})")

    # Retrieve thread_id so we can track the reply later
    thread_id: str | None = None
    try:
        results = _search().invoke({
            "query": f'in:sent subject:"{MEAL_EMAIL_SUBJECT}"',
            "resource": "messages",
            "max_results": 1,
        })
        if isinstance(results, list) and results:
            thread_id = results[0].get("threadId")
    except Exception as exc:
        logger.warning(f"Could not retrieve thread_id: {exc}")

    return f"Email sent successfully. thread_id={thread_id}"


@tool
def check_meal_reply(thread_id: str) -> str:
    """
    Checks if the meals department has replied to the subscription email.
    Use the thread_id returned by send_meal_subscription_email.

    Args:
        thread_id: Gmail thread ID of the original subscription email.

    Returns:
        The reply body if found, or a message saying no reply yet.
    """
    try:
        thread_msgs = _thread().invoke({"thread_id": thread_id})
        if isinstance(thread_msgs, list) and len(thread_msgs) > 1:
            reply = thread_msgs[-1]
            body = reply.get("body") or reply.get("snippet", "")
            logger.info(f"Reply found in thread {thread_id}")
            return body
    except Exception as exc:
        logger.warning(f"Thread lookup failed: {exc}")

    # Fallback — search inbox for any reply from department
    try:
        results = _search().invoke({
            "query": f"from:{settings.MEAL_DEPARTMENT_EMAIL} subject:Re:",
            "resource": "messages",
            "max_results": 1,
        })
        if isinstance(results, list) and results:
            return results[0].get("body") or results[0].get("snippet", "Reply found.")
    except Exception as exc:
        logger.warning(f"Search fallback failed: {exc}")

    return "NO_REPLY"


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