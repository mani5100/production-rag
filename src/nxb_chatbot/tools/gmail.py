import base64
from email.message import EmailMessage
import logging
from datetime import datetime, timezone
from uuid import uuid4
from html import escape

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
MIS_EMAIL_SUBJECT = "MIS Support Request — NXB Chatbot"
EMPLOYEE_REQUEST_EMAIL_SUBJECT = ("Employee Leave / Work From Home Request — NXB Chatbot")



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


@tool
def send_mis_request_email(
    issue_type: str,
    name: str,
    employee_id: str,
) -> str:
    """Sends an MIS support request and returns its tracking information."""
    request_reference = uuid4().hex
    subject = f"{MIS_EMAIL_SUBJECT} [{request_reference}]"

    body = (
        f"Dear MIS Team,<br><br>"
        f"An employee has submitted an MIS support request through the "
        f"NXB internal chatbot.<br><br>"
        f"&nbsp;&nbsp;Issue Type"
        f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: {issue_type}<br>"
        f"&nbsp;&nbsp;Full Name"
        f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: {name}<br>"
        f"&nbsp;&nbsp;Employee ID"
        f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: {employee_id}<br>"
        f"&nbsp;&nbsp;Request Reference: {request_reference}<br>"
        f"&nbsp;&nbsp;Request Time"
        f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}<br><br>"
        f"Please review this request and reply to this email with an update."
        f"<br><br>"
        f"Regards,<br>NXB Chatbot System"
    )

    _send().invoke(
        {
            "to": [settings.MIS_DEPARTMENT_EMAIL],
            "subject": subject,
            "message": body,
        }
    )

    thread_id: str | None = None

    try:
        results = _search().invoke(
            {
                "query": f'in:sent subject:"{request_reference}"',
                "resource": "messages",
                "max_results": 1,
            }
        )

        if isinstance(results, list) and results:
            thread_id = results[0].get("threadId")
    except Exception as exc:
        logger.warning(
            "Could not retrieve MIS thread_id for request %s: %s",
            request_reference,
            exc,
        )

    logger.info(
        "MIS request sent for %s (%s), reference=%s",
        name,
        employee_id,
        request_reference,
    )

    return (
        f"Email sent successfully. "
        f"thread_id={thread_id}; "
        f"request_reference={request_reference}"
    )


@tool
def check_mis_reply(thread_id: str) -> str:
    """Checks the Gmail thread belonging to the current MIS request."""
    if not thread_id or thread_id.lower() == "none":
        return "TRACKING_UNAVAILABLE"

    try:
        thread_data = _thread().invoke({"thread_id": thread_id})
        messages = (
            thread_data.get("messages", [])
            if isinstance(thread_data, dict)
            else thread_data
        )

        if not isinstance(messages, list) or len(messages) <= 1:
            return "NO_REPLY"

        for message in reversed(messages[1:]):
            sender = str(
                message.get("from")
                or message.get("sender")
                or message.get("From")
                or ""
            ).lower()

            if sender and settings.MIS_DEPARTMENT_EMAIL.lower() not in sender:
                continue

            body = message.get("body") or message.get("snippet", "")

            if body:
                logger.info("Reply found in MIS thread %s", thread_id)
                return body

        return "NO_REPLY"

    except Exception as exc:
        logger.warning(
            "MIS thread lookup failed for thread_id=%s: %s",
            thread_id,
            exc,
        )
        return "TRACKING_UNAVAILABLE"


@tool
def send_mis_acknowledgment(name: str, employee_id: str) -> str:
    """Sends an acknowledgment after the MIS department responds."""
    body = (
        f"Dear MIS Team,<br><br>"
        f"Thank you for your response regarding the MIS request for "
        f"{name} (ID: {employee_id}).<br><br>"
        f"We acknowledge receipt of your reply and will act accordingly."
        f"<br><br>"
        f"Regards,<br>NXB Chatbot System"
    )

    _send().invoke(
        {
            "to": [settings.MIS_DEPARTMENT_EMAIL],
            "subject": f"Re: {MIS_EMAIL_SUBJECT}",
            "message": body,
        }
    )

    logger.info("MIS acknowledgment sent for %s (%s)", name, employee_id)
    return "Acknowledgment email sent successfully."


@tool
def send_employee_request_to_gm(
    request_type: str,
    employee_name: str,
    employee_id: str,
    start_date: str,
    end_date: str,
    reason: str = "",
) -> str:
    """
    Sends a confirmed Leave or Work From Home request to the General Manager.

    Call this tool only after the employee explicitly confirms the final
    request summary.

    Args:
        request_type: Leave or Work From Home.
        employee_name: Full employee name.
        employee_id: Employee ID, for example NXB-0042.
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.
        reason: Optional employee-provided reason.

    Returns:
        A result containing thread_id and request_reference.
    """
    allowed_types = {"Leave", "Work From Home"}

    if request_type not in allowed_types:
        return "ERROR: Unsupported request type."

    request_reference = uuid4().hex
    subject = (
        f"{request_type} Request — "
        f"{employee_name} [{request_reference}]"
    )

    reason_section = ""
    if reason.strip():
        reason_section = (
            f"&nbsp;&nbsp;Reason"
            f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: "
            f"{escape(reason.strip())}<br>"
        )

    body = (
        f"Dear General Manager,<br><br>"
        f"An employee has submitted a "
        f"{escape(request_type)} request through the "
        f"NXB internal chatbot.<br><br>"
        f"&nbsp;&nbsp;Request Type"
        f"&nbsp;&nbsp;&nbsp;: {escape(request_type)}<br>"
        f"&nbsp;&nbsp;Full Name"
        f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: "
        f"{escape(employee_name)}<br>"
        f"&nbsp;&nbsp;Employee ID"
        f"&nbsp;&nbsp;&nbsp;&nbsp;: "
        f"{escape(employee_id)}<br>"
        f"&nbsp;&nbsp;Start Date"
        f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: {escape(start_date)}<br>"
        f"&nbsp;&nbsp;End Date"
        f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: "
        f"{escape(end_date)}<br>"
        f"{reason_section}"
        f"&nbsp;&nbsp;Reference"
        f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: "
        f"{request_reference}<br>"
        f"&nbsp;&nbsp;Submitted At"
        f"&nbsp;&nbsp;&nbsp;: "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        f"<br><br>"
        f"Please review this request and reply to this email with your "
        f"decision or any required clarification.<br><br>"
        f"Regards,<br>"
        f"NXB Chatbot System"
    )

    _send().invoke(
        {
            "to": [settings.GM_EMAIL],
            "subject": subject,
            "message": body,
        }
    )

    thread_id: str | None = None

    try:
        results = _search().invoke(
            {
                "query": f'in:sent subject:"{request_reference}"',
                "resource": "messages",
                "max_results": 1,
            }
        )

        if isinstance(results, list) and results:
            thread_id = results[0].get("threadId")

    except Exception as exc:
        logger.warning(
            "Could not retrieve GM request thread_id for reference %s: %s",
            request_reference,
            exc,
        )

    logger.info(
        "%s request sent to GM for %s (%s), reference=%s",
        request_type,
        employee_name,
        employee_id,
        request_reference,
    )

    return (
        "Email sent successfully. "
        f"thread_id={thread_id}; "
        f"request_reference={request_reference}"
    )
    
    
@tool
def check_gm_employee_request_reply(thread_id: str) -> str:
    """
    Checks the exact Gmail thread belonging to the employee's submitted
    Leave or Work From Home request.

    Args:
        thread_id: Gmail thread ID returned by
            send_employee_request_to_gm.

    Returns:
        The GM reply body, NO_REPLY, or TRACKING_UNAVAILABLE.
    """
    if not thread_id or thread_id.lower() == "none":
        logger.warning(
            "GM request reply check skipped because thread_id is missing."
        )
        return "TRACKING_UNAVAILABLE"

    try:
        thread_data = _thread().invoke({"thread_id": thread_id})

        messages = (
            thread_data.get("messages", [])
            if isinstance(thread_data, dict)
            else thread_data
        )

        if not isinstance(messages, list) or len(messages) <= 1:
            return "NO_REPLY"

        # First message is the outgoing request. Inspect replies only.
        for message in reversed(messages[1:]):
            sender = str(
                message.get("from")
                or message.get("sender")
                or message.get("From")
                or ""
            ).lower()

            if sender and settings.GM_EMAIL.lower() not in sender:
                continue

            body = message.get("body") or message.get("snippet", "")

            if body:
                logger.info(
                    "GM reply found in employee request thread %s",
                    thread_id,
                )
                return str(body)

        return "NO_REPLY"

    except Exception as exc:
        logger.warning(
            "GM thread lookup failed for thread_id=%s: %s",
            thread_id,
            exc,
        )
        return "TRACKING_UNAVAILABLE"
    
    
@tool
def send_gm_employee_request_acknowledgement(
    thread_id: str,
    acknowledgement: str,
) -> str:
    """
    Sends the employee-approved LLM-generated acknowledgement inside the
    original Gmail conversation.

    Call this only after:
    1. a GM response has been found;
    2. the LLM has generated an acknowledgement;
    3. the employee has confirmed sending it.

    Args:
        thread_id: Gmail thread ID of the original request.
        acknowledgement: Final acknowledgement written by the LLM.

    Returns:
        Confirmation or an error message.
    """
    if not thread_id or thread_id.lower() == "none":
        return "ERROR: Cannot send acknowledgement without a thread ID."

    if not acknowledgement.strip():
        return "ERROR: Acknowledgement content is empty."

    try:
        thread_data = _get_api_resource().users().threads().get(
            userId="me",
            id=thread_id,
            format="metadata",
            metadataHeaders=["Subject"],
        ).execute()

        original_subject = EMPLOYEE_REQUEST_EMAIL_SUBJECT

        messages = thread_data.get("messages", [])
        if messages:
            headers = messages[0].get("payload", {}).get("headers", [])

            for header in headers:
                if header.get("name", "").lower() == "subject":
                    original_subject = header.get(
                        "value",
                        EMPLOYEE_REQUEST_EMAIL_SUBJECT,
                    )
                    break

        if not original_subject.lower().startswith("re:"):
            subject = f"Re: {original_subject}"
        else:
            subject = original_subject

        email = EmailMessage()
        email["To"] = settings.GM_EMAIL
        email["Subject"] = subject
        email.set_content(acknowledgement)

        encoded_message = base64.urlsafe_b64encode(
            email.as_bytes()
        ).decode()

        _get_api_resource().users().messages().send(
            userId="me",
            body={
                "raw": encoded_message,
                "threadId": thread_id,
            },
        ).execute()

        logger.info(
            "LLM-generated acknowledgement sent in GM thread %s",
            thread_id,
        )

        return "Acknowledgement sent successfully."

    except Exception as exc:
        logger.exception(
            "Could not send acknowledgement in GM thread %s",
            thread_id,
        )
        return f"ERROR: Could not send acknowledgement: {exc}"
    
    
@tool
def send_gm_employee_request_acknowledgement(
    thread_id: str,
    acknowledgement: str,
) -> str:
    """
    Sends the employee-approved LLM-generated acknowledgement inside the
    original Gmail conversation.

    Call this only after:
    1. a GM response has been found;
    2. the LLM has generated an acknowledgement;
    3. the employee has confirmed sending it.

    Args:
        thread_id: Gmail thread ID of the original request.
        acknowledgement: Final acknowledgement written by the LLM.

    Returns:
        Confirmation or an error message.
    """
    if not thread_id or thread_id.lower() == "none":
        return "ERROR: Cannot send acknowledgement without a thread ID."

    if not acknowledgement.strip():
        return "ERROR: Acknowledgement content is empty."

    try:
        thread_data = _get_api_resource().users().threads().get(
            userId="me",
            id=thread_id,
            format="metadata",
            metadataHeaders=["Subject"],
        ).execute()

        original_subject = EMPLOYEE_REQUEST_EMAIL_SUBJECT

        messages = thread_data.get("messages", [])
        if messages:
            headers = messages[0].get("payload", {}).get("headers", [])

            for header in headers:
                if header.get("name", "").lower() == "subject":
                    original_subject = header.get(
                        "value",
                        EMPLOYEE_REQUEST_EMAIL_SUBJECT,
                    )
                    break

        if not original_subject.lower().startswith("re:"):
            subject = f"Re: {original_subject}"
        else:
            subject = original_subject

        email = EmailMessage()
        email["To"] = settings.GM_EMAIL
        email["Subject"] = subject
        email.set_content(acknowledgement)

        encoded_message = base64.urlsafe_b64encode(
            email.as_bytes()
        ).decode()

        _get_api_resource().users().messages().send(
            userId="me",
            body={
                "raw": encoded_message,
                "threadId": thread_id,
            },
        ).execute()

        logger.info(
            "LLM-generated acknowledgement sent in GM thread %s",
            thread_id,
        )

        return "Acknowledgement sent successfully."

    except Exception as exc:
        logger.exception(
            "Could not send acknowledgement in GM thread %s",
            thread_id,
        )
        return f"ERROR: Could not send acknowledgement: {exc}"