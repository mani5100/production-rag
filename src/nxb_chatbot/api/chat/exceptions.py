from nxb_chatbot.core.exceptions import NotFoundException, BadRequestException


class SessionNotFoundException(NotFoundException):
    def __init__(self, session_id: str):
        super().__init__(detail=f"Chat session '{session_id}' not found.")


class EmptyMessageException(BadRequestException):
    def __init__(self):
        super().__init__(detail="Message cannot be empty.")


class GraphInvokeException(BadRequestException):
    def __init__(self, detail: str):
        super().__init__(detail=f"Failed to invoke RAG graph: {detail}")