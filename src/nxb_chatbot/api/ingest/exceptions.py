from nxb_chatbot.core.exceptions import InternalServerException, NotFoundException


class IngestionFailedException(InternalServerException):
    def __init__(self, detail: str):
        super().__init__(detail=f"Ingestion failed: {detail}")


class DocumentNotFoundException(NotFoundException):
    def __init__(self, file_name: str):
        super().__init__(detail=f"Document '{file_name}' not found.")