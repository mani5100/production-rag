from fastapi import HTTPException, status


class NXBBaseException(HTTPException):
    pass


class NotFoundException(NXBBaseException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail,
        )


class BadRequestException(NXBBaseException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
        )


class InternalServerException(NXBBaseException):
    def __init__(self, detail: str = "An unexpected error occurred."):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
        )