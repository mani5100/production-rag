from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from nxb_chatbot.db.session import get_db

# Database session dependency
DBSession = Annotated[AsyncSession, Depends(get_db)]

# Graph dependency
def get_graph(request: Request):
    """
    Returns the compiled LangGraph instance from app state.
    Graph is compiled once at startup via lifespan.
    """
    return request.app.state.graph


GraphDep = Annotated[object, Depends(get_graph)]