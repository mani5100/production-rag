from pydantic import BaseModel

class GuardrailResult(BaseModel):
    passed: bool
    reason: str