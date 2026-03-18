from enum import Enum
from pydantic import BaseModel, Field

class NextAction(str, Enum):
    ASK_USER = "ask_user"
    CONTINUE = "continue"

class ExecutorResponse(BaseModel):
    
    next_action: NextAction = Field(
        ...,
        description="La siguiente acción que el agente debe tomar",
        examples=[NextAction.ASK_USER, NextAction.CONTINUE]
    )

    message: str = Field(
        ...,
        description= "Respuesta del agente: pregunta para el usuario o respuesta final"
    )
    
    def __str__(self) -> str:
        return f"Next Action: {self.next_action}\nMessage: {self.message}"