import operator
from enum import Enum
from typing import TypedDict, Annotated, Optional

from langchain_core.messages import AnyMessage

from deepagent.plan import Plan

class Status(Enum):
    OK = "ok"
    WAITING_USER = "waiting_user"
    ERROR = "error"
    DONE = "done"

class AgentState(TypedDict):
    user_query: str  # Consulta del usuario
    plan: Optional[Plan]  # Plan de acción del agente
    partial_results: Optional[dict[str, AnyMessage]]  # Resultados parciales de las acciones realizadas
    final_output: Optional[AnyMessage]  # Respuesta final generada por el agente
    status: Optional[Status]  # Estado actual del agente    
    messages: Annotated[list[AnyMessage], operator.add]  # Crea una lista de mensajes que se puede concatenar con el operador +
    memory: dict  # Memoria adicional para almacenar datos entre estados
    error: Optional[str]  # Mensaje de error en caso de fallo
