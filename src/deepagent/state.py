import operator
from typing import TypedDict, Annotated, Optional

from langchain_core.messages import AnyMessage

from deepagent.plan import Plan

def replace_messages(left: list[AnyMessage], right: list[AnyMessage]) -> list[AnyMessage]:
    """Reemplaza los mensajes en lugar de concatenarlos"""
    return right if right else left

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], replace_messages]     # Lista de mensajes que se reemplaza completamente
    plan: Optional[Plan]                                         # Plan de acción del agente
    scratch: Optional[dict]                                      # Resultados parciales de las acciones realizadas
    final: Optional[AnyMessage]                                  # Respuesta final generada por el agente
