from typing import TypedDict, Annotated, Optional

from langchain_core.messages import AnyMessage, SystemMessage

from deepagent.plan import Plan

MESSAGES_REPLACE_SENTINEL = "__REPLACE_MESSAGES__"

def merge_messages(left: list[AnyMessage], right: list[AnyMessage]) -> list[AnyMessage]:
    if not right:
        return left

    # Si el primer mensaje es el sentinel, reemplaza todo el historial
    if isinstance(right[0], SystemMessage) and right[0].content == MESSAGES_REPLACE_SENTINEL:
        return right[1:]

    # Si no, concatena (comportamiento esperado por ToolNode)
    return left + right

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], merge_messages]        # Lista de mensajes con el historial de la conversación
    plan: Optional[Plan]                                         # Plan de acción del agente
    scratch: Optional[dict]                                      # Resultados parciales de las acciones realizadas
    final: Optional[AnyMessage]                                  # Respuesta final generada por el agente
