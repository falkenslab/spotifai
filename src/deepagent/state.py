import operator
from typing import TypedDict, Annotated, Optional

from langchain_core.messages import AnyMessage

from deepagent.plan import Plan

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]     # Crea una lista de mensajes que se puede concatenar con el operador +
    plan: Optional[Plan]                                    # Plan de acción del agente
    scratch: Optional[dict]                                 # Resultados parciales de las acciones realizadas
    final: Optional[AnyMessage]                             # Respuesta final generada por el agente
