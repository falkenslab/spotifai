from typing import AsyncGenerator

from langchain.chat_models.base import BaseChatModel
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks.manager import dispatch_custom_event
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode

from deepagent.state import AgentState
from deepagent.chunk import Chunk, ChunkType
from deepagent.prompts import PromptFactory
from deepagent.plan import Plan
from deepagent.research import ResearchResult
from deepagent.tools import generate_tools_description


class DeepAgent:
    """
    Agente profundo que razona y actúa según un plan definido para llevar a cabo la petición del usuario.
    Args:
        model: instancia del modelo conversacional (por ejemplo, ``ChatOpenAI``).
        domain: descripción del dominio de conocimiento del agente.
        tone: tono de comunicación del agente.
        tools: lista de herramientas compatibles con LangChain a exponer al modelo.
        verbose: si es True, imprime mensajes de depuración y resultados de herramientas.
    """

    def __init__(
        self,
        model: BaseChatModel,
        domain: str,
        tone: str,
        tools: list[tool],
        verbose: bool = False,
    ):
        """
        Inicializa el agente con un modelo, herramientas y un prompt de sistema.
        Args:
            model: instancia del modelo conversacional (por ejemplo, ``ChatOpenAI``).
            domain: descripción del dominio de conocimiento del agente.
            tone: tono de comunicación del agente.
            tools: lista de herramientas compatibles con LangChain a exponer al modelo.
            verbose: si es True, imprime mensajes de depuración y resultados de herramientas.
        """

        self.tools = tools
        self.domain = domain
        self.tone = tone

        # Descripción de las herramientas disponibles
        self.tools_description = generate_tools_description(self.tools)

        # Indica si el agente debe operar en modo verbose
        self.verbose = verbose

        # Configuración del grafo
        self.config : RunnableConfig = {
            "recursionLimit": 50,      # Límite de recursión para evitar bucles infinitos en el grafo (para que no esté infinitamente dando vueltas)
            "configurable": {
                "thread_id": "1"        # Identificador del hilo de conversación (en este caso, 1 agente sólo puede mantener una conversación a la vez)
            },
        }

        # Construcción del grafo de estados del agente
        self.graph = self.__build_graph()

        # Asociar las herramientas al modelo
        self.model = model

        # Construir el prompt de sistema
        self.system_prompt = SystemMessage(content=PromptFactory.render(
            "system",
            {
                "domain": self.domain,
                "tools": self.tools_description,
                "tone": self.tone,
            },
        ))


    def __build_graph(self) -> CompiledStateGraph:
        """
        Construye y compila el grafo de estados del agente.
        Returns:
            El grafo de estados compilado.
        """
        # Construcción del grafo de estados del agente
        graph = StateGraph(AgentState)

        # Definición del punto de entrada del grafo
        graph.set_entry_point("planner")

        # Definición de nodos del grafo
        graph.add_node("planner", self.__plan)          # Nodo de planificación: genera el plan de acción
        graph.add_node("researcher", self.__research)   # Nodo de investigación: investiga cada paso del plan
        graph.add_node("summarizer", self.__summarize)  # Nodo de resumen: resume la conversación hasta ahora (si es necesario)
        graph.add_node("executor", self.__executor)     # Nodo de ejecución: determina qué herramientas hay que ejecutar
        graph.add_node("tools", ToolNode(self.tools))   # Nodo de herramientas: ejecuta herramientas solicitadas por el agente
        graph.add_node("critic", self.__critic)         # Nodo de juicio: decide si continuar o finalizar
        graph.add_node("finalizer", self.__finalize)    # Nodo de síntesis final: genera la respuesta final

        # Definición de aristas del grafo
        graph.add_edge("planner", "researcher")         # Desde el modelo, ir a investigar
        graph.add_conditional_edges(
            "researcher",                               # La arista condicional sale del nodo "researcher"
            self.__need_summarize,                      # Función que decide si se debe ir al nodo de resumen o continuar
            {True: "summarizer", False: "executor"}     # Si se debe resumir, ir al nodo de resumen; si no, al nodo de enrutamiento
        )
        graph.add_edge("summarizer", "executor")        # Si el modelo no pide herramientas, terminar
        graph.add_conditional_edges(
            "executor",                                 # La arista condicional sale del nodo "executor"
            self.__need_tools,                          # Función que decide si se deben ejecutar herramientas o resumir
            {True: "tools", False: "critic"}            # Si el modelo pide herramientas, ir al nodo de herramientas; si no, al nodo de juicio
        )
        graph.add_edge("tools", "executor")             # Desde las herramientas, ir al nodo de ejecución
        graph.add_conditional_edges(
            "critic",                                   # La arista condicional sale del nodo "critic"
            self.__need_more_steps,                     # Función que decide si se deben investigar más pasos
            {True: "researcher", False: "finalizer"}    # Si hay más pasos, ir al nodo de investigación; si no, al nodo de síntesis final
        )
        graph.add_edge("finalizer", END)                # Desde la síntesis final, terminar

        # Compilación del grafo para su ejecución con checkpointer en memoria (guarda el estado en memoria)
        return graph.compile(checkpointer=InMemorySaver())


    # -----------------------------------------------------------
    # Nodo de planificación
    # -----------------------------------------------------------

    def __plan(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """Genera un plan de acción basado en la consulta del usuario"""

        # Emitir evento de inicio de planificación
        dispatch_custom_event("planning_started", {}, config=config)

        # Construir el mensaje de sistema para la planificación
        plan_prompt = SystemMessage(content=PromptFactory.render("plan", {"domain": self.domain}))

        # Invocar el modelo para obtener el plan
        structured_model : BaseChatModel[Plan] = self.model.with_structured_output(Plan)
        plan : Plan = structured_model.invoke([
            self.system_prompt,
            plan_prompt, 
            self.human_query
        ], config=config)

        # Emitir evento de planificación completada
        dispatch_custom_event("planning_completed", {
            "plan": plan,
        }, config=config)

        return {
            "plan": plan,
            "messages": state.get("messages", []) + [AIMessage(content=f"He generado un plan de acción con {len(plan.steps)} pasos: {plan}")]  
        }


    # -----------------------------------------------------------
    # Nodo de investigación
    # -----------------------------------------------------------

    def __research(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """Realiza una investigación basada en el plan de acción"""

        plan: Plan = state.plan
        step : str = plan.next_step()
        
        # Emitir evento de investigación iniciada
        dispatch_custom_event("research_started", {
            "step": step if step else "",
            "step_index": (plan.current_step - 1) if step else 0,
        }, config=config)

        if step:
            reasearch_prompt = SystemMessage(content=PromptFactory.render(
                "research", 
                {
                    "domain": self.domain
                }
            ))
            messages = [
                self.system_prompt,
                reasearch_prompt,
                HumanMessage(content=f"Paso del plan a analizar: {step}."),
            ]
            structured_model : BaseChatModel[ResearchResult] = self.model.with_structured_output(ResearchResult)
            research_result = structured_model.invoke(messages, config=config)
        else:
            research_result = ResearchResult(
                intent="Ninguno",
                notes="No hay pasos para investigar."
            )

        # Emitir evento de planificación completada
        dispatch_custom_event("research_completed", {
            "research_result": research_result,
        }, config=config)

        return {
            "plan": plan, 
            "scratch": {
                "research_result": research_result
            },
        }


    # -----------------------------------------------------------
    # Nodo de resumen
    # -----------------------------------------------------------

    def __summarize(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """Resume los resultados de la investigación y genera una respuesta final"""

        dispatch_custom_event("summarizing_started", {
            "total_messages": len(state.get("messages", []))
        }, config=config)

        summary_prompt = HumanMessage(content="Resume en 5-8 líneas lo importante para seguir ejecutando. Devuelve texto.")
        summary: AIMessage = self.model.invoke(state.get("messages", []) + [summary_prompt], config=config)
        new_messages = [SystemMessage(content=summary.content)]

        dispatch_custom_event("summarizing_completed", {}, config=config)

        return {
            "messages": new_messages
        }


    # -----------------------------------------------------------
    # Nodo que determina que herramientas ejecutar
    # -----------------------------------------------------------

    def __executor(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """Decide la siguiente acción del agente: continuar investigando o sintetizar resultados."""

        #research_result =

        dispatch_custom_event("execution_started", {}, config=config)

        dispatch_custom_event("execution_completed", {}, config=config)
        return state


    # -----------------------------------------------------------
    # Nodo que determina si el agente termina o continúa
    # -----------------------------------------------------------

    def __critic(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """Evalúa el progreso del agente y decide si continuar o finalizar."""
        dispatch_custom_event("critic_started", {}, config=config)
        dispatch_custom_event("critic_completed", {}, config=config)
        return state


    # -----------------------------------------------------------
    # Nodo de síntesis final
    # -----------------------------------------------------------

    def __finalize(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """Sintetiza los resultados de la investigación en una respuesta final."""
        dispatch_custom_event("finalizing_started", {}, config=config)
        dispatch_custom_event("finalizing_completed", {}, config=config)
        return state


    # -----------------------------------------------------------
    # Nodos condicionales
    # -----------------------------------------------------------

    def __need_summarize(self, state: AgentState) -> bool:
        """
        Determina si el agente debe proceder a la síntesis de resultados.
        Args:
            state: estado actual del agente.
        Returns:
            True si no hay más pasos por ejecutar y se debe sintetizar, False en caso contrario.
        """
        messages = state.get("messages", [])
        need_summarize = len(messages) > 10
        print(f"❓Necesita resumir: {need_summarize}")
        return need_summarize

    def __need_tools(self, state: AgentState) -> bool:
        """
        Comprueba si el modelo ha solicitado la ejecución de herramientas.
        Args:
            state: estado actual con el historial de mensajes.
        Returns:
            ``True`` si el último mensaje del modelo incluye ``tool_calls``; en caso contrario ``False``.
        """
        # Último mensaje generado por el modelo
        last = state.get("messages", [])[-1]
        # Devuelve True si hay llamadas a herramientas en el último mensaje
        need_tools = isinstance(last, AIMessage) and getattr(last, "tool_calls", None)
        print(f"❓Necesita llamar a herramientas: {need_tools}")
        return need_tools
    

    def __need_more_steps(self, state: AgentState) -> bool:
        """
        Determina si debe continuar con la investigación o proceder a la síntesis.
        Args:
            state: estado actual del agente.
        Returns:
            True si hay más pasos por ejecutar, False en caso contrario.
        """
        plan: Plan = state.get("plan")
        # Continuar si hay más pasos por ejecutar
        should_continue = plan.current_step < len(plan.steps)
        print(f"❓Necesita más pasos: {should_continue}")
        return should_continue

    # -----------------------------------------------------------
    # Ejecución del agente
    # -----------------------------------------------------------

    async def invoke(self, query: str) -> AsyncGenerator[Chunk, None]:
        """
        Formula una pregunta al agente y devuelve la última respuesta.
        Args:
            query: texto de la consulta del usuario.
        Returns:
            Generador asíncrono que produce fragmentos de la respuesta final del agente.
        """
        self.human_query = HumanMessage(content=query)

        initial_state = {
            "messages": [ 
                self.system_prompt,
                self.human_query
            ]
        }

        # Ejecuta el grafo de modo asíncrono y obtiene los eventos
        events = self.graph.astream_events(
            input=initial_state,
            config=self.config,
        )

        async for event in events:
            # Obtiene el tipo de evento
            kind = event["event"]
            chunk : Chunk = None    # Fragmento de respuesta que emite el agente
            match kind:

                # Solo mostrar eventos del modelo del nodo researcher
                case "on_chat_model_stream":
                    # Filtrar por el nodo que queremos mostrar
                    lg_node = event['metadata'].get('langgraph_node', "")
                    content = event["data"]["chunk"].content
                    match lg_node:
                        case "researcher" | "planner":
                            pass
                            #chunk = Chunk(type=ChunkType.THINKING, content=content)
                        case "finalizer":
                            chunk = Chunk(type=ChunkType.TEXT, content=content)

                # Manejar eventos personalizados de planificación
                case "on_custom_event":
                    event_name = event.get("name", "")
                    data = event.get("data", {})
                    
                    match event_name:
                        case "planning_started":
                            chunk = Chunk(type=ChunkType.THINKING, content="🧠 Generando plan de acción...\n")
                        case "planning_completed":
                            plan : Plan = data["plan"] if "plan" in data else None
                            output = ""
                            for idx, step in enumerate(plan.steps):
                                output += f"* Paso [{idx + 1}]: {step}\n"
                            output += f"✅ Plan generado con {len(plan.steps)} pasos\n"
                            chunk = Chunk(type=ChunkType.THINKING, content=output)
                        case "research_started":
                            step = data.get("step", "")
                            step_index = data.get("step_index", 0)
                            chunk = Chunk(type=ChunkType.THINKING, content=f"\n🔎 Investigando paso [{step_index + 1}]: {step}\n")
                        case "research_completed":
                            research_result : ResearchResult = data.get("research_result", {})
                            intent = research_result.intent
                            notes = research_result.notes
                            output = f"""
* Objetivo: {intent}
* Notas   : {notes}
✅ Investifación del paso [{step_index + 1}] completada
"""
                            chunk = Chunk(type=ChunkType.THINKING, content=f"{output.strip()}\n")
                        case _:
                            # Otros eventos personalizados
                            chunk = Chunk(type=ChunkType.THINKING, content=f"[{event_name}]: {data}\n")

                case _:                    
                    #print(f"🔔 Evento no manejado: {kind} en nodo {node_name}")
                    pass

            if chunk:
                yield chunk

    # -----------------------------------------------------------
    # Utilidades de depuración
    # -----------------------------------------------------------

    def print_graph(self) -> None:
        """Imprime una representación del grafo del agente."""
        print(self.graph.get_graph().draw_mermaid())
