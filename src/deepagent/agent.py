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

    Construye un grafo de estados con los siguientes nodos:
    - "init": inicializa el estado del agente con el mensaje de sistema.
    - "planner": invoca el modelo para generar un plan de acción.
    - "researcher": realiza investigaciones basadas en el plan.
    - "summarizer": resume los resultados de la investigación y genera una respuesta final.
    - "executor": ejecuta herramientas solicitadas por el modelo y retorna sus resultados.

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

        # Definición de nodos del grafo
        graph.add_node("planner", self.__plan)          # Nodo de planificación
        graph.add_node("researcher", self.__research)   # Nodo de investigación
        graph.add_node("tools", ToolNode(self.tools))   # Nodo de ejecución de herramientas

        # graph.add_node("action", self.__take_action)              # Nodo de ejecución de herramientas (acciones)

        # Definición de aristas del grafo
        graph.add_edge("planner", "researcher")  # Desde el modelo, ir a investigar
        graph.add_edge("researcher", END)  # Si el modelo no pide herramientas, terminar
        ##graph.add_conditional_edges(
        ##    "llm",                                              # La arista condicional sale del nodo "llm"
        ##    self.__exists_action,                                 # Función que decide si se debe ir al nodo de acción o terminar
        ##    {True: "action", False: END},
        ##)                                                       # Si el modelo decide llamar a una herramienta, ir al nodo de acción; si no, terminar
        ##graph.add_edge("action", "llm")                         # Después de ejecutar una acción, volver al modelo

        # Definición del punto de entrada del grafo
        graph.set_entry_point("planner")

        # Compilación del grafo para su ejecución con checkpointer en memoria (guarda el estado en memoria)
        return graph.compile(checkpointer=InMemorySaver())


    def __plan(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """
        Genera un plan de acción basado en la consulta del usuario.
        Args:
            state: estado actual con la consulta del usuario.
            config: configuración del runnable con callbacks.
        Returns:
            Un estado actualizado con el plan de acción.
        """

        # Emitir evento de inicio de planificación
        dispatch_custom_event("planning_started", {"status": "planning_started"}, config=config)

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
            "status": "planning_completed",
            "steps": plan.steps,
            "steps_count": len(plan.steps)
        }, config=config)

        return {
            **state, 
            "plan": plan,
            "messages": state.get("messages", []) + [AIMessage(content=f"He generado un plan de acción con {len(plan.steps)} pasos: {plan}")]  
        }

    def __research(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """
        Realiza una investigación basada en el plan de acción.
        Args:
            state: estado actual con el plan de acción.
        Returns:
            Un estado actualizado con los resultados de la investigación.
        """
        plan: Plan = state["plan"]
        step : str = plan.next_step()
        if step:

            # Emitir evento de planificación completada
            dispatch_custom_event("research_started", {
                "status": "research_started",
                "step": step,
            }, config=config)

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
            intent = structured_model.invoke(messages, config=config)

            # Emitir evento de planificación completada
            dispatch_custom_event("research_completed", {
                "status": "research_completed",
                "step": step,
                "intent": intent.dict(),
            }, config=config)

        return {"plan": plan}


    def __summarize(self, state: AgentState) -> AgentState:
        """
        Resume los resultados de la investigación y genera una respuesta final.
        Args:
            state: estado actual con todos los resultados de la investigación.
        Returns:
            Un estado actualizado con la respuesta final.
        """
        if self.verbose:
            print("🧩 Resumiendo resultados finales...")

        # TODO implementar el resumen final

        sys = SystemMessage(
            content=self.system_prompt
            + "\nResume en 5-8 líneas lo importante para seguir ejecutando. Devuelve texto."
        )
        out: AIMessage = self.model.invoke([sys] + state.get("messages", []))

        # estrategia simple: guardamos el resumen como un SystemMessage y recortamos
        summary = out.content
        new_messages = [SystemMessage(content=f"RESUMEN CONTEXTO:\n{summary}")]
        return {**state, "messages": new_messages, "scratch": state.get("scratch", {})}


        if self.verbose:
            print("✅ Resumen completado")

        return state

    def __route(self, state: AgentState) -> AgentState:
        """
        Decide la siguiente acción del agente: continuar investigando o sintetizar resultados.
        Args:
            state: estado actual del agente.
        Returns:
            Un estado actualizado después de decidir la siguiente acción.
        """
        plan: Plan = state.get("plan")
        if not plan:
            return state

        if plan.current_step < len(plan.steps):
            if self.verbose:
                print(
                    f"🔄 Continuando investigación - Paso {plan.current_step + 1}/{len(plan.steps)}"
                )
            return self.__research(state)
        else:
            if self.verbose:
                print("🏁 Investigación completada, procediendo a síntesis")
            return self.__finalize(state)
        

    def __finalize(self, state: AgentState) -> AgentState:
        """
        Sintetiza los resultados de la investigación en una respuesta final.
        Args:
            state: estado actual con todos los resultados de la investigación.
        Returns:
            Un estado actualizado con la respuesta final.
        """
        if self.verbose:
            print("🧩 Sintetizando resultados finales...")

        # TODO implementar la síntesis final

        if self.verbose:
            print("✅ Síntesis completada")

        return state


    # ------------------------------------------
    # Nodos condicionales
    # ------------------------------------------

    def __need_summarize(self, state: AgentState) -> bool:
        """
        Determina si el agente debe proceder a la síntesis de resultados.
        Args:
            state: estado actual del agente.
        Returns:
            True si no hay más pasos por ejecutar y se debe sintetizar, False en caso contrario.
        """
        return len(state.get("messages", [])) > 30

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
        return isinstance(last, AIMessage) and getattr(last, "tool_calls", None)
    

    def __need_more_steps(self, state: AgentState) -> bool:
        """
        Determina si debe continuar con la investigación o proceder a la síntesis.
        Args:
            state: estado actual del agente.
        Returns:
            True si hay más pasos por ejecutar, False en caso contrario.
        """
        plan: Plan = state.get("plan")
        if not plan:
            return False

        # Continuar si hay más pasos por ejecutar
        should_continue = plan.current_step < len(plan.steps)

        if self.verbose:
            if should_continue:
                print(
                    f"🔄 Continuando investigación - Paso {plan.current_step + 1}/{len(plan.steps)}"
                )
            else:
                print("🏁 Investigación completada, procediendo a síntesis")

        return should_continue

    # ------------------------------------------
    # Ejecución del agente
    # ------------------------------------------

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
                            chunk = Chunk(type=ChunkType.THINKING, content=content)
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
                            steps_count = data.get("steps_count", 0)
                            chunk = Chunk(type=ChunkType.THINKING, content=f"\n✅ Plan generado con {steps_count} pasos\n")
                        case "research_started":
                            step = data.get("step", "")
                            chunk = Chunk(type=ChunkType.THINKING, content=f"\n🔎 Investigando paso: {step}\n")
                        case "research_completed":
                            intent = data.get("intent", {})
                            chunk = Chunk(type=ChunkType.THINKING, content=f"\n✅ Investigación completada. Intent: {intent}\n")
                        case _:
                            # Otros eventos personalizados
                            if self.verbose:
                                chunk = Chunk(type=ChunkType.THINKING, content=f"[{event_name}]: {data}\n")

                case _:                    
                    #print(f"🔔 Evento no manejado: {kind} en nodo {node_name}")
                    pass

            if chunk:
                yield chunk

    # ------------------------------------------
    # Utilidades de depuración
    # ------------------------------------------

    def print_graph(self) -> None:
        """Imprime una representación del grafo del agente."""
        print(self.graph.get_graph().draw_mermaid())
