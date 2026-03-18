import asyncio
import json
from typing import AsyncGenerator, Optional

from langchain.chat_models.base import BaseChatModel
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage, AnyMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks.manager import dispatch_custom_event
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command

from deepagent.state import AgentState, MESSAGES_REPLACE_SENTINEL
from deepagent.chunk import Chunk, ChunkType
from deepagent.prompts import PromptFactory
from deepagent.plan import Plan
from deepagent.research import Intent
from deepagent.tools import generate_tools_description
from deepagent.executor import ExecutorResponse
from deepagent.critic import CriticEvaluation


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
        mocked_tools: bool = False,
    ):
        """
        Inicializa el agente con un modelo, herramientas y un prompt de sistema.
        Args:
            model: instancia del modelo conversacional (por ejemplo, ``ChatOpenAI``).
            domain: descripción del dominio de conocimiento del agente.
            tone: tono de comunicación del agente.
            tools: lista de herramientas compatibles con LangChain a exponer al modelo.
            verbose: si es True, imprime mensajes de depuración y resultados de herramientas.
            mocked_tools: si es True, simula la ejecución de herramientas sin llamarlas realmente.
        """

        self.tools = tools
        self.domain = domain
        self.tone = tone
        self.mocked_tools = mocked_tools   # Indica si las herramientas están siendo simuladas (mocked)

        # Descripción de las herramientas disponibles
        self.tools_description = generate_tools_description(self.tools)

        # Indica si el agente debe operar en modo verbose
        self.verbose = verbose

        # Configuración del grafo
        self.config : RunnableConfig = {
            "recursion_limit": 50,      # Límite de recursión para evitar bucles infinitos en el grafo (para que no esté infinitamente dando vueltas)
            "configurable": {
                "thread_id": "1"        # Identificador del hilo de conversación (en este caso, 1 agente sólo puede mantener una conversación a la vez)
            },
        }

        # Construcción del grafo de estados del agente
        self.graph = self.__build_graph()

        # Asociar las herramientas al modelo
        self.model = model
        self.model_with_tools = self.model.bind_tools(self.tools) if self.tools else self.model

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
        graph.add_node("tools", ToolNode(self.tools)    # Nodo de herramientas: ejecuta herramientas solicitadas por el agente
                       if not self.mocked_tools 
                       else self.__tools_mock
        )   
        graph.add_node("ask_user", self.__ask_user)     # Nodo de interacción con el usuario: solicita información adicional al usuario
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
            self.__route_executor,                      # Función que decide el siguiente nodo tras la ejecución
            {"tools": "tools", "ask_user": "ask_user", "critic": "critic"}
        )
        graph.add_edge("tools", "executor")             # Desde las herramientas, ir al nodo de ejecución
        graph.add_edge("ask_user", "executor")          # Tras recibir respuesta del usuario, reintentar ejecución
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
        planner_prompt = SystemMessage(content=PromptFactory.render("plan", {"domain": self.domain}))

        # Invocar el modelo para obtener el plan
        plan : Plan = self.model.with_structured_output(Plan).invoke(
            [
                self.system_prompt,
                planner_prompt, 
                self.human_query
            ], 
            config=config
        )

        # Emitir evento de planificación completada
        dispatch_custom_event("planning_completed", {
            "plan": plan,
        }, config=config)

        return {
            "plan": plan,
        }


    # -----------------------------------------------------------
    # Nodo de investigación
    # -----------------------------------------------------------

    def __research(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """Realiza una investigación basada en el plan de acción"""

        plan: Plan = state["plan"]
        step : str = plan.next_step()
        
        # Emitir evento de investigación iniciada
        dispatch_custom_event("research_started", {
            "step": step if step else "",
            "step_index": (plan.current_step - 1) if step else 0,
        }, config=config)

        if step:
            researcher_prompt = SystemMessage(content=PromptFactory.render(
                "research", 
                {
                    "domain": self.domain
                }
            ))
            research_query_prompt = HumanMessage(content=f"""
                Objetivo del usuario: {self.human_query.content}
                Paso del plan a analizar: {step}
            """)
            intent = self.model.with_structured_output(Intent).invoke(
                [
                    self.system_prompt,
                    researcher_prompt,
                    research_query_prompt,
                ],
                config=config,
            )
        else:
            intent = Intent(
                goal="Ninguno",
                notes="No hay pasos para investigar."
            )

        # Emitir evento de planificación completada
        dispatch_custom_event("research_completed", {
            "intent": intent,
        }, config=config)

        return {
            "scratch": {
                "step": step,
                "intent": intent
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

        messages = state.get("messages", [])

        summary_prompt = HumanMessage(
            content="Resume en 5-8 líneas lo importante para seguir ejecutando. Devuelve texto."
        )
        
        summary: AIMessage = self.model.invoke(messages + [summary_prompt], config=config)

        dispatch_custom_event("summarizing_completed", {
            "summary": summary.content
        }, config=config)

        # Reemplazar todos los mensajes con el system_prompt, la query original y el resumen        
        return {
            "messages": [
                SystemMessage(content=MESSAGES_REPLACE_SENTINEL),
                self.system_prompt,
                self.human_query,
                SystemMessage(content=summary.content),
            ]        
        }


    # -----------------------------------------------------------
    # Nodo que determina que realiza acciones
    # -----------------------------------------------------------

    def __executor(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """
        Decide la siguiente acción del agente:
        * Ejecutar una herramienta.
        * Pedir más información.
        * Finalizar la ejecución.
        """

        intent : Intent = state.get("scratch", {}).get("intent")
        step : Optional[str] = state.get("scratch", {}).get("step")
        messages = state.get("messages", [])

        dispatch_custom_event("execution_started", {
            "step": step if step else "",
            "intent": intent
        }, config=config)

        executor_prompt = SystemMessage(content=PromptFactory.render("executor"))
        query = HumanMessage(content=f"""
            Objetivo: {self.human_query.content}
            Paso actual: {step}
            Análisis del paso:
            - Objetivo de la investigación: {intent.goal}
            - Notas de la investigación: {intent.notes}
            Decide la siguiente acción a tomar.
        """)
        ai_response: AIMessage = self.model_with_tools.invoke(
            messages + [ executor_prompt, query ],
            config=config
        )

        # Si no hay tool_calls, intentar parsear el JSON para detectar "ask_user"
        updates: AgentState = {"messages": [ai_response]}
        if not getattr(ai_response, "tool_calls", None):
            try:
                data = json.loads(ai_response.content)
                if data.get("action") == "ask_user":
                    question = data.get("reason", "¿Puedes proporcionar más información?")
                    updates["scratch"] = {**state.get("scratch", {}), "question": question}
            except (json.JSONDecodeError, AttributeError, TypeError):
                pass

        dispatch_custom_event("execution_completed", {
            "response": ai_response.content
        }, config=config)

        return updates

    # -----------------------------------------------------------
    # Nodo para preguntar al usuario (human in the loop)
    # -----------------------------------------------------------

    def __ask_user(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """Pausa la ejecución, pregunta al usuario y añade su respuesta al historial."""
        question = state.get("scratch", {}).get("question", "¿Puedes proporcionar más información?")
        dispatch_custom_event("ask_user_started", {"question": question}, config=config)

        # Pausa el grafo hasta que se reanude con Command(resume=respuesta)
        user_answer = interrupt(question)

        dispatch_custom_event("ask_user_completed", {"answer": user_answer}, config=config)

        # Limpiar la pregunta del scratch y añadir la respuesta del usuario a los mensajes
        scratch = {k: v for k, v in state.get("scratch", {}).items() if k != "question"}
        return {
            "messages": [HumanMessage(content=user_answer)],
            "scratch": scratch,
        }

    # -----------------------------------------------------------
    # Nodo de tools (mock): simula la ejecución de las tools (testing)
    # -----------------------------------------------------------

    def __tools_mock(self, state: AgentState, config: RunnableConfig) -> AgentState:
        last = (state.get("messages") or [])[-1]

        tool_messages: list[ToolMessage] = []

        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            for tc in last.tool_calls:
                # `tc` suele ser dict-like: {"name": "...", "args": {...}, "id": "..."}
                name = tc.get("name", "unknown_tool")
                args = tc.get("args", {})
                tc_id = tc.get("id", "")

                payload = json.dumps(args, ensure_ascii=False)
                content = f"[MOCK] Ejecutaría: {name}({payload})"

                tool_message = ToolMessage(content=content, tool_call_id=tc_id)
                tool_message.pretty_print()
                
                tool_messages.append(tool_message)

        return {"messages": tool_messages}


    # -----------------------------------------------------------
    # Nodo que determina si el agente termina o continúa
    # -----------------------------------------------------------

    def __critic(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """Evalúa si el paso actual del plan se completó correctamente."""

        scratch = state.get("scratch", {})
        step = scratch.get("step", "")
        intent: Intent = scratch.get("intent")
        messages = state.get("messages", [])

        dispatch_custom_event("critic_started", {"step": step}, config=config)

        critic_prompt = SystemMessage(content=PromptFactory.render("critic"))
        query = HumanMessage(content=f"""
            Paso evaluado: {step}
            Objetivo del paso: {intent.goal if intent else ""}
        """)

        evaluation: CriticEvaluation = self.model.with_structured_output(CriticEvaluation).invoke(
            messages + [critic_prompt, query],
            config=config,
        )

        dispatch_custom_event("critic_completed", {"evaluation": evaluation}, config=config)

        return {
            "scratch": {**scratch, "critic": evaluation},
        }


    # -----------------------------------------------------------
    # Nodo de síntesis final
    # -----------------------------------------------------------

    def __finalize(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """Genera la respuesta final al usuario sintetizando todo el trabajo realizado."""

        messages = state.get("messages", [])

        dispatch_custom_event("finalizing_started", {}, config=config)

        finalizer_prompt = SystemMessage(content=PromptFactory.render("finalizer"))
        query = HumanMessage(content=f"Petición original del usuario: {self.human_query.content}")

        response: AIMessage = self.model.invoke(
            messages + [finalizer_prompt, query],
            config=config,
        )

        dispatch_custom_event("finalizing_completed", {}, config=config)

        return {
            "messages": [response],
            "final": response,
        }


    # -----------------------------------------------------------
    # Nodos condicionales
    # -----------------------------------------------------------

    def __need_summarize(self, state: AgentState, config: RunnableConfig) -> bool:
        """
        Determina si el agente debe proceder a la síntesis de resultados.
        Args:
            state: estado actual del agente.
        Returns:
            True si no hay más pasos por ejecutar y se debe sintetizar, False en caso contrario.
        """
        messages = state.get("messages", [])
        need_summarize = len(messages) > 10
        dispatch_custom_event("need_summarize", {"need_summarize": need_summarize}, config=config)
        return need_summarize


    def __route_executor(self, state: AgentState, config: RunnableConfig) -> str:
        """
        Decide el siguiente nodo tras la ejecución:
        - ``"tools"``    si el modelo solicitó herramientas.
        - ``"ask_user"`` si el executor necesita más información del usuario.
        - ``"critic"``   en caso contrario.
        """
        last = state.get("messages", [])[-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            dispatch_custom_event("need_tools", {"need_tools": True}, config=config)
            return "tools"
        if state.get("scratch", {}).get("question"):
            dispatch_custom_event("need_tools", {"need_tools": False}, config=config)
            return "ask_user"
        dispatch_custom_event("need_tools", {"need_tools": False}, config=config)
        return "critic"
    

    def __need_more_steps(self, state: AgentState, config: RunnableConfig) -> bool:
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
        dispatch_custom_event("need_more_steps", {"should_continue": should_continue}, config=config)
        return should_continue

    # -----------------------------------------------------------
    # Ejecución del agente
    # -----------------------------------------------------------

    async def invoke(self, query: str, ask_user_fn=None) -> AsyncGenerator[Chunk, None]:
        """
        Formula una pregunta al agente y devuelve la última respuesta.
        Args:
            query: texto de la consulta del usuario.
            ask_user_fn: función síncrona opcional ``(question: str) -> str`` que se llama
                         cuando el agente necesita información adicional del usuario.
                         Si no se proporciona, se emite un ``Chunk`` de tipo ``QUESTION`` y se detiene.
        Returns:
            Generador asíncrono que produce fragmentos de la respuesta final del agente.
        """
        self.human_query = HumanMessage(content=query)

        input_data = {
            "messages": [
                self.system_prompt,
                self.human_query
            ]
        }

        while True:
            # Ejecuta el grafo de modo asíncrono y obtiene los eventos
            events = self.graph.astream_events(
                input=input_data,
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
                        intent = event["data"]["chunk"].content
                        match lg_node:
                            case "researcher" | "planner":
                                pass
                                #chunk = Chunk(type=ChunkType.THINKING, content=content)
                            case "finalizer":
                                chunk = Chunk(type=ChunkType.TEXT, content=intent)

                    # Manejar eventos personalizados de planificación
                    case "on_custom_event":
                        event_name = event.get("name", "")
                        data = event.get("data", {})

                        match event_name:
                            case "need_summarize":
                                need_summarize = data.get("need_summarize", False)
                                chunk = Chunk(type=ChunkType.THINKING, content=f"\n❓ Necesito resumir la conversación para continuar: {need_summarize}\n")
                            case "need_tools":
                                need_tools = data.get("need_tools", False)
                                chunk = Chunk(type=ChunkType.THINKING, content=f"\n❓ Necesito llamar a herramientas para continuar: {need_tools}\n")
                            case "need_more_steps":
                                should_continue = data.get("should_continue", False)
                                chunk = Chunk(type=ChunkType.THINKING, content=f"\n❓ Quedan más pasos por investigar: {should_continue}\n")
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
                                intent : Intent = data.get("intent", {})
                                output = (
                                    f"* Objetivo: {intent.goal}\n"
                                    f"* Notas   : {intent.notes}\n"
                                    f"✅ Investigación del paso [{step_index + 1}] completada"
                                )
                                chunk = Chunk(type=ChunkType.THINKING, content=f"{output}\n")
                            case "summarizing_started":
                                total_messages = data.get("total_messages", 0)
                                chunk = Chunk(type=ChunkType.THINKING, content=f"\n📝 Resumiendo conversación (total mensajes: {total_messages})...\n")
                            case "summarizing_completed":
                                summary = data.get("summary", "")
                                output = f"✅ Resumen completado:\n{summary}"
                                chunk = Chunk(type=ChunkType.THINKING, content=f"{output}\n")
                            case "execution_started":
                                step = data.get("step", "")
                                intent = data.get("intent", {})
                                output = (
                                    f"\n🚀 Ejecutando paso: {step}\n"
                                    f"* Objetivo de la investigación: {intent.goal}\n"
                                    f"* Notas de la investigación: {intent.notes}\n"
                                )
                                chunk = Chunk(type=ChunkType.THINKING, content=output)
                            case "execution_completed":
                                response = data.get("response", None)
                                output = f"✅ Ejecución del paso completada. Respuesta del agente: '{response}'\n"
                                chunk = Chunk(type=ChunkType.THINKING, content=output)
                            case "ask_user_started":
                                question = data.get("question", "")
                                chunk = Chunk(type=ChunkType.THINKING, content=f"\n🙋 El agente necesita más información: {question}\n")
                            case "ask_user_completed":
                                answer = data.get("answer", "")
                                chunk = Chunk(type=ChunkType.THINKING, content=f"✅ Respuesta del usuario recibida: '{answer}'\n")
                            case "critic_started":
                                step = data.get("step", "")
                                chunk = Chunk(type=ChunkType.THINKING, content=f"\n🔍 Evaluando paso: {step}\n")
                            case "critic_completed":
                                evaluation: CriticEvaluation = data.get("evaluation")
                                chunk = Chunk(type=ChunkType.THINKING, content=f"{evaluation}\n")
                            case "finalizing_started":
                                chunk = Chunk(type=ChunkType.THINKING, content="\n✍️ Generando respuesta final...\n")
                            case "finalizing_completed":
                                chunk = Chunk(type=ChunkType.THINKING, content="✅ Respuesta final generada.\n")
                            case _:
                                # Otros eventos personalizados
                                chunk = Chunk(type=ChunkType.THINKING, content=f"\n[{event_name}]: {data}\n")

                    case _:
                        #print(f"🔔 Evento no manejado: {kind} en nodo {node_name}")
                        pass

                if chunk:
                    yield chunk

            # Comprobar si el grafo está pausado esperando input del usuario
            state = await self.graph.aget_state(self.config)
            interrupted = state.tasks and any(task.interrupts for task in state.tasks)

            if interrupted:
                question = state.tasks[0].interrupts[0].value
                if ask_user_fn:
                    user_answer = await asyncio.to_thread(ask_user_fn, question)
                    input_data = Command(resume=user_answer)
                else:
                    yield Chunk(type=ChunkType.QUESTION, content=question)
                    break
            else:
                break

    # -----------------------------------------------------------
    # Utilidades de depuración
    # -----------------------------------------------------------

    def print_graph(self) -> None:
        """Imprime una representación del grafo del agente."""
        print(self.graph.get_graph().draw_mermaid())
