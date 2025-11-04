import sys
import json
from typing import AsyncGenerator

from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import InMemorySaver

from deepagent.plan import Plan
from deepagent.prompts import PLAN_PROMPT_TEMPLATE
from deepagent.state import AgentState, Status

class DeepAgent:
    """
    Agente profundo que razona y actúa según un plan definido para llevar a cabo la petición del usuario.

    Construye un grafo de estados con los siguientes nodos:
    - "init": inicializa el estado del agente con el mensaje de sistema.
    - "planner": invoca el modelo para generar un plan de acción.
    - "researcher": realiza investigaciones basadas en el plan.
    - "summarizer": resume los resultados de la investigación y genera una respuesta final.
    - "executor": ejecuta herramientas solicitadas por el modelo y retorna sus resultados.

    Atributos:
        system: texto del mensaje de sistema que se añadirá una vez al principio.
        graph: grafo compilado de LangGraph que gestiona el flujo de mensajes.
        tools: diccionario de herramientas disponibles indexadas por nombre.
        model: modelo LLM con herramientas enlazadas mediante ``bind_tools``.
        verbose: si es True, imprime mensajes de depuración y resultados de herramientas.        
    """

    def __init__(self, model: BaseChatModel, tools, system: str = "", verbose: bool = False):
        """
        Inicializa el agente con un modelo, herramientas y un prompt de sistema.
        Args:
            model: instancia del modelo conversacional (por ejemplo, ``ChatOpenAI``).
            tools: lista de herramientas compatibles con LangChain a exponer al modelo.
            system: mensaje de sistema opcional que se antepone solo una vez.
            verbose: si es True, imprime mensajes de depuración y resultados de herramientas.
        """
        self.system = SystemMessage(content=system) if system else None
        self.verbose = verbose
        self.config = {
            "recursion_limit": 50,                              # Límite de recursión para evitar bucles infinitos en el grafo (para que no esté infinitamente dando vueltas)
            "configurable": { "thread_id" : "1"}                # Identificador del hilo de conversación (en este caso, 1 agente sólo puede mantener una conversación a la vez)
        }

        # Construcción del grafo de estados del agente
        self.graph = self.__build_graph()

        # Mapeo de herramientas por nombre
        self.tools = {t.name: t for t in tools}

        # Asociar las herramientas al modelo
        self.model = model.bind_tools(tools)

    def __build_graph(self) -> CompiledStateGraph:
        """
        Construye y compila el grafo de estados del agente.
        Returns:
            El grafo de estados compilado.
        """
        # Construcción del grafo de estados del agente
        graph = StateGraph(AgentState)

        # Definición de nodos del grafo
        graph.add_node("init", self.__initialize)                 # Nodo de inicialización del estado del agente
        graph.add_node("planner", self.__plan)              # Nodo de planificación
        graph.add_node("researcher", self.__research)       # Nodo de investigación
        #graph.add_node("action", self.__take_action)              # Nodo de ejecución de herramientas (acciones)

        # Definición de aristas del grafo
        graph.add_edge("init", "planner")                   # Desde el estado inicial, ir al modelo
        graph.add_edge("planner", "researcher")             # Desde el modelo, ir a investigar
        graph.add_edge("researcher", END)                      # Si el modelo no pide herramientas, terminar
        ##graph.add_conditional_edges(
        ##    "llm",                                              # La arista condicional sale del nodo "llm"
        ##    self.__exists_action,                                 # Función que decide si se debe ir al nodo de acción o terminar
        ##    {True: "action", False: END},
        ##)                                                       # Si el modelo decide llamar a una herramienta, ir al nodo de acción; si no, terminar
        ##graph.add_edge("action", "llm")                         # Después de ejecutar una acción, volver al modelo

        # Definición del punto de entrada del grafo
        graph.set_entry_point("init")

        # Compilación del grafo para su ejecución con checkpointer en memoria (guarda el estado en memoria)
        return graph.compile(checkpointer=InMemorySaver())
    

    def __call_tool(self, tool_call) -> ToolMessage:
        """
        Ejecuta una herramienta solicitada por el modelo y retorna su resultado.
        Args:
            tool_call: Un diccionario con la información de la llamada a la herramienta.
        Returns:
            Un mensaje de tipo ToolMessage con el resultado de la ejecución.
        """
        # Si está en modo verbose, imprimir la acción que se va a ejecutar
        if self.verbose:
            print(f"\nEjecutando acción: {tool_call}\n")
        # Comprobar si la herramienta existe
        if not tool_call["name"] in self.tools:
            # Si está en modo verbose, imprimir mensaje de error
            if self.verbose:
                print("\n ....nombre de tool no válida....", file=sys.stderr)
            result = "nombre de tool no válida, reintentar"  # instruir al LLM a reintentar si el nombre es incorrecto
        else:
            # Ejecutar la herramienta y obtener el resultado
            result = self.tools[tool_call["name"]].invoke(tool_call["args"])
        # Devuelve un mensaje de tipo ToolMessage con el resultado de ejecutar la herramienta
        return ToolMessage(tool_call_id=tool_call["id"], name=tool_call["name"], content=str(result))
    

    def __initialize(self, state: AgentState) -> AgentState:
        """
        Crea el estado inicial del agente con el mensaje de sistema.
        Returns:
            Un estado inicial con el mensaje de sistema si está configurado.
        """
        return {
            "messages": [ self.system ] if self.system else [],
            "user_query": state["messages"][-1].content,
            "status": Status.OK
        }
    
    
    def __plan(self, state: AgentState) -> AgentState:
        """
        Genera un plan de acción basado en la consulta del usuario.
        Args:
            state: estado actual con la consulta del usuario.
        Returns:
            Un estado actualizado con el plan de acción.
        """
        query = PLAN_PROMPT_TEMPLATE.format(task=state["user_query"])

        # Invocar el modelo para obtener el plan
        messages = state["messages"] + [SystemMessage(content=query)]
        message = self.model.invoke(messages)
        
        try:
            # Intentar parsear el JSON directamente
            plan_data = json.loads(message.content.strip())
            plan = Plan(**plan_data)
            plan.current_step = 0  # Reiniciar el paso actual
        except (json.JSONDecodeError, Exception) as e:
            # Plan por defecto en caso de error
            plan = Plan(
                steps=["Analizar la consulta del usuario", "Ejecutar acciones necesarias", "Proporcionar respuesta"],
                current_step=0
            )

        state["plan"] = plan
        state["messages"] = messages + message
        return state
    
    def __research(self, state: AgentState) -> AgentState:
        """
        Realiza una investigación basada en el plan de acción.
        Args:
            state: estado actual con el plan de acción.
        Returns:
            Un estado actualizado con los resultados de la investigación.
        """
        plan: Plan = state["plan"]
        if plan.current_step < len(plan.steps):
            step = plan.steps[plan.current_step]
            # Aquí se podría implementar la lógica de investigación
            print(f"Investigando: {step}")
            # Por ahora, simplemente avanzamos al siguiente paso
            plan.current_step += 1
            state["plan"] = plan
        return state

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

        # Preparar un resumen de lo que se ha ejecutado
        plan: Plan = state.get("plan")
        partial_results = state.get("partial_results", {})

        summary_prompt = f"""
        Consulta original del usuario: {state['user_query']}

        Plan ejecutado:
        {chr(10).join([f"{i+1}. {step}" for i, step in enumerate(plan.steps)])}

        Resultados obtenidos:
        {chr(10).join([f"- Paso {k}: {v.get('step_description', 'Sin descripción')}" for k, v in partial_results.items()])}

        Instrucciones:
        1. Proporciona una respuesta clara y útil basada en los resultados obtenidos
        2. Incluye detalles específicos de las acciones realizadas
        3. Si se crearon playlists, menciona sus IDs y URLs
        4. Si se encontraron canciones, incluye los detalles relevantes
        5. Sé conciso pero informativo

        Genera una respuesta final coherente para el usuario.
        """

        messages = state["messages"] + [HumanMessage(content=summary_prompt)]
        final_response = self.model.invoke(messages)

        # Actualizar el estado con la respuesta final
        state["final_output"] = final_response
        state["messages"] = messages + [final_response]
        state["status"] = Status.DONE

        if self.verbose:
            print("✅ Resumen completado")

        return state

    def __exists_action(self, state: AgentState) -> bool:
        """Indica si el último mensaje contiene llamadas a herramientas.
        Args:
            state: estado actual con el historial de mensajes.
        Returns:
            ``True`` si el último mensaje del modelo incluye ``tool_calls``; en caso contrario ``False``.
        """
        result = state["messages"][-1]          # Último mensaje generado por el modelo
        return len(result.tool_calls) > 0       # Devuelve True si hay llamadas a herramientas en el último mensaje
    

    def __call_model(self, state: AgentState) -> AgentState:
        """
        Invoca el modelo LLM con el historial de mensajes.
        Inserta el mensaje de sistema una única vez si está configurado.
        Args:
            state: estado actual con los mensajes acumulados.
        Returns:
            Un nuevo estado con el mensaje de salida del modelo en ``messages``.
        """
        messages = state["messages"]
        message = self.model.invoke(messages)
        return {"messages": [message]}
    

    def __take_action(self, state: AgentState) -> AgentState:
        """
        Ejecuta las herramientas solicitadas por el modelo y retorna sus resultados.
        Args:
            state: estado actual cuyo último mensaje contiene ``tool_calls``.
        Returns:
            Un estado con los mensajes de tipo ``ToolMessage`` correspondientes a cada ejecución.
        """
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            message = self.__call_tool(t)
            results.append(message)
        return {"messages": results}

    def __should_continue_research(self, state: AgentState) -> bool:
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
                print(f"🔄 Continuando investigación - Paso {plan.current_step + 1}/{len(plan.steps)}")
            else:
                print("🏁 Investigación completada, procediendo a síntesis")
        
        return should_continue

    def __synthesize(self, state: AgentState) -> AgentState:
        """
        Sintetiza los resultados de la investigación en una respuesta final.
        Args:
            state: estado actual con todos los resultados de la investigación.
        Returns:
            Un estado actualizado con la respuesta final.
        """
        if self.verbose:
            print("🧩 Sintetizando resultados finales...")

        # Preparar un resumen de lo que se ha ejecutado
        plan: Plan = state.get("plan")
        partial_results = state.get("partial_results", {})
        
        synthesis_prompt = f"""
        Consulta original del usuario: {state['user_query']}
        
        Plan ejecutado:
        {chr(10).join([f"{i+1}. {step}" for i, step in enumerate(plan.steps)])}
        
        Resultados obtenidos:
        {chr(10).join([f"- Paso {k}: {v.get('step_description', 'Sin descripción')}" for k, v in partial_results.items()])}
        
        Instrucciones:
        1. Proporciona una respuesta clara y útil basada en los resultados obtenidos
        2. Incluye detalles específicos de las acciones realizadas
        3. Si se crearon playlists, menciona sus IDs y URLs
        4. Si se encontraron canciones, incluye los detalles relevantes
        5. Sé conciso pero informativo
        
        Genera una respuesta final coherente para el usuario.
        """
        
        messages = state["messages"] + [HumanMessage(content=synthesis_prompt)]
        final_response = self.model.invoke(messages)
        
        # Actualizar el estado con la respuesta final
        state["final_output"] = final_response
        state["messages"] = messages + [final_response]
        state["status"] = Status.DONE
        
        if self.verbose:
            print("✅ Síntesis completada")
        
        return state
    

    async def invoke(self, query: str) -> AsyncGenerator[str, None]:
        """
        Formula una pregunta al agente y devuelve la última respuesta.
        Args:
            question: texto de la consulta del usuario.
        Returns:
            Generador asíncrono que produce fragmentos de la respuesta final del agente.
        """
        # Guarda la consulta del usuario como un mensaje humano
        self.human_query = HumanMessage(content=query)
        # Ejecuta el grafo de modo asíncrono y obtiene los eventos
        events = self.graph.astream_events(input={"messages": [self.human_query]}, config=self.config)
        async for event in events:
            # Obtiene el tipo de evento
            kind = event["event"]
            # Si el evento es un fragmento de respuesta del modelo, lo "yieldea"
            if kind == "on_chat_model_stream":
                # Extrae el contenido del evento (fragmento) y lo "yieldea" (es como un "return" dentro de un async)
                yield event["data"]["chunk"].content


    def print_graph(self) -> None:
        """Imprime una representación del grafo del agente."""
        print(self.graph.get_graph().draw_mermaid())