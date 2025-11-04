import sys
import json
import asyncio
from typing import AsyncGenerator

from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import InMemorySaver

from deepagent.plan import Plan
from deepagent.prompts import PLAN_PROMPT_TEMPLATE
from deepagent.state import AgentState, Status
from deepagent.utils import bold

class DeepAgent:
    """Agente orquestador que combina un modelo LLM con herramientas.

    Construye un grafo de estados con dos nodos principales:
    - "llm": invoca el modelo para generar la siguiente acción o respuesta.
    - "action": ejecuta herramientas solicitadas por el modelo y retorna sus resultados.

    El grafo itera hasta que el modelo no solicite más herramientas.

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
        # Construcción del grafo de estados del agente
        graph = StateGraph(AgentState)

        # Definición de nodos del grafo
        graph.add_node("init", self.__init)                     # Nodo de inicialización del estado del agente
        graph.add_node("planner", self.__plan)                 # Nodo de planificación
        #graph.add_node("llm", self.__call_model)                 # Nodo de llamada al modelo
        #graph.add_node("action", self.__take_action)              # Nodo de ejecución de herramientas (acciones)

        # Definición de aristas del grafo
        graph.add_edge("init", "planner")                           # Desde el estado inicial, ir al modelo
        graph.add_edge("planner", END)                             # Si el modelo no pide herramientas, terminar
        ##graph.add_conditional_edges(
        ##    "llm",                                              # La arista condicional sale del nodo "llm"
        ##    self.__exists_action,                                 # Función que decide si se debe ir al nodo de acción o terminar
        ##    {True: "action", False: END},
        ##)                                                       # Si el modelo decide llamar a una herramienta, ir al nodo de acción; si no, terminar
        ##graph.add_edge("action", "llm")                         # Después de ejecutar una acción, volver al modelo

        # Definición del punto de entrada del grafo
        graph.set_entry_point("init")

        # Compilación del grafo para su ejecución
        return graph.compile(checkpointer=InMemorySaver())
    

    def __call_tool(self, tool_call) -> ToolMessage:
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
    

    def __init(self, state: AgentState) -> AgentState:
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

        messages = state["messages"] + [SystemMessage(content=query)]
        message = self.model.invoke(messages)
        
        try:
            # Intentar parsear el JSON directamente
            plan_data = json.loads(message.content.strip())
            plan = Plan(**plan_data)
        except (json.JSONDecodeError, Exception) as e:
            # Plan por defecto en caso de error
            plan = Plan(
                steps=["Analizar la consulta del usuario", "Ejecutar acciones necesarias", "Proporcionar respuesta"],
                current_step=0
            )

        state["plan"] = plan
        state["messages"] = messages + message
        return state
    

    def __exists_action(self, state: AgentState) -> bool:
        """Indica si el último mensaje contiene llamadas a herramientas.
        Args:
            state: estado actual con el historial de mensajes.
        Returns:
            ``True`` si el último mensaje del modelo incluye ``tool_calls``; en caso contrario ``False``.
        """
        self._current_state = state
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
        self._current_state = state
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
        self._current_state = state
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            if self.verbose:
                print(f"\nEjecutando acción: {t}\n")
            if not t["name"] in self.tools:  # check for bad tool name from LLM
                if self.verbose:
                    print("\n ....nombre de tool no válida....")
                result = "nombre de tool no válida, reintentar"  # instruir al LLM a reintentar si el nombre es incorrecto
            else:
                result = self.tools[t["name"]].invoke(t["args"])
            message = ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result))
            if self.verbose:
                message.pretty_print()
            results.append(message)
        return {"messages": results}
    

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


    def __user_prompt(self, human_name) -> str:
        return f"🙂 {bold(human_name)}: "

    def __agent_prompt(self, agent_name) -> None:
        return f"🤖 {bold(agent_name)}: "
    
    async def __handle_response(self, response):
        print(self.__agent_prompt(self.agent_name), end="", flush=True)
        async for chunk in response:
            print(chunk, end="", flush=True)
        print("\n")  # Nueva línea al final

    def chat(self, agent_name: str = "DeepAgent", human_name: str = "Tú") -> None:
        try:
            while True:
                user_input = input(self.__user_prompt(human_name))
                if user_input.lower() == "salir":
                    break   
                print()
                response = self.invoke(user_input)
                asyncio.run(self.__handle_response(response))                
        except KeyboardInterrupt:
            print(f"\n{self.__agent_prompt(agent_name)}¡Chao {human_name}!")


    def print_graph(self) -> None:
        """Imprime una representación del grafo del agente."""
        print(self.graph.get_graph().draw_mermaid())