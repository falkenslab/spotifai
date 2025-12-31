import asyncio
from typing import AsyncGenerator

from deepagent.utils import bold
from deepagent.agent import DeepAgent
from deepagent.chunk import Chunk

def __user_prompt(human_name) -> str:
    """
    Genera el prompt del usuario con formato.
    Args:
        human_name: Nombre del usuario.
    """
    return f"🙂 {bold(human_name)}:"

def __agent_prompt(agent_name) -> None:
    """
    Genera el prompt del agente con formato.
    Args:
        agent_name: Nombre del agente.
    """
    return f"🤖 {bold(agent_name)}:"

async def __handle_response(agent_name: str, response: AsyncGenerator[Chunk, None], verbose: bool = False) -> None:
    """
    Maneja la respuesta del agente, que es un generador asíncrono, imprimiendo los fragmentos a medida que llegan.    
    Args:
        response: Puede ser una cadena o un AsyncGenerator de cadenas.
    """
    thinking = True
    async for chunk in response:
        thinking = chunk.type != "text"
        if not thinking:
            print(__agent_prompt(agent_name) + " ", end="", flush=True)
        if chunk.type == "text" or verbose:
            print(f"\033[90m{chunk.content}\033[0m", end="", flush=True)
    print("\n")  # Nueva línea al final

def chat(agent: DeepAgent, agent_name: str = "DeepAgent", human_name: str = "Tú", intro: str = "¡Hola!", verbose: bool = True) -> None:
    """
    Inicia un chat interactivo entre el humano y el agente.
    Args:
        agent: Instancia del DeepAgent.
        agent_name: Nombre del agente para mostrar en el prompt.
        human_name: Nombre del humano para mostrar en el prompt.
        intro: Mensaje de introducción al iniciar el chat.
        verbose: Si es True, muestra información adicional de depuración.
    """
    try:
        print(f"\n{__agent_prompt(agent_name)} {intro}\n")

        while True:
            human_input = input(__user_prompt(human_name) + " ")
            if human_input.lower() == "salir":
                break
            print()
            agent_response = agent.invoke(human_input)
            asyncio.run(__handle_response(agent_name, agent_response, verbose=verbose))
    except KeyboardInterrupt:
        print(f"\n\n{__agent_prompt(agent_name)}", f"¡Chao {human_name}!")