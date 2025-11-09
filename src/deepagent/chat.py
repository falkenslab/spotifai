import asyncio
from typing import AsyncGenerator

from deepagent.utils import bold
from deepagent.agent import DeepAgent

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

async def __handle_response(response) -> None:
    """
    Maneja la respuesta del agente, que puede ser un generador asíncrono.
    Si la respuesta es un generador, imprime los fragmentos a medida que llegan.
    Si es una respuesta completa, la imprime directamente.
    Args:
        response: Puede ser una cadena o un AsyncGenerator de cadenas.
    """
    if isinstance(response, AsyncGenerator):
        async for chunk in response:
            print(chunk, end="", flush=True)
    else:
        print(response)
    print("\n")  # Nueva línea al final

def chat(agent: DeepAgent, agent_name: str = "DeepAgent", human_name: str = "Tú", intro: str = "¡Hola!") -> None:
    """
    Inicia un chat interactivo entre el humano y el agente.
    Args:
        agent: Instancia del DeepAgent.
        agent_name: Nombre del agente para mostrar en el prompt.
        human_name: Nombre del humano para mostrar en el prompt.
        intro: Mensaje de introducción al iniciar el chat.
    """
    try:
        print(__agent_prompt(agent_name), intro, "\n")

        while True:
            human_input = input(__user_prompt(human_name) + " ")
            if human_input.lower() == "salir":
                break
            print()
            agent_response = agent.invoke(human_input)
            print(__agent_prompt(agent_name) + " ", end="", flush=True)
            asyncio.run(__handle_response(agent_response))
    except KeyboardInterrupt:
        print(f"\n\n{__agent_prompt(agent_name)}", f"¡Chao {human_name}!")