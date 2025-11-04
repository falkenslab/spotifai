import os

from langchain.chat_models import init_chat_model

from spotifai.spotify_tools import init_spotify_manager

from deepagent.chat import chat
from deepagent.prompts import SYSTEM_PROMPT_TEMPLATE
from deepagent.tool_formatter import generate_tools_description
from deepagent.agent import DeepAgent

def init_model():
    model_name = os.getenv("DEFAULT_MODEL_NAME", "gpt-4o")
    model_provider = os.getenv("DEFAULT_MODEL_PROVIDER", "openai")
    model = init_chat_model(
        model=model_name,
        model_provider=model_provider,
        temperature=0.7
    )
    return model
  
def main():

    # Inicializar el gestor de Spotify y las herramientas (gestiona el inicio de sesión en Spotify si es necesario)
    current_user, spotify_tools = init_spotify_manager()
    
    # Generar descripción automática de herramientas
    tools_description = generate_tools_description(spotify_tools)

    # Crear el prompt del sistema
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        domain="Eres experto en música de todos los géneros, historia musical, energía, BPM y creación de playlists en Spotify.",
        tools=tools_description,
        output_format="Devuelve las canciones en JSON, con artista, año y BPM.",
        tone="Cercano, profesional y entusiasta."
    )

    # Inicializar el LLM
    model = init_model()

    # Crear el agente DeepAgent con las herramientas de Spotify y el prompt del sistema
    agent = DeepAgent(
        model=model,
        system=system_prompt,
        tools=spotify_tools,
        verbose=True
    )

    # Iniciar el chat con el agente
    chat(
        agent=agent, 
        agent_name="SpotifAgent", 
        human_name=current_user, 
        intro="¡Hola! Soy tu asistente de IA para ayudarte a gestionar tus playlist de Spotify. ¿En qué puedo ayudarte hoy?"
    )

if __name__ == "__main__":
    main()
