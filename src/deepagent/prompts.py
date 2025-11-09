from importlib import resources
from langchain_core.prompts import PromptTemplate

def read_resource_file(package: str, filename: str) -> str:
    """Lee el contenido de un archivo de recursos dentro de un paquete."""
    try:
        # Para Python 3.9+
        with resources.files(package).joinpath(filename).open("r", encoding="utf-8") as f:
            return f.read()
    except AttributeError:
        # Fallback para Python 3.7-3.8
        with resources.open_text(package, filename, encoding="utf-8") as f:
            return f.read()

PLAN_PROMPT_TEMPLATE = PromptTemplate(
    template=read_resource_file("deepagent", "PLAN.md"),
)

SYSTEM_PROMPT_TEMPLATE = PromptTemplate(
    template=read_resource_file("deepagent", "INSTRUCTIONS.md"),
    input_variables=[
        "domain",
        "tools",
        "output_format",
        "tone"            
    ],
)