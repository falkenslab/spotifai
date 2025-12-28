from pathlib import Path
from langchain_core.prompts import PromptTemplate

TEMPLATES = {
    "system": ("INSTRUCTIONS.md", ["domain", "tools", "tone"]),
    "plan": ("PLAN.md", ["domain"]),
    "research": ("RESEARCH.md", ["domain"]),
}

class PromptFactory:
    """Fábrica para cargar y gestionar plantillas de prompts desde archivos."""

    _assets_dir = Path(__file__).parent / "assets"
    _template_cache = {}

    @classmethod
    def read_asset(cls, filename: str) -> str:
        """Lee el contenido de un archivo desde el directorio assets."""
        file_path = cls._assets_dir / filename
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    @classmethod
    def create(cls, filename: str, input_variables: list[str] = None) -> PromptTemplate:
        """
        Crea una plantilla de prompt desde un archivo (con cache).
        Args:
            filename: Nombre del archivo en assets/
            input_variables: Variables que acepta el template. Si es None, se auto-detectan.
        """
        cache_key = (filename, tuple(input_variables) if input_variables else None)

        if cache_key in cls._template_cache:
            return cls._template_cache[cache_key]

        template_str = cls.read_asset(filename)
        template = (
            PromptTemplate.from_template(template_str)
            if input_variables is None
            else PromptTemplate(template=template_str, input_variables=input_variables)
        )

        cls._template_cache[cache_key] = template
        return template

    @classmethod
    def get(cls, name: str) -> PromptTemplate:
        """
        Obtiene un prompt predefinido por nombre.
        Args:
            name: Nombre del prompt ('plan', 'system', etc.)
        """
        if name not in TEMPLATES:
            raise ValueError(f"Prompt desconocido: {name}")
        
        filename, variables = TEMPLATES[name]
        return cls.create(filename, variables)

    @classmethod
    def render(cls, name: str, variables: dict) -> str:
        """
        Renderiza una plantilla predefinida con las variables proporcionadas.
        Args:
            name: Nombre del prompt ('plan', 'system', etc.)
            variables: Diccionario con las variables para renderizar
        Returns:
            String con la plantilla formateada
        """
        template = cls.get(name)
        return template.format(**variables)