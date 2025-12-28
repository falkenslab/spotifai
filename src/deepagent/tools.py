"""
Utilitario para generar automáticamente descripciones de herramientas desde decoradores @tool
"""
import inspect
from typing import List, Any, get_type_hints, get_origin, get_args
from langchain_core.tools.base import BaseTool

def format_type_hint(type_hint) -> str:
    """Convierte una anotación de tipo en string legible."""
    if type_hint is None:
        return "None"
    
    # Manejar tipos genéricos como List[dict], Optional[str], etc.
    origin = get_origin(type_hint)
    args = get_args(type_hint)
    
    if origin is not None:
        if origin is list:
            if args:
                return f"List[{format_type_hint(args[0])}]"
            return "List"
        elif origin is dict:
            if len(args) == 2:
                return f"dict[{format_type_hint(args[0])}, {format_type_hint(args[1])}]"
            return "dict"
        elif origin is type(None) or str(origin) == "typing.Union":
            # Manejar Optional[T] que es Union[T, None]
            if len(args) == 2 and type(None) in args:
                non_none_type = args[0] if args[1] is type(None) else args[1]
                return f"Optional[{format_type_hint(non_none_type)}]"
            return f"Union[{', '.join(format_type_hint(arg) for arg in args)}]"
    
    # Tipos básicos
    if hasattr(type_hint, '__name__'):
        return type_hint.__name__
    
    return str(type_hint)


def extract_tool_signature(tool: BaseTool) -> str:
    """Extrae la signatura de una herramienta con tipos."""
    func = tool.func if hasattr(tool, 'func') else None
    if not func:
        return f"{tool.name}(...)"
    
    # Obtener signature y type hints
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)
    
    # Construir parámetros con tipos
    params = []
    for param_name, param in sig.parameters.items():
        # Obtener tipo
        param_type = type_hints.get(param_name, Any)
        type_str = format_type_hint(param_type)
        
        # Construir parámetro con valor por defecto si existe
        if param.default is not inspect.Parameter.empty:
            if isinstance(param.default, str):
                default_str = f'"{param.default}"'
            else:
                default_str = str(param.default)
            params.append(f"{param_name}: {type_str} = {default_str}")
        else:
            params.append(f"{param_name}: {type_str}")
    
    # Obtener tipo de retorno
    return_type = type_hints.get('return', Any)
    return_type_str = format_type_hint(return_type)
    
    return f"{tool.name}({', '.join(params)}) -> {return_type_str}"


def extract_tool_description(tool: BaseTool) -> str:
    """Extrae la descripción de una herramienta desde su docstring."""
    if tool.description:
        # Usar la primera línea del description si existe
        return tool.description.split('\n')[0].strip()
    
    func = tool.func if hasattr(tool, 'func') else None
    if func and func.__doc__:
        # Usar la primera línea del docstring
        lines = func.__doc__.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('Args:') and not line.startswith('Returns:'):
                return line
    
    return "Herramienta sin descripción"


def generate_tools_description(tools: List[BaseTool]) -> str:
    """Genera automáticamente la descripción de herramientas para el prompt."""
    tool_descriptions = []
    
    for tool in tools:
        signature = extract_tool_signature(tool)
        description = extract_tool_description(tool)
        tool_descriptions.append(f"- {signature}: {description}")
    
    return '\n        '.join(tool_descriptions)


def print_tools_debug(tools: List[BaseTool]) -> None:
    """Imprime información de debug sobre las herramientas."""
    print("=== DEBUG: Herramientas detectadas ===")
    for tool in tools:
        print(f"Nombre: {tool.name}")
        print(f"Signatura: {extract_tool_signature(tool)}")
        print(f"Descripción: {extract_tool_description(tool)}")
        print("---")