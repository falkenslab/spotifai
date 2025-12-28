
def bold(text: str) -> str:
    return f"\033[1m{text}\033[0m"

def extract_code_blocks(text: str) -> list[dict[str, str]]:
    """Extrae bloques de código delimitados por triples backticks de un texto.
    
    Returns:
        Lista de diccionarios con 'type' y 'content' de cada bloque de código.
        Si no hay tipo especificado, 'type' será una cadena vacía.
    """
    import re
    pattern = r"```([a-zA-Z]*)\n?(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    
    result = []
    for language_type, content in matches:
        result.append({
            "type": language_type.strip(),
            "content": content.strip()
        })
    
    return result