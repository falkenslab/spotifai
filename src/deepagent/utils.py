
def bold(text: str) -> str:
    return f"\033[1m{text}\033[0m"

def confirm_action(message: str, default_yes: bool = False) -> dict:
    """Pide confirmación al usuario por consola (s/n).

    Args:
        message (str): Mensaje de confirmación a mostrar.
        default_yes (bool): Si True, Enter cuenta como 'sí'; en caso contrario, 'no'.

    Returns:
        dict: {"confirmed": bool, "answer": "s"|"n"}
    """
    print("Confirmación requerida:", message, default_yes)
    suffix = "[S/n]" if default_yes else "[s/N]"
    prompt = f"{message} {suffix} "
    while True:
        try:
            ans = input(prompt).strip().lower()
        except EOFError:
            print("mmmh...!!!")
            # Entorno no interactivo: usar el valor por defecto
            return {"confirmed": default_yes, "answer": "s" if default_yes else "n"}

        if ans == "":
            return {"confirmed": default_yes, "answer": "s" if default_yes else "n"}
        if ans in ("s", "si", "sí", "y", "yes"):
            return {"confirmed": True, "answer": "s"}
        if ans in ("n", "no"):
            return {"confirmed": False, "answer": "n"}
        print("Por favor, responde 's' o 'n'.")