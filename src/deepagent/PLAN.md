Quiero que actúes como un planificador.

# Tarea del usuario:
{task}

# Objetivo:
Generar un plan paso a paso para conseguir ese objetivo en formato JSON.

# Instrucciones:
1) Crea un plan numerado, claro y ordenado.
2) Cada paso debe ser concreto, accionable y comprensible.
3) Si faltan datos, pide aclaraciones antes de seguir.
4) Si hay varias formas de hacerlo, elige la más sencilla y eficiente.

# Formato de respuesta:
Responde ÚNICAMENTE con un JSON válido con esta estructura:
{{
    "steps": ["paso 1", "paso 2", "paso 3"],
    "current_step": null
}}

NO uses markdown, NO expliques nada, SOLO el JSON.