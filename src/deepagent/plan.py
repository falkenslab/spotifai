from typing import List, Optional
from pydantic import BaseModel, Field

class Plan(BaseModel):
    """Plan de acción estructurado para el agente."""    

    steps: List[str] = Field(
        description="Lista de pasos a seguir en el plan",
        min_length=1,
        examples=[
            ["Buscar canciones de rock clásico", "Crear playlist privada", "Añadir canciones encontradas"]
        ]
    )
    
    current_step: Optional[int] = Field(
        default=None,
        description="Índice del paso actual en el plan (0-based)",
        ge=0,
        examples=[0, 1, 2]
    )

    def __init__(self, **data):
        super().__init__(**data)
        if self.current_step is None:
            self.current_step = 0