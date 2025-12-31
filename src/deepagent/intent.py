
from typing import Optional
from pydantic import BaseModel, Field

class Intent(BaseModel):
    """Representa la intención derivada de un paso del plan."""

    goal: str = Field(
        description="Breve descripción del objetivo del paso",
        examples=["Buscar canciones de rock clásico", "Crear una playlist privada"]
    )
    
    notes: Optional[str] = Field(
        default=None,
        description="Cualquier nota relevante para la ejecución del paso",
        examples=["Priorizar canciones de los años 70 y 80", "Incluir al menos 10 canciones"]
    )

    def __str__(self):
        return f"Intent(goal={self.goal}, notes={self.notes})"
    
    def pretty_print(self):
        print("Intención:")
        print(f"  - Objetivo: {self.goal}")
        if self.notes:
            print(f"  - Notas: {self.notes}")