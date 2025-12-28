from typing import List, Optional
from pydantic import BaseModel, Field

class Plan(BaseModel):
    """Plan de acción estructurado para el agente."""    

    steps: List[str] = Field(
        description="Lista de pasos a seguir en el plan",
        min_length=1,
        examples=[
            ["Recopilar información necesaria", "Procesar y analizar datos", "Generar resultado final"]
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

    def next_step(self) -> Optional[str]:
        """Obtiene el siguiente paso en el plan, si existe."""
        if self.current_step is not None and self.current_step < len(self.steps):
            step = self.steps[self.current_step]
            self.current_step += 1
            return step
        return None
    
    def __str__(self):
        return f"Plan(steps={self.steps}, current_step={self.current_step})"

    def pretty(self) -> str:
        output = "Plan de Acción:\n"
        for idx, step in enumerate(self.steps):
            marker = "->" if idx == self.current_step else "  "
            output += f"  {marker} Paso {idx + 1}: {step}\n"
        return output
    
    def pretty_print(self):
        print(self.pretty())
