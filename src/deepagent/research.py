from pydantic import BaseModel, Field

class Intent(BaseModel):
    """Resultado estructurado de una investigación realizada por el agente."""
    
    goal: str = Field(
        description="Breve descripción del objetivo del paso",
        examples=["Investigar las tendencias actuales en IA para mejorar el modelo"]
    )
    
    notes: str = Field(
        description="Cualquier nota relevante para la ejecución del paso",
        examples=["Se encontraron varios artículos recientes que destacan el uso de transformers."]
    )