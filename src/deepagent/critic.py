from pydantic import BaseModel, Field


class CriticEvaluation(BaseModel):
    """Evaluación del critic sobre la completitud de un paso del plan."""

    completed: bool = Field(
        description="Indica si el paso del plan se completó correctamente",
        examples=[True, False]
    )

    summary: str = Field(
        description="Resumen breve de lo que se logró en este paso",
        examples=["Se buscaron 10 canciones de rock clásico y se añadieron a la playlist."]
    )

    issues: str = Field(
        default="",
        description="Problemas detectados, o cadena vacía si todo está bien",
        examples=["La herramienta devolvió un error 404.", ""]
    )

    def __str__(self) -> str:
        status = "✅" if self.completed else "❌"
        result = f"{status} Completado: {self.completed}\n📋 Resumen: {self.summary}"
        if self.issues:
            result += f"\n⚠️ Problemas: {self.issues}"
        return result
