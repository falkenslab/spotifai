from pydantic import BaseModel, Field
from enum import Enum

class ChunkType(Enum):
    TEXT = "text"
    THINKING = "thinking"

class Chunk(BaseModel):

    content: str = Field(
        description="Contenido del fragmento de texto.",
        examples=[
            "Este es un fragmento de texto que contiene información relevante."
        ]
    )

    type: ChunkType = Field(
        description="Tipo de fragmento.",
        examples=["text", "thinking"]
    )

    def __init__(self, **data):
        super().__init__(**data)        

    def is_text(self) -> bool:
        """Indica si el fragmento es de tipo texto."""
        return self.type == ChunkType.TEXT

    def is_thinking(self) -> bool:
        """Indica si el fragmento es de tipo pensamiento."""
        return self.type == ChunkType.THINKING

    def __str__(self) -> str:
        return f"Chunk(type={self.type}, content={self.content})"