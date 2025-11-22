from pydantic import BaseModel, Field, validator
from typing import Optional, Literal
import os

class LlamaConfig(BaseModel):
    engine: Literal["LLAMA"] = Field(default="LLAMA")
    model_name: str = Field(..., description="Nazwa modelu Llama")
    llama_model_path: str = Field(..., description="Ścieżka do pliku modelu .gguf")
    llama_gpu_layers: int = Field(default=1, ge=0, description="Liczba warstw GPU")
    llama_context_size: int = Field(default=2048, ge=1, description="Rozmiar kontekstu")
    llama_top_k: int = Field(default=40, ge=-1, description="Top-k sampling parameter")
    llama_top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p (nucleus) sampling parameter")
    llama_temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature for randomness control")
    
    @validator('llama_model_path')
    def validate_model_path(cls, v):
        if not os.path.exists(v):
            raise ValueError(f"Plik modelu nie istnieje: {v}")
        if not v.endswith('.gguf'):
            raise ValueError("Plik modelu musi mieć rozszerzenie .gguf")
        return v
