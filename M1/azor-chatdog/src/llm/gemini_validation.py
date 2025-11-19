from pydantic import BaseModel, Field, validator
from typing import Optional, Literal

class GeminiConfig(BaseModel):
    engine: Literal["GEMINI"] = Field(default="GEMINI")
    model_name: str = Field(..., description="Nazwa modelu Gemini")
    gemini_api_key: str = Field(..., min_length=1, description="Klucz API Google Gemini")
    gemini_top_k: int = Field(default=40, ge=-1, description="Top-k sampling parameter")
    gemini_top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p (nucleus) sampling parameter")
    gemini_temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature for randomness control")

    @validator('gemini_api_key')
    def validate_api_key(cls, v):
        if not v or v.strip() == "":
            raise ValueError("GEMINI_API_KEY nie może być pusty")
        return v.strip()
