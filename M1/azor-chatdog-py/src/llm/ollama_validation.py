"""
Ollama Configuration Validation using Pydantic

Validates environment variables for Ollama client configuration.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Literal
import os
from pathlib import Path

# Add src directory to path if needed
import sys
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))


class OllamaConfig(BaseModel):
    """
    Configuration model for Ollama client.

    Validates all required and optional parameters for connecting to an Ollama server.
    """
    engine: Literal["OLLAMA"] = Field(default="OLLAMA")
    model_name: str = Field(..., description="Display name for the model")
    ollama_model_name: str = Field(..., description="Model name as recognized by Ollama (e.g., 'llama2', 'mistral')")
    ollama_base_url: str = Field(default="http://localhost:11434", description="Base URL of the Ollama server")
    ollama_request_timeout: int = Field(default=300, ge=1, description="Request timeout in seconds")
    ollama_top_k: int = Field(default=40, ge=-1, description="Top-k sampling parameter")
    ollama_top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p (nucleus) sampling parameter")
    ollama_temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature for randomness control")

    @validator('ollama_base_url')
    def validate_base_url(cls, v):
        """Validate that base URL is properly formatted."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError("Base URL must start with http:// or https://")
        # Remove trailing slash if present
        return v.rstrip('/')

    @validator('ollama_model_name')
    def validate_model_name(cls, v):
        """Validate that model name is not empty and is reasonable."""
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        # Model names in Ollama are lowercase with optional version tags
        if any(char in v for char in ['/', '\\']):
            raise ValueError("Model name contains invalid characters")
        return v.lower()

    @validator('model_name')
    def validate_display_name(cls, v):
        """Validate that display name is not empty."""
        if not v or not v.strip():
            raise ValueError("Display name cannot be empty")
        return v
