"""
Ollama LLM Client Implementation

Encapsulates all Ollama model interactions using the Ollama REST API.
Provides a chat interface compatible with GeminiLLMClient and LlamaClient.
"""

import os
import sys
import requests
from typing import Optional, List, Any, Dict
from pathlib import Path
from dotenv import load_dotenv

# Add src directory to path if needed
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from cli import console
from .ollama_validation import OllamaConfig


class OlamaChatSession:
    """
    Wrapper class that provides a chat session interface compatible with Gemini's interface.
    Manages conversation history and provides send_message() and get_history() methods.
    Communicates with Ollama server via REST API.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        system_instruction: str,
        request_timeout: int = 300,
        history: Optional[List[Dict]] = None,
        top_k: int = 40,
        top_p: float = 0.9,
        temperature: float = 0.7
    ):
        """
        Initialize the Ollama chat session.

        Args:
            base_url: Base URL of the Ollama server (e.g., http://localhost:11434)
            model_name: Model name as recognized by Ollama
            system_instruction: System prompt for the assistant
            request_timeout: Timeout for HTTP requests in seconds
            history: Previous conversation history
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            temperature: Temperature for randomness control
        """
        self.base_url = base_url
        self.model_name = model_name
        self.system_instruction = system_instruction
        self.request_timeout = request_timeout
        self._history = history or []
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.api_endpoint = f"{base_url}/api/generate"

    def send_message(self, text: str) -> Any:
        """
        Sends a message to the Ollama model and returns a response object.

        Args:
            text: User's message

        Returns:
            Response object with .text attribute containing the response

        Raises:
            RuntimeError: If the request to Ollama fails
        """
        # Add user message to history
        user_message = {"role": "user", "parts": [{"text": text}]}
        self._history.append(user_message)

        # Build prompt from system instruction + conversation history
        prompt = self._build_prompt_from_history()

        try:
            # Send request to Ollama server with sampling parameters
            response = requests.post(
                self.api_endpoint,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "top_k": self.top_k,
                        "top_p": self.top_p,
                        "temperature": self.temperature,
                    }
                },
                timeout=self.request_timeout,
            )

            # Check for HTTP errors
            response.raise_for_status()

            # Parse response
            response_data = response.json()
            response_text = response_data.get("response", "").strip()

            # Add assistant response to history
            assistant_message = {"role": "model", "parts": [{"text": response_text}]}
            self._history.append(assistant_message)

            # Return response object compatible with Gemini interface
            return OlamaResponse(response_text)

        except requests.exceptions.Timeout:
            console.print_error(f"Timeout przy komunikacji z serwerem Ollama: {self.base_url}")
            error_text = "Przepraszam, serwer Ollama nie odpowiada w wyznaczonym czasie."
            assistant_message = {"role": "model", "parts": [{"text": error_text}]}
            self._history.append(assistant_message)
            return OlamaResponse(error_text)

        except requests.exceptions.ConnectionError:
            console.print_error(f"Nie można połączyć się z serwerem Ollama: {self.base_url}")
            error_text = "Przepraszam, serwer Ollama jest niedostępny. Sprawdź czy serwer jest uruchomiony."
            assistant_message = {"role": "model", "parts": [{"text": error_text}]}
            self._history.append(assistant_message)
            return OlamaResponse(error_text)

        except requests.exceptions.RequestException as e:
            console.print_error(f"Błąd żądania HTTP do Ollamy: {e}")
            error_text = "Przepraszam, wystąpił błąd podczas komunikacji z serwerem."
            assistant_message = {"role": "model", "parts": [{"text": error_text}]}
            self._history.append(assistant_message)
            return OlamaResponse(error_text)

        except Exception as e:
            console.print_error(f"Błąd podczas generowania odpowiedzi Ollama: {e}")
            error_text = "Przepraszam, wystąpił błąd podczas generowania odpowiedzi."
            assistant_message = {"role": "model", "parts": [{"text": error_text}]}
            self._history.append(assistant_message)
            return OlamaResponse(error_text)

    def get_history(self) -> List[Dict]:
        """Returns the current conversation history."""
        return self._history

    def _build_prompt_from_history(self) -> str:
        """
        Builds a prompt string from the conversation history and system instruction.

        Returns:
            Formatted prompt string for the Ollama model
        """
        prompt_parts = []

        # Add system instruction
        if self.system_instruction:
            prompt_parts.append(f"System: {self.system_instruction}")

        # Add conversation history (exclude the last message which is current user input)
        for message in self._history[:-1]:
            role = message["role"]
            text = message["parts"][0]["text"]

            if role == "user":
                prompt_parts.append(f"User: {text}")
            elif role == "model":
                prompt_parts.append(f"Assistant: {text}")

        # Add the current user message
        if self._history:
            last_message = self._history[-1]
            if last_message["role"] == "user":
                user_text = last_message["parts"][0]["text"]
                prompt_parts.append(f"User: {user_text}")

        prompt_parts.append("Assistant:")

        return "\n\n".join(prompt_parts)


class OlamaResponse:
    """
    Response object that mimics the Gemini response interface.
    Provides a .text attribute containing the response text.
    """

    def __init__(self, text: str):
        self.text = text


class OllamaClient:
    """
    Encapsulates all Ollama model interactions via REST API.
    Provides a clean interface compatible with GeminiLLMClient and LlamaClient.
    """

    def __init__(
        self,
        model_name: str,
        ollama_model_name: str,
        base_url: str = "http://localhost:11434",
        request_timeout: int = 300,
        top_k: int = 40,
        top_p: float = 0.9,
        temperature: float = 0.7
    ):
        """
        Initialize the Ollama client.

        Args:
            model_name: Display name for the model
            ollama_model_name: Model name as recognized by Ollama (e.g., 'llama2', 'mistral')
            base_url: Base URL of the Ollama server
            request_timeout: Timeout for HTTP requests in seconds
            top_k: Top-k sampling parameter (default: 40)
            top_p: Top-p (nucleus) sampling parameter (default: 0.9)
            temperature: Temperature for randomness control (default: 0.7)

        Raises:
            RuntimeError: If Ollama server is not accessible
        """
        self.model_name = model_name
        self.ollama_model_name = ollama_model_name
        self.base_url = base_url.rstrip('/')
        self.request_timeout = request_timeout
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.api_endpoint = f"{self.base_url}/api/generate"

        # Verify server is accessible
        if not self._check_server_availability():
            raise RuntimeError(
                f"Serwer Ollama jest niedostępny pod adresem: {self.base_url}"
            )

    def _check_server_availability(self) -> bool:
        """
        Checks if the Ollama server is accessible.

        Returns:
            True if server is accessible, False otherwise
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    @staticmethod
    def preparing_for_use_message() -> str:
        """
        Returns a message indicating that Ollama client is being prepared.

        Returns:
            Formatted preparation message string
        """
        return "🦙 Przygotowywanie klienta Ollama..."

    @classmethod
    def from_environment(cls) -> 'OllamaClient':
        """
        Factory method that creates an OllamaClient instance from environment variables.

        Returns:
            OllamaClient instance initialized with environment variables

        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If Ollama server is not accessible
        """
        load_dotenv()

        # Validate configuration with Pydantic
        config = OllamaConfig(
            model_name=os.getenv('OLLAMA_MODEL_NAME', 'gemma3:4b'),
            ollama_model_name=os.getenv('OLLAMA_MODEL_NAME_ACTUAL', 'gemma3:4b'),
            ollama_base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
            ollama_request_timeout=int(os.getenv('OLLAMA_REQUEST_TIMEOUT', '300')),
            ollama_top_k=int(os.getenv('OLLAMA_TOP_K', '40')),
            ollama_top_p=float(os.getenv('OLLAMA_TOP_P', '0.9')),
            ollama_temperature=float(os.getenv('OLLAMA_TEMPERATURE', '0.7'))
        )

        console.print_info(f"Łączenie z serwerem Ollama: {config.ollama_base_url}")

        return cls(
            model_name=config.model_name,
            ollama_model_name=config.ollama_model_name,
            base_url=config.ollama_base_url,
            request_timeout=config.ollama_request_timeout,
            top_k=config.ollama_top_k,
            top_p=config.ollama_top_p,
            temperature=config.ollama_temperature
        )

    def create_chat_session(
        self,
        system_instruction: str,
        history: Optional[List[Dict]] = None,
        thinking_budget: int = 0
    ) -> OlamaChatSession:
        """
        Creates a new chat session with the specified configuration.

        Args:
            system_instruction: System role/prompt for the assistant
            history: Previous conversation history (optional)
            thinking_budget: Ignored for Ollama (compatibility parameter)

        Returns:
            OlamaChatSession object
        """
        return OlamaChatSession(
            base_url=self.base_url,
            model_name=self.ollama_model_name,
            system_instruction=system_instruction,
            request_timeout=self.request_timeout,
            history=history or [],
            top_k=self.top_k,
            top_p=self.top_p,
            temperature=self.temperature
        )

    def count_history_tokens(self, history: List[Dict]) -> int:
        """
        Counts tokens for the given conversation history.
        Note: This is an approximation using character-based estimation.

        Args:
            history: Conversation history

        Returns:
            Estimated token count
        """
        if not history:
            return 0

        try:
            # Build text from history
            text_parts = []
            for message in history:
                if "parts" in message and message["parts"]:
                    text_parts.append(message["parts"][0]["text"])

            full_text = " ".join(text_parts)

            # Use character-based approximation (roughly 4 chars per token)
            # This is a common approximation for token counting
            token_count = len(full_text.encode('utf-8')) // 4

            return max(1, token_count)  # At least 1 token

        except Exception as e:
            console.print_error(f"Błąd podczas szacowania tokenów: {e}")
            return 0

    def get_model_name(self) -> str:
        """Returns the currently configured model name."""
        return self.model_name

    def is_available(self) -> bool:
        """
        Checks if the Ollama service is available.

        Returns:
            True if server is accessible
        """
        return self._check_server_availability()

    def ready_for_use_message(self) -> str:
        """
        Returns a ready-to-use message with model info and parameters.

        Returns:
            Formatted message string for display
        """
        return (
            f"✅ Klient Ollama gotowy do użycia "
            f"(model: {self.model_name}, serwer: {self.base_url}, timeout: {self.request_timeout}s)"
        )
