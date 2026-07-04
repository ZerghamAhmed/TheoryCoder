"""LLM client abstraction with strategy pattern for different providers.

This module provides a unified interface for querying different LLM providers
(OpenAI, Groq, custom API gateways, etc.) with timing and token tracking.
"""
import os
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import requests


@dataclass
class LLMResponse:
    """Standardized response from LLM query."""
    content: str
    raw_response: Any
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def query(self, prompt: str, **kwargs) -> LLMResponse:
        """Query the LLM with a prompt and return the response."""
        pass


class OpenAIDirectProvider(LLMProvider):
    """OpenAI direct API provider."""

    def __init__(self, model: str = "gpt-4o-2024-11-20", temperature: float = 1.0):
        import openai
        self.client = openai
        self.model = model
        self.temperature = temperature

    def query(self, prompt: str, **kwargs) -> LLMResponse:
        messages = [{"role": "user", "content": prompt}]
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            seed=42,
        )

        # Extract token usage
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None
        try:
            usage = getattr(completion, "usage", None)
            if usage:
                prompt_tokens = getattr(usage, "prompt_tokens", None)
                completion_tokens = getattr(usage, "completion_tokens", None)
                total_tokens = getattr(usage, "total_tokens", None)
        except Exception:
            pass

        return LLMResponse(
            content=completion.choices[0].message.content.strip(),
            raw_response=completion,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )


class GroqProvider(LLMProvider):
    """Groq API provider."""

    def __init__(self, model: str = "llama3-8b-8192"):
        from groq import Groq
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = model

    def query(self, prompt: str, **kwargs) -> LLMResponse:
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model
        )
        return LLMResponse(
            content=response.choices[0].message.content.strip(),
            raw_response=response,
        )


class CustomAPIProvider(LLMProvider):
    """Custom OpenAI-compatible API gateway provider."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "gpt-4o-2024-11-20",
        temperature: float = 1.0,
    ):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.temperature = temperature

    def query(self, prompt: str, **kwargs) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }

        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        completion = r.json()

        # Extract token usage
        usage = completion.get("usage", {})

        return LLMResponse(
            content=completion["choices"][0]["message"]["content"].strip(),
            raw_response=completion,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
        )


class LangchainOpenAIProvider(LLMProvider):
    """LangChain OpenAI provider (legacy)."""

    def __init__(self, model: str = "gpt-4o-2024-11-20", temperature: float = 1.0):
        # Import only when needed to avoid hard dependency
        from langchain.prompts.chat import HumanMessagePromptTemplate
        # from langchain.chat_models import ChatOpenAI
        self.HumanMessagePromptTemplate = HumanMessagePromptTemplate
        self.model = model
        self.temperature = temperature
        self.client = None  # Will be initialized if needed

    def query(self, prompt: str, **kwargs) -> LLMResponse:
        chat_prompt = self.HumanMessagePromptTemplate.from_template(prompt)
        out = self.client.invoke(chat_prompt.to_messages())
        return LLMResponse(
            content=out.content,
            raw_response=None,
        )


class LLMClient:
    """Unified LLM client with timing and metrics tracking.

    This class provides a consistent interface for querying LLMs regardless
    of the underlying provider, while tracking timing and token usage.
    """

    def __init__(
        self,
        query_mode: str = "openai_direct",
        language_model: str = "gpt-4o-2024-11-20",
        temperature: float = 1.0,
        groq_model: str = "llama3-8b-8192",
        custom_base_url: Optional[str] = None,
        custom_api_key: Optional[str] = None,
    ):
        """Initialize the LLM client.

        Args:
            query_mode: Provider mode ('openai_direct', 'groq', 'custom', 'langchain_openai')
            language_model: Model name for OpenAI-style providers
            temperature: Sampling temperature
            groq_model: Model name for Groq provider
            custom_base_url: Base URL for custom API gateway
            custom_api_key: API key for custom gateway
        """
        self.query_mode = query_mode
        self.language_model = language_model
        self.temperature = temperature

        # Initialize timing
        self.timing: Dict[str, Any] = {"events": [], "rollup": {}}

        # Create the appropriate provider
        if query_mode == "openai_direct":
            self._provider = OpenAIDirectProvider(language_model, temperature)
        elif query_mode == "groq":
            self._provider = GroqProvider(groq_model)
        elif query_mode == "custom":
            if not custom_base_url:
                custom_base_url = os.environ.get("CUSTOM_BASE_URL")
            if not custom_api_key:
                custom_api_key = os.environ.get("CUSTOM_API_KEY")
            if not custom_base_url:
                raise ValueError("CUSTOM_BASE_URL must be set for query_mode='custom'")
            if not custom_api_key:
                raise ValueError("CUSTOM_API_KEY must be set for query_mode='custom'")
            self._provider = CustomAPIProvider(
                custom_base_url, custom_api_key, language_model, temperature
            )
        elif query_mode == "langchain_openai":
            self._provider = LangchainOpenAIProvider(language_model, temperature)
        else:
            raise ValueError(f"Unsupported query_mode: {query_mode}")

    def query(self, prompt: str, *, label: Optional[str] = None) -> Tuple[str, Any]:
        """Query the LLM with timing and metrics tracking.

        Args:
            prompt: The prompt to send to the LLM
            label: Optional label for timing tracking

        Returns:
            Tuple of (response_content, raw_response)
        """
        meta_extra = {"model": self.language_model, "provider": self.query_mode}

        with self._record_time("llm", detail=label or self.query_mode, extra=meta_extra):
            response = self._provider.query(prompt)

        # Record token usage if available
        if response.total_tokens is not None:
            self.timing["events"][-1].update({
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "total_tokens": response.total_tokens,
            })

        return response.content, response.raw_response

    @contextmanager
    def _record_time(self, category: str, detail: Optional[str] = None, extra: Optional[dict] = None):
        """Context manager to capture wall time for any block."""
        t0 = time.perf_counter()
        err = None
        try:
            yield
        except Exception as e:
            err = str(e)
            raise
        finally:
            dt = time.perf_counter() - t0
            entry = {
                "ts": time.time(),
                "category": category,
                "detail": detail or "",
                "duration_s": dt,
            }
            if extra:
                entry.update(extra)
            if err:
                entry["error"] = err
            self.timing["events"].append(entry)

    def rollup_timings(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Compute rollup statistics from timing events."""
        out: Dict[Tuple[str, str], List[float]] = {}
        for ev in self.timing["events"]:
            key = (ev["category"], ev.get("detail", ""))
            out.setdefault(key, []).append(ev["duration_s"])

        roll: Dict[str, Dict[str, Dict[str, float]]] = {}
        for (cat, det), arr in out.items():
            roll.setdefault(cat, {})
            roll[cat][det] = {
                "count": len(arr),
                "total_s": sum(arr),
                "mean_s": mean(arr),
                "max_s": max(arr),
                "min_s": min(arr),
            }
        self.timing["rollup"] = roll
        return roll

    def init_timing(self):
        """Reset timing events and rollup."""
        self.timing = {"events": [], "rollup": {}}


def load_prompt(name: str, game_name: Optional[str] = None, **kwargs) -> str:
    """Load a prompt from the abstraction_prompts directory.

    Looks for:
      1. abstraction_prompts/<game_name>/<name>.txt (game-specific)
      2. abstraction_prompts/<name>.txt (default)

    Args:
        name: Prompt file name (without .txt extension)
        game_name: Optional game name for game-specific prompts
        **kwargs: Format arguments for the prompt template

    Returns:
        Formatted prompt string

    Raises:
        FileNotFoundError: If prompt file not found
    """
    base = Path("abstraction_prompts")

    # Try game-specific prompt first
    if game_name:
        game_specific = base / game_name / f"{name}.txt"
        if game_specific.exists():
            text = game_specific.read_text()
            return text.format(**kwargs)

    # Fall back to default prompt
    default_path = base / f"{name}.txt"
    if default_path.exists():
        text = default_path.read_text()
        return text.format(**kwargs)

    raise FileNotFoundError(
        f"Prompt '{name}' not found in abstraction_prompts/ "
        + (f"or abstraction_prompts/{game_name}/" if game_name else "")
    )
