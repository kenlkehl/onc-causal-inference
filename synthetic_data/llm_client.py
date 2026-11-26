# synthetic_data/llm_client.py
"""OpenAI-compatible LLM client wrapper."""

import logging
import json
import re
import threading
from typing import Optional, Dict, Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised only when optional dependency is absent
    OpenAI = None

from .config import LLMConfig


logger = logging.getLogger(__name__)

_NEMOTRON_3_5_RE = re.compile(r"nemotron[-_ ]?3[._]5(?![0-9])", re.IGNORECASE)


def is_nemotron_3_5(model_name: Optional[str]) -> bool:
    """True for NVIDIA Nemotron 3.5 Super models (which require temp=1.0, top_p=0.95)."""
    return bool(model_name and _NEMOTRON_3_5_RE.search(str(model_name)))


def nemotron_3_5_extra_body(*, enable_thinking: bool = False) -> Dict[str, Any]:
    """NVIDIA recommended sampling profile for Nemotron 3.5."""
    return {
        "top_p": 0.95,
        "top_k": 0,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }


class LLMClient:
    """Client for OpenAI-compatible LLM APIs (works with vLLM, local models, etc.)."""
    
    def __init__(self, config: LLMConfig):
        """
        Initialize the LLM client.
        
        Args:
            config: LLM configuration with API URL, key, model name, etc.
        """
        if OpenAI is None:
            raise ImportError(
                "The 'openai' package is required for API-server synthetic data "
                "generation. Install it with: pip install openai"
            )
        self.config = config
        self.client = OpenAI(
            base_url=config.api_base_url,
            api_key=config.api_key if config.api_key else "dummy-key",  # Some servers require non-empty
            timeout=config.timeout,
        )
        self._clients = [self.client]
        if config.api_base_urls:
            self.client.close()
            self._clients = [
                OpenAI(base_url=url, api_key=config.api_key or "dummy-key", timeout=config.timeout)
                for url in config.api_base_urls
            ]
            self.client = self._clients[0]
        self._dispatch_lock = threading.Lock()
        self._next_replica = 0
        logger.info(f"LLM client initialized with base_url={config.api_base_url}, model={config.model_name}")

    def _replica(self):
        with self._dispatch_lock:
            client = self._clients[self._next_replica % len(self._clients)]
            self._next_replica += 1
            return client

    def close(self):
        for client in self._clients:
            client.close()

    def _content(self, response):
        choice = response.choices[0]
        if self.config.reject_incomplete:
            if choice.finish_reason != "stop" or not (choice.message.content or "").strip():
                raise ValueError(f"Incomplete generation: finish_reason={choice.finish_reason!r}")
        return choice.message.content
    
    def _request_sampling_kwargs(
        self, temperature: Optional[float], max_tokens: Optional[int]
    ) -> Dict[str, Any]:
        """Build sampling arguments, applying NVIDIA's recommended profile for Nemotron 3.5."""
        if is_nemotron_3_5(self.config.model_name):
            eff_temp = 1.0
            extra = {**nemotron_3_5_extra_body(enable_thinking=False), **(self.config.extra_body or {})}
        else:
            eff_temp = temperature if temperature is not None else self.config.temperature
            extra = dict(self.config.extra_body) if self.config.extra_body else {}
        kwargs: Dict[str, Any] = {
            "temperature": eff_temp,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
        }
        if extra:
            kwargs["extra_body"] = extra
        return kwargs

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Override config temperature
            max_tokens: Override config max_tokens
            
        Returns:
            Generated text response
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._replica().chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            **self._request_sampling_kwargs(temperature, max_tokens),
        )
        return self._content(response)
    
    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """
        Generate a JSON response from the LLM, retrying on failure.

        Attempts to use JSON mode if supported, falls back to parsing.
        Retries up to max_retries times on parse failures.

        Args:
            prompt: User prompt (should request JSON output)
            system_prompt: Optional system prompt
            temperature: Override config temperature
            max_tokens: Override config max_tokens
            max_retries: Number of attempts before raising (default: 3)

        Returns:
            Parsed JSON as dictionary
        """
        # Add JSON instruction to prompt if not present
        if "json" not in prompt.lower():
            prompt = prompt + "\n\nRespond with valid JSON only."

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_error = None
        for attempt in range(1, max_retries + 1):
            client = self._replica()
            try:
                sampling = self._request_sampling_kwargs(temperature, max_tokens)
                try:
                    # Try with JSON response format (OpenAI compatible)
                    response = client.chat.completions.create(
                        model=self.config.model_name,
                        messages=messages,
                        response_format={"type": "json_object"},
                        **sampling,
                    )
                except Exception as e:
                    # Fall back to regular generation if JSON mode not supported
                    logger.debug(f"JSON mode not supported, falling back to regular generation: {e}")
                    response = client.chat.completions.create(
                        model=self.config.model_name,
                        messages=messages,
                        **sampling,
                    )

                # Parse JSON from response
                text = self._content(response)
                return self._parse_json(text)
            except (ValueError, json.JSONDecodeError) as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(
                        f"JSON parse failed (attempt {attempt}/{max_retries}): {e}. Retrying..."
                    )
                else:
                    logger.error(
                        f"JSON parse failed after {max_retries} attempts: {e}"
                    )
        raise last_error
    
    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response, handling code blocks and extra text.
        
        Args:
            text: Raw LLM response text
            
        Returns:
            Parsed JSON dictionary
        """
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from code block
        code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if code_block_match:
            try:
                return json.loads(code_block_match.group(1).strip())
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON object in text
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        raise ValueError(f"Could not parse JSON from LLM response: {text[:500]}...")
