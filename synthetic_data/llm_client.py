# synthetic_data/llm_client.py
"""OpenAI-compatible LLM client wrapper."""

import logging
import json
import re
from typing import Optional, Dict, Any, List

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised only when optional dependency is absent
    OpenAI = None

from .config import LLMConfig


logger = logging.getLogger(__name__)


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
                "generation. Install it or use --use-vllm-batch."
            )
        self.config = config
        self.client = OpenAI(
            base_url=config.api_base_url,
            api_key=config.api_key if config.api_key else "dummy-key",  # Some servers require non-empty
            timeout=config.timeout,
        )
        logger.info(f"LLM client initialized with base_url={config.api_base_url}, model={config.model_name}")
    
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
        
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
        )
        
        return response.choices[0].message.content
    
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
            try:
                try:
                    # Try with JSON response format (OpenAI compatible)
                    response = self.client.chat.completions.create(
                        model=self.config.model_name,
                        messages=messages,
                        temperature=temperature if temperature is not None else self.config.temperature,
                        max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
                        response_format={"type": "json_object"},
                    )
                    text = response.choices[0].message.content
                except Exception as e:
                    # Fall back to regular generation if JSON mode not supported
                    logger.debug(f"JSON mode not supported, falling back to regular generation: {e}")
                    response = self.client.chat.completions.create(
                        model=self.config.model_name,
                        messages=messages,
                        temperature=temperature if temperature is not None else self.config.temperature,
                        max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
                    )
                    text = response.choices[0].message.content

                # Parse JSON from response
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
