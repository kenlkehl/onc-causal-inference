# synthetic_data/vllm_batch_client.py
"""Direct vLLM batch inference client for faster generation."""

import logging
from typing import List, Optional
from dataclasses import dataclass

from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)


@dataclass
class VLLMConfig:
    """Configuration for direct vLLM inference."""
    model_name: str = "openai/gpt-oss-120b"
    tensor_parallel_size: int = 2
    gpu_memory_utilization: float = 0.90
    max_model_len: Optional[int] = None  # Use model default
    temperature: float = 0.8
    max_tokens: int = 10000
    download_dir: str = "./"
    reasoning_marker: Optional[str] = "assistantfinal"  # Text after this marker is the real output
    device_ids: Optional[List[int]] = None  # GPU IDs to use (sets CUDA_VISIBLE_DEVICES before loading model)


class VLLMBatchClient:
    """Client for direct vLLM batch inference (much faster than HTTP API)."""

    def __init__(self, config: VLLMConfig):
        """
        Initialize the vLLM batch client.

        Args:
            config: vLLM configuration
        """
        import os

        self.config = config

        # Set CUDA_VISIBLE_DEVICES before loading model if device_ids specified
        if config.device_ids is not None:
            device_str = ",".join(str(d) for d in config.device_ids)
            os.environ["CUDA_VISIBLE_DEVICES"] = device_str
            logger.info(f"Set CUDA_VISIBLE_DEVICES={device_str}")

        logger.info(f"Loading vLLM model: {config.model_name} with TP={config.tensor_parallel_size}")
        
        self.llm = LLM(
            model=config.model_name,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            download_dir = config.download_dir,
            trust_remote_code=True,
        )
        logger.info("vLLM model loaded successfully")
    
    @staticmethod
    def strip_reasoning_prefix(text: str, reasoning_marker: Optional[str] = None) -> str:
        """
        Strip model reasoning prefix from text.
        
        Some models output chain-of-thought reasoning before the actual content.
        This method removes everything before the reasoning_marker.
        
        Args:
            text: Raw text output from model
            reasoning_marker: Marker string (case-insensitive). Text after this is kept.
                             If None, returns text unchanged.
                             
        Returns:
            Text with reasoning prefix removed
        """
        if not reasoning_marker or not text:
            return text
        
        # Case-insensitive search for marker
        marker_lower = reasoning_marker.lower()
        text_lower = text.lower()
        
        idx = text_lower.find(marker_lower)
        if idx != -1:
            # Return text after the marker
            return text[idx + len(reasoning_marker):].strip()
        
        # Marker not found, return original
        return text
    
    def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        """
        Generate responses for a batch of prompts.
        
        Args:
            prompts: List of user prompts
            system_prompt: Optional system prompt (prepended to each prompt)
            temperature: Override config temperature
            max_tokens: Override config max_tokens
            
        Returns:
            List of generated text responses (same order as prompts)
        """
        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens
        
        # Build full prompts with system message if provided
        # Use chat template format for the model
        if system_prompt:
            full_prompts = [
                f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
                for prompt in prompts
            ]
        else:
            full_prompts = [f"User: {prompt}\n\nAssistant:" for prompt in prompts]
        
        sampling_params = SamplingParams(
            temperature=temp,
            max_tokens=max_tok,
            stop=["User:", "\n\nUser:"],  # Stop at next user turn
        )
        
        logger.info(f"Generating {len(full_prompts)} prompts with vLLM batch inference...")
        outputs = self.llm.generate(full_prompts, sampling_params)
        
        # Extract text from outputs (maintain order)
        results = []
        for output in outputs:
            if output.outputs and len(output.outputs) > 0:
                text = output.outputs[0].text.strip()
                results.append(text)
            else:
                logger.warning(f"Empty output for prompt")
                results.append("")
        
        logger.info(f"Batch generation complete: {len(results)} responses")
        return results
    
    def generate_single(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a single response (convenience wrapper)."""
        results = self.generate_batch(
            prompts=[prompt],
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return results[0] if results else ""
    
    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """Generate a JSON response and parse it."""
        import json
        import re
        
        # Add JSON instruction to prompt if not present
        if "json" not in prompt.lower():
            prompt = prompt + "\n\nRespond with valid JSON only."
        
        text = self.generate_single(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        return self._parse_json(text)
    
    @staticmethod
    def _parse_json(text: str) -> dict:
        """Parse JSON from LLM response, handling code blocks, reasoning, and extra text."""
        import json
        import re
        
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
        
        # Look for JSON object with confounders or other expected keys
        # Use a more targeted pattern to find complete JSON objects
        json_patterns = [
            r'\{\s*"confounders"\s*:\s*\[.*?\]\s*\}',  # Confounder response
            r'\{\s*"treatment_equation"\s*:\s*\{.*?\}\s*,\s*"outcome_equation"\s*:\s*\{.*?\}\s*\}',  # Equations
            r'\{\s*"summary_statistics"\s*:\s*\{.*?\}\s*\}',  # Summary stats
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
        
        # Find the last complete JSON object in text (often at the end after reasoning)
        # Look for balanced braces
        brace_count = 0
        start_idx = -1
        last_complete_json = None
        
        for i, char in enumerate(text):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    candidate = text[start_idx:i+1]
                    try:
                        parsed = json.loads(candidate)
                        last_complete_json = parsed
                    except json.JSONDecodeError:
                        pass
                    start_idx = -1
        
        if last_complete_json:
            return last_complete_json
        
        # Try to find any JSON object in text (fallback)
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        raise ValueError(f"Could not parse JSON from LLM response: {text[:500]}...")
