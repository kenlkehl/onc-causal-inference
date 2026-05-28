# oci/extraction/explicit_features.py
"""LLM-based explicit feature extraction from clinical text.

This module extracts researcher-specified feature variables from clinical text
using a large language model (via vLLM). The extracted features are returned
as structured data that can be featurized and used alongside text embeddings
for causal inference.

Three vLLM modes are supported:
- "server": Connect to a running vLLM OpenAI-compatible server
- "start_server": Start vLLM server subprocess, then connect (cleans up after)
- "python_api": Use vLLM Python API directly (no server, in-process inference)

Example usage:
    from oci.extraction.explicit_features import VLLMFeatureExtractor
    from oci.config import ExplicitFeatureSpec

    specs = [
        ExplicitFeatureSpec(
            name="performance_status",
            type="categorical",
            categories=["0", "1", "2", "3", "4"],
            description="ECOG performance status",
            roles=["confounder", "effect_modifier"],
        ),
        ExplicitFeatureSpec(
            name="age_at_diagnosis",
            type="continuous",
            description="Patient age at diagnosis in years",
            roles=["confounder"],
        )
    ]

    extractor = VLLMFeatureExtractor(
        specs=specs,
        mode="python_api",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        tensor_parallel_size=2
    )

    results = extractor.extract(clinical_texts)
    # results: List[Dict[str, ExplicitFeatureValue]]
"""

import json
import logging
import re
import subprocess
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from tqdm import tqdm

from ..config import ExplicitFeatureSpec

logger = logging.getLogger(__name__)


@dataclass
class ExplicitFeatureValue:
    """Extracted value for a single feature."""
    name: str
    type: str  # "categorical" or "continuous"
    value: Optional[Union[str, float]]  # Extracted value (None if missing)
    is_missing: bool  # True if extraction failed after retries


def build_extraction_prompt(
    clinical_text: str,
    specs: List[ExplicitFeatureSpec],
    max_text_length: int = 8000
) -> str:
    """Build prompt for feature extraction.

    Args:
        clinical_text: Clinical text to extract from
        specs: List of feature specifications
        max_text_length: Maximum characters of text to include

    Returns:
        Formatted prompt string for the LLM
    """
    instructions = []
    json_fields = []

    for i, spec in enumerate(specs, 1):
        name = spec.name
        conf_type = spec.type
        description = spec.description or name.replace("_", " ").title()

        if conf_type == "categorical":
            categories = spec.categories or []
            cat_list = ", ".join(f'"{c}"' for c in categories)
            instructions.append(
                f'{i}. {name} (categorical): {description}\n'
                f'   Valid values: {cat_list}'
            )
            json_fields.append(f'"{name}": "<category>"')
        else:  # continuous
            instructions.append(
                f'{i}. {name} (continuous): {description}\n'
                f'   Respond with a numeric value.'
            )
            json_fields.append(f'"{name}": <number>')

    instructions_text = "\n".join(instructions)
    json_example = "{" + ", ".join(json_fields) + "}"

    # Truncate text if needed
    text = clinical_text[:max_text_length]

    prompt = f"""Read this clinical note and extract the following patient characteristics.
Use only information available before or at treatment initiation. If the value is not explicitly stated or cannot be inferred from pre-treatment information, return null for that field.

{instructions_text}

Clinical Note:
{text}

Respond with JSON only, no other text:
{json_example}"""

    return prompt


def parse_extraction_response(
    response: str,
    specs: List[ExplicitFeatureSpec]
) -> Dict[str, ExplicitFeatureValue]:
    """Parse LLM JSON response to extract feature values.

    Args:
        response: Raw LLM response text (expected to be JSON)
        specs: List of feature specifications

    Returns:
        Dictionary mapping feature names to ExplicitFeatureValue objects.
        Categorical values are validated; invalid ones are marked as missing.
        Continuous values that fail parsing are marked as missing.
    """
    response = response.strip()

    # Try to extract JSON from response (handle markdown code blocks)
    json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
    else:
        json_str = response

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        logger.debug(f"Could not parse JSON response: {response[:200]}")
        # Return missing for all features
        result = {}
        for spec in specs:
            result[spec.name] = ExplicitFeatureValue(
                name=spec.name,
                type=spec.type,
                value=None,
                is_missing=True
            )
        return result

    # Validate and extract each feature
    result = {}
    for spec in specs:
        name = spec.name
        conf_type = spec.type
        value = parsed.get(name)

        if conf_type == "categorical":
            categories = spec.categories or []
            if value is None:
                result[name] = ExplicitFeatureValue(
                    name=name, type=conf_type, value=None, is_missing=True
                )
            elif str(value) in categories:
                result[name] = ExplicitFeatureValue(
                    name=name, type=conf_type, value=str(value), is_missing=False
                )
            else:
                # Try case-insensitive match
                value_lower = str(value).lower()
                matched_cat = None
                for cat in categories:
                    if cat.lower() == value_lower:
                        matched_cat = cat
                        break
                if matched_cat:
                    result[name] = ExplicitFeatureValue(
                        name=name, type=conf_type, value=matched_cat, is_missing=False
                    )
                else:
                    logger.debug(f"Invalid category '{value}' for {name}, valid: {categories}")
                    result[name] = ExplicitFeatureValue(
                        name=name, type=conf_type, value=None, is_missing=True
                    )
        else:  # continuous
            if value is None:
                result[name] = ExplicitFeatureValue(
                    name=name, type=conf_type, value=None, is_missing=True
                )
            else:
                try:
                    float_value = float(value)
                    result[name] = ExplicitFeatureValue(
                        name=name, type=conf_type, value=float_value, is_missing=False
                    )
                except (ValueError, TypeError):
                    logger.debug(f"Could not parse continuous value '{value}' for {name}")
                    result[name] = ExplicitFeatureValue(
                        name=name, type=conf_type, value=None, is_missing=True
                    )

    return result


class VLLMFeatureExtractor:
    """Extractor for explicit features using vLLM.

    Supports three modes:
    - "server": Connect to running vLLM OpenAI-compatible server
    - "start_server": Start vLLM server subprocess, then connect
    - "python_api": Use vLLM Python API directly (in-process)
    """

    def __init__(
        self,
        specs: List[ExplicitFeatureSpec],
        mode: str = "server",
        server_url: str = "http://localhost:8000/v1",
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        download_dir: Optional[str] = None,
        max_model_len: Optional[int] = None,
        api_key: str = "EMPTY",
        max_retries: int = 3,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_text_length: int = 8000
    ):
        """Initialize extractor.

        Args:
            specs: List of feature specifications
            mode: "server", "start_server", or "python_api"
            server_url: URL for vLLM server (used in server modes)
            model_name: Model name/path for vLLM
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory fraction to use
            download_dir: Model download directory
            max_model_len: Maximum model context length (for start_server/python_api)
            api_key: API key (use "EMPTY" for local vLLM)
            max_retries: Maximum retries per patient before marking as missing
            temperature: LLM temperature (0 for deterministic)
            max_tokens: Maximum tokens in response
            max_text_length: Maximum clinical text characters included in prompt
        """
        if mode not in ("server", "start_server", "python_api"):
            raise ValueError(f"mode must be 'server', 'start_server', or 'python_api', got '{mode}'")

        self.specs = specs
        self.mode = mode
        self.server_url = server_url
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.download_dir = download_dir
        self.max_model_len = max_model_len
        self.api_key = api_key
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_text_length = max_text_length

        # These are set lazily
        self._client = None
        self._llm = None
        self._server_process = None

        logger.info(f"VLLMFeatureExtractor initialized: mode={mode}, model={model_name}")
        logger.info(f"Extracting {len(specs)} features: {[s.name for s in specs]}")

    def _init_server_client(self):
        """Initialize OpenAI client for server mode."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

        self._client = OpenAI(
            base_url=self.server_url,
            api_key=self.api_key,
            timeout=30.0,    # 30s per request (default is 10 min)
            max_retries=0,   # No internal retries (we have our own outer retry loop)
        )
        logger.info(f"Connected to vLLM server at: {self.server_url}")

    def _start_server(self):
        """Start vLLM server subprocess."""
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_name,
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--trust-remote-code"
        ]
        if self.download_dir:
            cmd.extend(["--download-dir", self.download_dir])
        if self.max_model_len:
            cmd.extend(["--max-model-len", str(self.max_model_len)])

        logger.info(f"Starting vLLM server: {' '.join(cmd)}")
        self._server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Wait for server to be ready
        logger.info("Waiting for vLLM server to start...")
        time.sleep(30)  # Initial wait

        import requests
        for i in range(60):  # Wait up to 5 minutes
            try:
                resp = requests.get(f"{self.server_url.rstrip('/v1')}/health")
                if resp.status_code == 200:
                    logger.info("vLLM server is ready")
                    break
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(5)
        else:
            raise RuntimeError("vLLM server failed to start within 5 minutes")

        self._init_server_client()

    def _init_python_api(self):
        """Initialize vLLM Python API."""
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("vllm package required. Install with: pip install vllm")

        logger.info(f"Loading vLLM model: {self.model_name} with TP={self.tensor_parallel_size}")

        kwargs = {
            "model": self.model_name,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "trust_remote_code": True,
        }
        if self.download_dir:
            kwargs["download_dir"] = self.download_dir
        if self.max_model_len:
            kwargs["max_model_len"] = self.max_model_len

        self._llm = LLM(**kwargs)
        logger.info("vLLM model loaded successfully")

    def _ensure_initialized(self):
        """Ensure backend is initialized."""
        if self.mode == "server":
            if self._client is None:
                self._init_server_client()
        elif self.mode == "start_server":
            if self._server_process is None:
                self._start_server()
        elif self.mode == "python_api":
            if self._llm is None:
                self._init_python_api()

    def _extract_single_server(self, text: str) -> Dict[str, ExplicitFeatureValue]:
        """Extract features from single text using server API."""
        prompt = build_extraction_prompt(text, self.specs, max_text_length=self.max_text_length)
        best_result = None

        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=30.0,  # Hard per-request cap
                )
                content = response.choices[0].message.content
                if content:
                    result = parse_extraction_response(content, self.specs)
                    # Track best partial result (fewest missing values)
                    if best_result is None or sum(
                        1 for v in result.values() if not v.is_missing
                    ) > sum(1 for v in best_result.values() if not v.is_missing):
                        best_result = result
                    # Return immediately if all values extracted
                    if all(not v.is_missing for v in result.values()):
                        return result
            except Exception as e:
                logger.debug(f"Extraction attempt {attempt + 1} failed: {e}")

        # Return best partial result, or all-missing if no successful parse
        if best_result is not None:
            return best_result
        return {
            spec.name: ExplicitFeatureValue(
                name=spec.name, type=spec.type, value=None, is_missing=True
            )
            for spec in self.specs
        }

    def _extract_batch_python_api(
        self,
        texts: List[str]
    ) -> List[Dict[str, ExplicitFeatureValue]]:
        """Extract features from batch using vLLM Python API."""
        from vllm import SamplingParams

        # Build prompts
        prompts = []
        for text in texts:
            user_content = build_extraction_prompt(
                text,
                self.specs,
                max_text_length=self.max_text_length,
            )
            tokenizer = self._llm.get_tokenizer()

            if hasattr(tokenizer, 'apply_chat_template'):
                try:
                    prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": user_content}],
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except Exception:
                    prompt = f"User: {user_content}\n\nAssistant:"
            else:
                prompt = f"User: {user_content}\n\nAssistant:"
            prompts.append(prompt)

        # Sample params
        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        # Generate
        logger.info(f"Running vLLM batch inference on {len(prompts)} texts...")
        outputs = self._llm.generate(prompts, sampling_params)

        # Parse results
        results = []
        for output in outputs:
            if output.outputs and len(output.outputs) > 0:
                content = output.outputs[0].text.strip()
                result = parse_extraction_response(content, self.specs)
            else:
                result = {
                    spec.name: ExplicitFeatureValue(
                        name=spec.name, type=spec.type, value=None, is_missing=True
                    )
                    for spec in self.specs
                }
            results.append(result)

        return results

    def extract(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[Dict[str, ExplicitFeatureValue]]:
        """Extract features from a list of clinical texts.

        Args:
            texts: List of clinical text strings
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar

        Returns:
            List of dictionaries mapping feature names to ExplicitFeatureValue
        """
        self._ensure_initialized()

        if self.mode == "python_api":
            # Process all at once (vLLM handles batching internally)
            return self._extract_batch_python_api(texts)
        else:
            # Server mode: process with progress bar
            results = []
            iterator = tqdm(texts, desc="Extracting features") if show_progress else texts
            for text in iterator:
                result = self._extract_single_server(text)
                results.append(result)
            return results

    def extract_to_dataframe(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """Extract features and return as DataFrame.

        Args:
            texts: List of clinical text strings
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar

        Returns:
            DataFrame with columns: explicit_feat_{name}, explicit_feat_{name}_missing
        """
        results = self.extract(texts, batch_size, show_progress)

        # Convert to DataFrame format
        data = {}
        for spec in self.specs:
            values = []
            missing_flags = []
            for result in results:
                val = result.get(spec.name)
                if val:
                    values.append(val.value)
                    missing_flags.append(val.is_missing)
                else:
                    values.append(None)
                    missing_flags.append(True)

            data[f"explicit_feat_{spec.name}"] = values
            data[f"explicit_feat_{spec.name}_missing"] = missing_flags

        return pd.DataFrame(data)

    def cleanup(self):
        """Clean up resources."""
        if self._server_process is not None:
            logger.info("Stopping vLLM server...")
            self._server_process.terminate()
            self._server_process.wait()
            self._server_process = None

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()


def extract_explicit_features(
    texts: List[str],
    specs: List[ExplicitFeatureSpec],
    mode: str = "server",
    server_url: str = "http://localhost:8000/v1",
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    download_dir: Optional[str] = None,
    max_model_len: Optional[int] = None,
    max_retries: int = 3,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    max_text_length: int = 8000,
    batch_size: int = 32
) -> pd.DataFrame:
    """Convenience function to extract features from texts.

    Args:
        texts: List of clinical text strings
        specs: List of feature specifications
        mode: vLLM mode ("server", "start_server", or "python_api")
        server_url: URL for vLLM server
        model_name: Model name/path
        tensor_parallel_size: Number of GPUs
        gpu_memory_utilization: GPU memory fraction
        download_dir: Model download directory
        max_model_len: Maximum model context length (for start_server/python_api)
        max_retries: Retries per patient before marking as missing
        temperature: LLM temperature
        max_tokens: Max response tokens
        max_text_length: Maximum clinical text characters included in prompt
        batch_size: Batch size for processing

    Returns:
        DataFrame with columns: explicit_feat_{name}, explicit_feat_{name}_missing
    """
    extractor = VLLMFeatureExtractor(
        specs=specs,
        mode=mode,
        server_url=server_url,
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        download_dir=download_dir,
        max_model_len=max_model_len,
        max_retries=max_retries,
        temperature=temperature,
        max_tokens=max_tokens,
        max_text_length=max_text_length
    )

    try:
        return extractor.extract_to_dataframe(texts, batch_size=batch_size)
    finally:
        extractor.cleanup()
