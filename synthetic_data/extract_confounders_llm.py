#!/usr/bin/env python
"""LLM-based confounder extraction from clinical text.

This script extracts confounder values from clinical text using an LLM (vLLM or OpenAI API).
It dynamically extracts all confounders defined in the dataset's metadata.json file.

Usage:
    python synthetic_data/extract_confounders_llm.py \
        --input example_synthetic_data/dataset.parquet \
        --output example_synthetic_data/dataset_with_extraction.parquet \
        --vllm-url http://localhost:8000/v1 \ (or --vllm-direct)
        --model openai/gpt-oss-120b \
	--download-dir /data1/ken/models

    # With explicit metadata path:
    python synthetic_data/extract_confounders_llm.py \
        --input example_synthetic_data/dataset.parquet \
        --output example_synthetic_data/dataset_with_extraction.parquet \
        --metadata example_synthetic_data/metadata.json \
        --vllm-url http://localhost:8000/v1 \
        --model openai/gpt-oss-120b
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global reasoning marker (can be set via command line)
REASONING_MARKER = "assistantfinal"


def truncate_to_last_n_tokens(text: str, tokenizer, max_tokens: int) -> str:
    """Truncate text to the last N tokens using the model's tokenizer.

    Args:
        text: Full clinical text
        tokenizer: HuggingFace tokenizer instance
        max_tokens: Maximum number of tokens to keep (from the end)

    Returns:
        Decoded text from the last max_tokens tokens
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text
    truncated_ids = token_ids[-max_tokens:]
    return tokenizer.decode(truncated_ids, skip_special_tokens=True)


def load_metadata(metadata_path: Path) -> Dict[str, Any]:
    """Load metadata.json containing confounder definitions.

    Args:
        metadata_path: Path to metadata.json file

    Returns:
        Dictionary containing metadata with 'confounders' key
    """
    with open(metadata_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_metadata_path(input_path: Path) -> Optional[Path]:
    """Auto-discover metadata.json in the same directory as input dataset.

    Args:
        input_path: Path to input dataset file

    Returns:
        Path to metadata.json if found, None otherwise
    """
    parent_dir = input_path.parent
    metadata_path = parent_dir / "metadata.json"
    if metadata_path.exists():
        return metadata_path
    return None


def build_extraction_prompt(clinical_text: str, confounders: List[Dict[str, Any]]) -> str:
    """Build a dynamic extraction prompt based on confounder definitions.

    Args:
        clinical_text: Clinical text to extract from
        confounders: List of confounder definitions from metadata

    Returns:
        Formatted prompt string for the LLM
    """
    instructions = []
    json_fields = []

    for i, conf in enumerate(confounders, 1):
        name = conf["name"]
        conf_type = conf["type"]
        description = conf.get("description", name.replace("_", " ").title())

        if conf_type == "categorical":
            categories = conf.get("categories", [])
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

    prompt = f"""Read this clinical note and extract the following patient characteristics:

{instructions_text}

Clinical Note:
{clinical_text}

Respond with JSON only, no other text:
{json_example}"""

    return prompt


def parse_harmony_response(text: str) -> str:
    """Parse harmony format response to extract the final channel content.

    gpt-oss models use the harmony format with channels like:
    - <|channel|>analysis - reasoning/chain-of-thought
    - <|channel|>commentary - tool preambles
    - <|channel|>final - user-facing answer

    Uses regex parsing only (no openai_harmony/tiktoken dependency).

    Args:
        text: Raw harmony format model output

    Returns:
        Content from the final channel, or the original text if not in harmony format
    """
    if not text:
        return text

    # Look for final channel content
    final_match = re.search(r'<\|channel\|>final[^<]*<\|message\|>(.+?)(?:<\||$)', text, re.DOTALL)
    if final_match:
        return final_match.group(1).strip()

    # Look for any message content
    message_match = re.search(r'<\|message\|>(.+?)(?:<\||$)', text, re.DOTALL)
    if message_match:
        return message_match.group(1).strip()

    return text


def strip_reasoning_prefix(text: str, marker: str = "assistantfinal") -> str:
    """Strip reasoning prefix from model output.

    Some reasoning models output chain-of-thought before the final answer,
    separated by a marker like 'assistantfinal'.

    Args:
        text: Raw model output
        marker: Marker that precedes the final answer (case-insensitive)

    Returns:
        Text after the marker, or original text if marker not found
    """
    if not text or not marker:
        return text

    # First try harmony format parsing for gpt-oss models
    if '<|channel|>' in text or '<|message|>' in text:
        return parse_harmony_response(text)

    marker_lower = marker.lower()
    text_lower = text.lower()

    idx = text_lower.find(marker_lower)
    if idx != -1:
        # Return text after the marker
        return text[idx + len(marker):].strip()

    return text


def parse_extraction_response(
    response: str,
    confounders: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Parse the LLM JSON response to extract all confounder values.

    Args:
        response: Raw LLM response text (expected to be JSON)
        confounders: List of confounder definitions from metadata

    Returns:
        Dictionary mapping confounder names to extracted values.
        Categorical values are validated; invalid ones become "unknown".
        Continuous values that fail parsing become None.
    """
    # Strip reasoning prefix if present
    response = strip_reasoning_prefix(response, REASONING_MARKER)
    response = response.strip()

    # Try to extract JSON from the response
    # Handle cases where LLM might include markdown code blocks
    json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
    else:
        json_str = response

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        logger.warning(f"Could not parse JSON response: {response[:200]}")
        # Return unknown/None for all confounders
        result = {}
        for conf in confounders:
            name = conf["name"]
            if conf["type"] == "categorical":
                result[name] = "unknown"
            else:
                result[name] = None
        return result

    # Validate and extract each confounder
    result = {}
    for conf in confounders:
        name = conf["name"]
        conf_type = conf["type"]
        value = parsed.get(name)

        if conf_type == "categorical":
            categories = conf.get("categories", [])
            if value is None:
                result[name] = "unknown"
            elif str(value) in categories:
                result[name] = str(value)
            else:
                # Try case-insensitive match
                value_lower = str(value).lower()
                matched = False
                for cat in categories:
                    if cat.lower() == value_lower:
                        result[name] = cat
                        matched = True
                        break
                if not matched:
                    logger.debug(f"Invalid category '{value}' for {name}, valid: {categories}")
                    result[name] = "unknown"
        else:  # continuous
            if value is None:
                result[name] = None
            else:
                try:
                    result[name] = float(value)
                except (ValueError, TypeError):
                    logger.debug(f"Could not parse continuous value '{value}' for {name}")
                    result[name] = None

    return result


def parse_ground_truth_from_patient_prompt(
    patient_prompt: str,
    confounders: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Parse ground truth values from the patient_prompt column.

    The patient_prompt format is:
    - Description: value (instructions)
    - Description: "value" (instructions)

    Args:
        patient_prompt: The patient_prompt string from the dataset
        confounders: List of confounder definitions

    Returns:
        Dictionary mapping confounder names to ground truth values
    """
    result = {}

    for conf in confounders:
        name = conf["name"]
        conf_type = conf["type"]
        description = conf.get("description", name.replace("_", " ").title())

        # Build pattern to match this confounder's line
        # Format: "- Description: value" or "- Description: "value""
        # The description might have slight variations, so use flexible matching
        pattern = rf'-\s*{re.escape(description)}:\s*'

        for line in patient_prompt.split('\n'):
            if re.search(pattern, line, re.IGNORECASE):
                # Extract value after the colon
                match = re.search(pattern + r'(.+?)(?:\s*\(|$)', line, re.IGNORECASE)
                if match:
                    value_str = match.group(1).strip()

                    if conf_type == "continuous":
                        # Extract number from strings like "65 years old" or "3.5"
                        num_match = re.search(r'([\d.]+)', value_str)
                        if num_match:
                            try:
                                result[name] = float(num_match.group(1))
                            except ValueError:
                                result[name] = None
                        else:
                            result[name] = None
                    else:  # categorical
                        # Extract quoted value or plain value
                        quoted_match = re.search(r'"([^"]+)"', value_str)
                        if quoted_match:
                            result[name] = quoted_match.group(1)
                        else:
                            result[name] = value_str.strip()
                    break

        # If not found, set default
        if name not in result:
            if conf_type == "categorical":
                result[name] = "unknown"
            else:
                result[name] = None

    return result


class OpenAIClient:
    """Client for OpenAI-compatible API (including vLLM)."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "openai/gpt-oss-120b",
        api_key: str = "EMPTY",
        confounders: Optional[List[Dict[str, Any]]] = None,
        max_text_tokens: int = 100000
    ):
        """Initialize the client.

        Args:
            base_url: Base URL for the API (vLLM or OpenAI)
            model: Model name to use
            api_key: API key (use "EMPTY" for local vLLM)
            confounders: List of confounder definitions from metadata
            max_text_tokens: Max tokens to keep from the end of each clinical text
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

        from transformers import AutoTokenizer
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.confounders = confounders or []
        self.max_text_tokens = max_text_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        logger.info(f"Initialized OpenAI client: {base_url} / {model}")
        logger.info(f"Max text tokens (from end): {max_text_tokens}")
        if self.confounders:
            logger.info(f"Extracting {len(self.confounders)} confounders: {[c['name'] for c in self.confounders]}")

    def extract_batch(
        self,
        clinical_texts: List[str],
        batch_size: int = 10,
        temperature: float = 0.0,
        max_tokens: int = 5000
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Extract confounder values from a batch of clinical texts.

        Args:
            clinical_texts: List of clinical text strings
            batch_size: Number of texts to process in parallel (for rate limiting)
            temperature: LLM temperature (0 for deterministic)
            max_tokens: Maximum tokens in response

        Returns:
            Tuple of (extracted_values_dicts, raw_responses) lists
        """
        results = []
        raw_responses = []

        for i in tqdm(range(0, len(clinical_texts), batch_size), desc="Extracting"):
            batch = clinical_texts[i:i + batch_size]
            batch_results = []
            batch_raw = []

            for text in batch:
                truncated = truncate_to_last_n_tokens(text, self.tokenizer, self.max_text_tokens)
                prompt = build_extraction_prompt(truncated, self.confounders)

                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    # Handle reasoning models that may return content in different fields
                    msg = response.choices[0].message
                    content = msg.content

                    # For reasoning models, try multiple sources for content
                    if content is None:
                        # Try reasoning_content field
                        if hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                            content = msg.reasoning_content
                        # Try model_extra for additional fields (OpenAI SDK v1.x)
                        elif hasattr(msg, 'model_extra') and msg.model_extra:
                            extra = msg.model_extra
                            if 'reasoning_content' in extra:
                                content = extra['reasoning_content']
                            elif 'reasoning' in extra:
                                content = extra['reasoning']
                        # Check the raw response dict
                        elif hasattr(msg, '__dict__'):
                            for key in ['reasoning_content', 'reasoning', 'text']:
                                if key in msg.__dict__ and msg.__dict__[key]:
                                    content = msg.__dict__[key]
                                    break

                    if content is None:
                        logger.warning(f"LLM returned None content. Message fields: {dir(msg)}")
                        # Try to access as dict
                        try:
                            msg_dict = msg.model_dump() if hasattr(msg, 'model_dump') else dict(msg)
                            logger.warning(f"Message dict: {msg_dict}")
                        except:
                            pass
                        # Return unknown/None for all confounders
                        empty_result = {}
                        for conf in self.confounders:
                            if conf["type"] == "categorical":
                                empty_result[conf["name"]] = "unknown"
                            else:
                                empty_result[conf["name"]] = None
                        batch_results.append(empty_result)
                        batch_raw.append("<None>")
                        continue

                    raw_response = content.strip()
                    batch_raw.append(raw_response)
                    extracted = parse_extraction_response(raw_response, self.confounders)
                    batch_results.append(extracted)
                except Exception as e:
                    logger.error(f"Error extracting from text: {e}")
                    # Return unknown/None for all confounders
                    empty_result = {}
                    for conf in self.confounders:
                        if conf["type"] == "categorical":
                            empty_result[conf["name"]] = "unknown"
                        else:
                            empty_result[conf["name"]] = None
                    batch_results.append(empty_result)
                    batch_raw.append(f"<Error: {e}>")

            results.extend(batch_results)
            raw_responses.extend(batch_raw)

        return results, raw_responses


class VLLMBatchClientWrapper:
    """Wrapper around VLLMBatchClient for direct vLLM inference."""

    def __init__(
        self,
        model_name: str = "openai/gpt-oss-120b",
	download_dir: str = "/data1/ken/models",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        confounders: Optional[List[Dict[str, Any]]] = None,
        max_text_tokens: int = 100000
    ):
        """Initialize the vLLM batch client.

        Args:
            model_name: Model name/path
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory fraction to use
            confounders: List of confounder definitions from metadata
            max_text_tokens: Max tokens to keep from the end of each clinical text
        """
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("vllm package required. Install with: pip install vllm")

        logger.info(f"Loading vLLM model: {model_name} with TP={tensor_parallel_size}")

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
	    download_dir=download_dir
        )
        self.model_name = model_name
        self.confounders = confounders or []
        self.max_text_tokens = max_text_tokens
        self.tokenizer = self.llm.get_tokenizer()
        logger.info("vLLM model loaded successfully")
        logger.info(f"Max text tokens (from end): {max_text_tokens}")
        if self.confounders:
            logger.info(f"Extracting {len(self.confounders)} confounders: {[c['name'] for c in self.confounders]}")

    def extract_batch(
        self,
        clinical_texts: List[str],
        batch_size: int = 100,  # vLLM handles large batches efficiently
        temperature: float = 0.0,
        max_tokens: int = 5000
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Extract confounder values from a batch of clinical texts.

        Args:
            clinical_texts: List of clinical text strings
            batch_size: Batch size for vLLM inference
            temperature: LLM temperature (0 for deterministic)
            max_tokens: Maximum tokens in response

        Returns:
            Tuple of (extracted_values_dicts, raw_responses) lists
        """
        from vllm import SamplingParams

        messages_list = []

        for text in clinical_texts:
            truncated = truncate_to_last_n_tokens(text, self.tokenizer, self.max_text_tokens)
            user_content = build_extraction_prompt(truncated, self.confounders)
            messages_list.append([{"role": "user", "content": user_content}])

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=["User:"]
        )

        logger.info(f"Running vLLM batch chat inference on {len(messages_list)} texts...")
        outputs = self.llm.chat(messages_list, sampling_params=sampling_params)

        results = []
        raw_responses = []
        for output in outputs:
            if output.outputs and len(output.outputs) > 0:
                raw_response = output.outputs[0].text.strip()
                raw_responses.append(raw_response)
                extracted = parse_extraction_response(raw_response, self.confounders)
                results.append(extracted)
            else:
                # Return unknown/None for all confounders
                empty_result = {}
                for conf in self.confounders:
                    if conf["type"] == "categorical":
                        empty_result[conf["name"]] = "unknown"
                    else:
                        empty_result[conf["name"]] = None
                results.append(empty_result)
                raw_responses.append("<Empty>")

        return results, raw_responses


def evaluate_extraction_accuracy(
    df: pd.DataFrame,
    confounders: List[Dict[str, Any]],
    extracted_prefix: str = "llm_extracted_",
    true_prefix: str = "true_"
) -> Dict[str, Any]:
    """Evaluate extraction accuracy against ground truth for all confounders.

    Args:
        df: DataFrame with extracted and true values
        confounders: List of confounder definitions from metadata
        extracted_prefix: Prefix for extracted value columns
        true_prefix: Prefix for ground truth columns

    Returns:
        Dictionary with per-confounder and aggregate metrics
    """
    metrics = {
        "per_confounder": {},
        "aggregate": {}
    }

    total_categorical_correct = 0
    total_categorical_valid = 0
    all_continuous_correlations = []

    for conf in confounders:
        name = conf["name"]
        conf_type = conf["type"]
        extracted_col = f"{extracted_prefix}{name}"
        true_col = f"{true_prefix}{name}"

        if extracted_col not in df.columns:
            logger.warning(f"Extracted column '{extracted_col}' not found")
            continue
        if true_col not in df.columns:
            logger.warning(f"True column '{true_col}' not found")
            continue

        if conf_type == "categorical":
            # Filter out unknown extractions
            valid_mask = df[extracted_col] != "unknown"
            valid_df = df[valid_mask]

            total = len(df)
            total_valid = len(valid_df)
            unknown_count = (~valid_mask).sum()

            if total_valid > 0:
                correct = (valid_df[extracted_col].astype(str) == valid_df[true_col].astype(str)).sum()
                accuracy = correct / total_valid
                total_categorical_correct += correct
                total_categorical_valid += total_valid
            else:
                accuracy = 0.0

            coverage = total_valid / total if total > 0 else 0.0

            # Per-category accuracy
            categories = conf.get("categories", [])
            per_category = {}
            for cat in categories:
                cat_mask = valid_df[true_col].astype(str) == str(cat)
                if cat_mask.sum() > 0:
                    cat_correct = (valid_df.loc[cat_mask, extracted_col].astype(str) == str(cat)).sum()
                    per_category[cat] = cat_correct / cat_mask.sum()

            metrics["per_confounder"][name] = {
                "type": "categorical",
                "accuracy": accuracy,
                "coverage": coverage,
                "total_samples": total,
                "valid_extractions": total_valid,
                "unknown_extractions": unknown_count,
                "per_category_accuracy": per_category
            }

        else:  # continuous
            # Filter out None values
            valid_mask = df[extracted_col].notna() & df[true_col].notna()
            valid_df = df[valid_mask]

            total = len(df)
            total_valid = len(valid_df)
            missing_count = (~valid_mask).sum()

            if total_valid > 1:
                extracted_vals = valid_df[extracted_col].astype(float)
                true_vals = valid_df[true_col].astype(float)

                # Pearson correlation
                correlation = np.corrcoef(extracted_vals, true_vals)[0, 1]
                if not np.isnan(correlation):
                    all_continuous_correlations.append(correlation)

                # Mean Absolute Error
                mae = np.abs(extracted_vals - true_vals).mean()

                # Root Mean Squared Error
                rmse = np.sqrt(((extracted_vals - true_vals) ** 2).mean())
            else:
                correlation = None
                mae = None
                rmse = None

            coverage = total_valid / total if total > 0 else 0.0

            metrics["per_confounder"][name] = {
                "type": "continuous",
                "correlation": correlation,
                "mae": mae,
                "rmse": rmse,
                "coverage": coverage,
                "total_samples": total,
                "valid_extractions": total_valid,
                "missing_extractions": missing_count
            }

    # Compute aggregate metrics
    if total_categorical_valid > 0:
        metrics["aggregate"]["categorical_accuracy"] = total_categorical_correct / total_categorical_valid
        metrics["aggregate"]["total_categorical_samples"] = total_categorical_valid
    else:
        metrics["aggregate"]["categorical_accuracy"] = None
        metrics["aggregate"]["total_categorical_samples"] = 0

    if all_continuous_correlations:
        metrics["aggregate"]["mean_continuous_correlation"] = np.mean(all_continuous_correlations)
        metrics["aggregate"]["num_continuous_confounders"] = len(all_continuous_correlations)
    else:
        metrics["aggregate"]["mean_continuous_correlation"] = None
        metrics["aggregate"]["num_continuous_confounders"] = 0

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Extract confounder values from clinical text using LLM"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input parquet/csv file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to output parquet file"
    )
    parser.add_argument(
        "--metadata", "-m",
        type=str,
        default=None,
        help="Path to metadata.json (default: auto-discover in same directory as input)"
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="clinical_text",
        help="Name of the text column"
    )
    parser.add_argument(
        "--patient-prompt-column",
        type=str,
        default="patient_prompt",
        help="Column with ground truth patient characteristics (for evaluation)"
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default=None,
        help="vLLM OpenAI-compatible API URL (e.g., http://localhost:8000/v1)"
    )
    parser.add_argument(
        "--vllm-direct",
        action="store_true",
        help="Use direct vLLM inference instead of API"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-oss-120b",
        help="Model name/path"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="Tensor parallel size for direct vLLM"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="EMPTY",
        help="API key (use EMPTY for local vLLM)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--reasoning-marker",
        type=str,
        default="assistantfinal",
        help="Marker text that precedes final answer in reasoning models"
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip accuracy evaluation against ground truth"
    )

    parser.add_argument(
        "--max-text-tokens",
        type=int,
        default=100000,
        help="Max tokens to keep from the end of each clinical text (default: 100000)"
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=5000,
        help="Max tokens for model response (default: 5000)"
    )
    parser.add_argument(
        "--download-dir",
        type=str,
	default="/data1/ken/models",
        help="VLLM model download dir"
    )



    args = parser.parse_args()

    # Set global reasoning marker
    global REASONING_MARKER
    REASONING_MARKER = args.reasoning_marker
    logger.info(f"Using reasoning marker: '{REASONING_MARKER}'")

    # Load dataset
    input_path = Path(args.input)
    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    logger.info(f"Loaded {len(df)} samples from {input_path}")

    # Check text column exists
    if args.text_column not in df.columns:
        raise ValueError(f"Text column '{args.text_column}' not found. Available: {df.columns.tolist()}")

    # Load metadata
    if args.metadata:
        metadata_path = Path(args.metadata)
    else:
        metadata_path = find_metadata_path(input_path)

    if metadata_path is None or not metadata_path.exists():
        raise ValueError(
            f"Could not find metadata.json. Either place it in {input_path.parent} "
            f"or specify with --metadata"
        )

    logger.info(f"Loading metadata from: {metadata_path}")
    metadata = load_metadata(metadata_path)
    confounders = metadata.get("confounders", [])

    if not confounders:
        raise ValueError("No confounders found in metadata.json")

    logger.info(f"Found {len(confounders)} confounders to extract:")
    for conf in confounders:
        conf_type = conf["type"]
        if conf_type == "categorical":
            cats = conf.get("categories", [])
            logger.info(f"  - {conf['name']} (categorical): {cats}")
        else:
            logger.info(f"  - {conf['name']} (continuous)")

    # Initialize client with confounders
    if args.vllm_direct:
        client = VLLMBatchClientWrapper(
            model_name=args.model,
	    download_dir=args.download_dir,
            tensor_parallel_size=args.tensor_parallel_size,
            confounders=confounders,
            max_text_tokens=args.max_text_tokens
        )
    elif args.vllm_url:
        client = OpenAIClient(
            base_url=args.vllm_url,
            model=args.model,
            api_key=args.api_key,
            confounders=confounders,
            max_text_tokens=args.max_text_tokens
        )
    else:
        raise ValueError("Must specify either --vllm-url or --vllm-direct")

    # Extract confounders
    clinical_texts = df[args.text_column].tolist()
    extracted_values_list, raw_responses = client.extract_batch(
        clinical_texts,
        batch_size=args.batch_size,
        max_tokens=args.max_completion_tokens
    )

    # Add extracted values to dataframe (one column per confounder)
    for conf in confounders:
        name = conf["name"]
        col_name = f"llm_extracted_{name}"
        df[col_name] = [ev.get(name) for ev in extracted_values_list]

    # Add raw responses
    df["llm_raw_response"] = raw_responses

    # Report extraction statistics per confounder
    logger.info("\nExtraction Statistics:")
    for conf in confounders:
        name = conf["name"]
        col_name = f"llm_extracted_{name}"
        logger.info(f"\n  {name}:")

        if conf["type"] == "categorical":
            value_counts = df[col_name].value_counts()
            for val, count in value_counts.items():
                logger.info(f"    {val}: {count} ({count/len(df)*100:.1f}%)")
        else:
            valid_mask = df[col_name].notna()
            valid_count = valid_mask.sum()
            if valid_count > 0:
                values = df.loc[valid_mask, col_name].astype(float)
                logger.info(f"    Valid: {valid_count} ({valid_count/len(df)*100:.1f}%)")
                logger.info(f"    Mean: {values.mean():.2f}, Std: {values.std():.2f}")
                logger.info(f"    Min: {values.min():.2f}, Max: {values.max():.2f}")
            else:
                logger.info(f"    No valid extractions")

    # Parse ground truth and evaluate accuracy
    if not args.skip_evaluation and args.patient_prompt_column in df.columns:
        logger.info(f"\nParsing ground truth from '{args.patient_prompt_column}' column...")

        # Parse ground truth values
        ground_truth_list = []
        for patient_prompt in tqdm(df[args.patient_prompt_column], desc="Parsing ground truth"):
            gt = parse_ground_truth_from_patient_prompt(patient_prompt, confounders)
            ground_truth_list.append(gt)

        # Add ground truth columns
        for conf in confounders:
            name = conf["name"]
            true_col = f"true_{name}"
            df[true_col] = [gt.get(name) for gt in ground_truth_list]

        # Evaluate accuracy
        metrics = evaluate_extraction_accuracy(df, confounders)

        logger.info("\n" + "=" * 60)
        logger.info("EXTRACTION ACCURACY METRICS")
        logger.info("=" * 60)

        # Per-confounder metrics
        for name, conf_metrics in metrics["per_confounder"].items():
            logger.info(f"\n{name} ({conf_metrics['type']}):")
            if conf_metrics["type"] == "categorical":
                logger.info(f"  Accuracy: {conf_metrics['accuracy']:.4f}")
                logger.info(f"  Coverage: {conf_metrics['coverage']:.4f}")
                logger.info(f"  Valid/Unknown: {conf_metrics['valid_extractions']}/{conf_metrics['unknown_extractions']}")
                if conf_metrics.get("per_category_accuracy"):
                    logger.info("  Per-category accuracy:")
                    for cat, acc in conf_metrics["per_category_accuracy"].items():
                        logger.info(f"    {cat}: {acc:.4f}")
            else:
                if conf_metrics["correlation"] is not None:
                    logger.info(f"  Correlation: {conf_metrics['correlation']:.4f}")
                    logger.info(f"  MAE: {conf_metrics['mae']:.4f}")
                    logger.info(f"  RMSE: {conf_metrics['rmse']:.4f}")
                logger.info(f"  Coverage: {conf_metrics['coverage']:.4f}")
                logger.info(f"  Valid/Missing: {conf_metrics['valid_extractions']}/{conf_metrics['missing_extractions']}")

        # Aggregate metrics
        logger.info(f"\n{'=' * 60}")
        logger.info("AGGREGATE METRICS:")
        agg = metrics["aggregate"]
        if agg.get("categorical_accuracy") is not None:
            logger.info(f"  Categorical accuracy (all): {agg['categorical_accuracy']:.4f}")
            logger.info(f"  Total categorical samples: {agg['total_categorical_samples']}")
        if agg.get("mean_continuous_correlation") is not None:
            logger.info(f"  Mean continuous correlation: {agg['mean_continuous_correlation']:.4f}")
            logger.info(f"  Num continuous confounders: {agg['num_continuous_confounders']}")

        # Save metrics to JSON
        metrics_path = Path(args.output).with_suffix('.metrics.json')
        with open(metrics_path, 'w') as f:
            # Convert numpy values to Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(v) for v in obj]
                return obj

            json.dump(convert_numpy(metrics), f, indent=2)
        logger.info(f"\nSaved metrics to: {metrics_path}")

    elif args.skip_evaluation:
        logger.info("\nSkipping accuracy evaluation (--skip-evaluation)")
    else:
        logger.info(f"\nSkipping accuracy evaluation: '{args.patient_prompt_column}' column not found")

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"\nSaved dataset with extractions to: {output_path}")


if __name__ == "__main__":
    main()
