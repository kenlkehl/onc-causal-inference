#!/usr/bin/env python
"""LLM-based confounder extraction from clinical text.

This script extracts confounder values from clinical text using an LLM (vLLM or OpenAI API).
For the concept-aware experiment, it extracts the number of metastatic sites.

Usage:
    python scripts/extract_confounders_llm.py \
        --input example_synthetic_data_one_confounder/dataset.parquet \
        --output example_synthetic_data_one_confounder/dataset_with_extraction.parquet \
        --vllm-url http://localhost:8000/v1 \
        --model openai/gpt-oss-120b
"""

import argparse
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global reasoning marker (can be set via command line)
REASONING_MARKER = "assistantfinal"


EXTRACTION_PROMPT = """Read this clinical note carefully and determine the number of metastatic sites.

Clinical Note:
{clinical_text}

Based on the information in this note, how many distinct metastatic sites does the patient have?
Respond with ONLY one of these options: 1, 2, 3, 4_or_more

Your answer:"""


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

    marker_lower = marker.lower()
    text_lower = text.lower()

    idx = text_lower.find(marker_lower)
    if idx != -1:
        # Return text after the marker
        return text[idx + len(marker):].strip()

    return text


def parse_extraction_response(response: str) -> str:
    """Parse the LLM response to extract the category.

    Args:
        response: Raw LLM response text

    Returns:
        One of: "1", "2", "3", "4_or_more", or "unknown" if parsing fails
    """
    # Strip reasoning prefix if present (default marker)
    response = strip_reasoning_prefix(response, REASONING_MARKER)
    response = response.strip().lower()

    # Check for exact matches first
    valid_categories = ["1", "2", "3", "4_or_more"]
    if response in valid_categories:
        return response

    # Handle variations
    if "4_or_more" in response or "4 or more" in response or "four or more" in response:
        return "4_or_more"
    if response.startswith("4") or "four" in response:
        return "4_or_more"

    # Try to extract just the number
    match = re.search(r'\b([1-3])\b', response)
    if match:
        return match.group(1)

    # Check for word numbers
    word_to_num = {"one": "1", "two": "2", "three": "3"}
    for word, num in word_to_num.items():
        if word in response:
            return num

    logger.warning(f"Could not parse response: {response[:100]}")
    return "unknown"


class OpenAIClient:
    """Client for OpenAI-compatible API (including vLLM)."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "openai/gpt-oss-120b",
        api_key: str = "EMPTY"
    ):
        """Initialize the client.

        Args:
            base_url: Base URL for the API (vLLM or OpenAI)
            model: Model name to use
            api_key: API key (use "EMPTY" for local vLLM)
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        logger.info(f"Initialized OpenAI client: {base_url} / {model}")

    def extract_batch(
        self,
        clinical_texts: List[str],
        batch_size: int = 10,
        temperature: float = 0.0,
        max_tokens: int = 5000
    ) -> Tuple[List[str], List[str]]:
        """Extract confounder values from a batch of clinical texts.

        Args:
            clinical_texts: List of clinical text strings
            batch_size: Number of texts to process in parallel (for rate limiting)
            temperature: LLM temperature (0 for deterministic)
            max_tokens: Maximum tokens in response

        Returns:
            Tuple of (extracted_values, raw_responses) lists
        """
        results = []
        raw_responses = []

        for i in tqdm(range(0, len(clinical_texts), batch_size), desc="Extracting"):
            batch = clinical_texts[i:i + batch_size]
            batch_results = []
            batch_raw = []

            for text in batch:
                prompt = EXTRACTION_PROMPT.format(clinical_text=text[:8000])  # Truncate if needed

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
                        batch_results.append("unknown")
                        batch_raw.append("<None>")
                        continue

                    raw_response = content.strip()
                    batch_raw.append(raw_response)
                    extracted = parse_extraction_response(raw_response)
                    batch_results.append(extracted)
                except Exception as e:
                    logger.error(f"Error extracting from text: {e}")
                    batch_results.append("unknown")
                    batch_raw.append(f"<Error: {e}>")

            results.extend(batch_results)
            raw_responses.extend(batch_raw)

        return results, raw_responses


class VLLMBatchClientWrapper:
    """Wrapper around VLLMBatchClient for direct vLLM inference."""

    def __init__(
        self,
        model_name: str = "openai/gpt-oss-120b",
        tensor_parallel_size: int = 2,
        gpu_memory_utilization: float = 0.90
    ):
        """Initialize the vLLM batch client.

        Args:
            model_name: Model name/path
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory fraction to use
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
            trust_remote_code=True
        )
        self.model_name = model_name
        logger.info("vLLM model loaded successfully")

    def extract_batch(
        self,
        clinical_texts: List[str],
        batch_size: int = 100,  # vLLM handles large batches efficiently
        temperature: float = 0.0,
        max_tokens: int =5000
    ) -> Tuple[List[str], List[str]]:
        """Extract confounder values from a batch of clinical texts.

        Args:
            clinical_texts: List of clinical text strings
            batch_size: Batch size for vLLM inference
            temperature: LLM temperature (0 for deterministic)
            max_tokens: Maximum tokens in response

        Returns:
            Tuple of (extracted_values, raw_responses) lists
        """
        from vllm import SamplingParams

        # Build prompts
        prompts = [
            f"User: {EXTRACTION_PROMPT.format(clinical_text=text[:8000])}\n\nAssistant:"
            for text in clinical_texts
        ]

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=["User:", "\n\n"]
        )

        logger.info(f"Running vLLM batch inference on {len(prompts)} texts...")
        outputs = self.llm.generate(prompts, sampling_params)

        results = []
        raw_responses = []
        for output in outputs:
            if output.outputs and len(output.outputs) > 0:
                raw_response = output.outputs[0].text.strip()
                raw_responses.append(raw_response)
                extracted = parse_extraction_response(raw_response)
                results.append(extracted)
            else:
                results.append("unknown")
                raw_responses.append("<Empty>")

        return results, raw_responses


def evaluate_extraction_accuracy(
    df: pd.DataFrame,
    extracted_column: str = "llm_extracted_metastatic_sites",
    true_column: str = "metastatic_site_count_category"
) -> dict:
    """Evaluate extraction accuracy against ground truth.

    Args:
        df: DataFrame with extracted and true values
        extracted_column: Column name for extracted values
        true_column: Column name for true values (if available)

    Returns:
        Dictionary with accuracy metrics
    """
    if true_column not in df.columns:
        logger.warning(f"True column '{true_column}' not found. Skipping accuracy evaluation.")
        return {}

    # Filter out unknown extractions
    valid_mask = df[extracted_column] != "unknown"
    valid_df = df[valid_mask]

    # Calculate accuracy
    correct = (valid_df[extracted_column] == valid_df[true_column].astype(str)).sum()
    total_valid = len(valid_df)
    total = len(df)
    unknown_count = (~valid_mask).sum()

    accuracy = correct / total_valid if total_valid > 0 else 0.0
    coverage = total_valid / total if total > 0 else 0.0

    # Per-category accuracy
    per_category = {}
    for category in ["1", "2", "3", "4_or_more"]:
        cat_mask = valid_df[true_column].astype(str) == category
        if cat_mask.sum() > 0:
            cat_correct = (
                valid_df.loc[cat_mask, extracted_column] == category
            ).sum()
            per_category[category] = cat_correct / cat_mask.sum()

    metrics = {
        "accuracy": accuracy,
        "coverage": coverage,
        "total_samples": total,
        "valid_extractions": total_valid,
        "unknown_extractions": unknown_count,
        "per_category_accuracy": per_category
    }

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
        "--text-column",
        type=str,
        default="clinical_text",
        help="Name of the text column"
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
        "--true-column",
        type=str,
        default=None,
        help="Column with true confounder values for accuracy evaluation"
    )
    parser.add_argument(
        "--reasoning-marker",
        type=str,
        default="assistantfinal",
        help="Marker text that precedes final answer in reasoning models"
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

    # Initialize client
    if args.vllm_direct:
        client = VLLMBatchClientWrapper(
            model_name=args.model,
            tensor_parallel_size=args.tensor_parallel_size
        )
    elif args.vllm_url:
        client = OpenAIClient(
            base_url=args.vllm_url,
            model=args.model,
            api_key=args.api_key
        )
    else:
        raise ValueError("Must specify either --vllm-url or --vllm-direct")

    # Extract confounders
    clinical_texts = df[args.text_column].tolist()
    extracted_values, raw_responses = client.extract_batch(
        clinical_texts,
        batch_size=args.batch_size
    )

    # Add to dataframe
    df["llm_extracted_metastatic_sites"] = extracted_values
    df["llm_raw_response"] = raw_responses

    # Report extraction statistics
    value_counts = df["llm_extracted_metastatic_sites"].value_counts()
    logger.info("\nExtraction value counts:")
    for val, count in value_counts.items():
        logger.info(f"  {val}: {count} ({count/len(df)*100:.1f}%)")

    # Evaluate accuracy if true column specified
    if args.true_column:
        metrics = evaluate_extraction_accuracy(
            df,
            extracted_column="llm_extracted_metastatic_sites",
            true_column=args.true_column
        )
        if metrics:
            logger.info("\nExtraction Accuracy Metrics:")
            logger.info(f"  Overall accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  Coverage: {metrics['coverage']:.4f}")
            logger.info(f"  Unknown extractions: {metrics['unknown_extractions']}")
            logger.info("  Per-category accuracy:")
            for cat, acc in metrics.get('per_category_accuracy', {}).items():
                logger.info(f"    {cat}: {acc:.4f}")

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"\nSaved dataset with extractions to: {output_path}")


if __name__ == "__main__":
    main()
