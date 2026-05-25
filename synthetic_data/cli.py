# synthetic_data/cli.py
"""Command-line interface for synthetic data generation and confounder extraction."""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import SyntheticDataConfig, LLMConfig, StructuredDataConfig, DEFAULT_CLINICAL_QUESTION
from .generator import generate_synthetic_dataset, generate_synthetic_dataset_batch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Confounder extraction helpers
# ---------------------------------------------------------------------------

def truncate_to_last_n_tokens(text: str, tokenizer, max_tokens: int) -> str:
    """Truncate text to the last N tokens using the model's tokenizer."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text
    truncated_ids = token_ids[-max_tokens:]
    return tokenizer.decode(truncated_ids, skip_special_tokens=True)


def build_extraction_prompt(clinical_text: str, confounders: List[Dict[str, Any]]) -> str:
    """Build a dynamic extraction prompt based on confounder definitions."""
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
    """Parse harmony format response to extract the final channel content."""
    if not text:
        return text

    final_match = re.search(r'<\|channel\|>final[^<]*<\|message\|>(.+?)(?:<\||$)', text, re.DOTALL)
    if final_match:
        return final_match.group(1).strip()

    message_match = re.search(r'<\|message\|>(.+?)(?:<\||$)', text, re.DOTALL)
    if message_match:
        return message_match.group(1).strip()

    return text


def strip_reasoning_prefix(text: str, marker: str = "assistantfinal") -> str:
    """Strip reasoning prefix from model output."""
    if not text or not marker:
        return text

    if '<|channel|>' in text or '<|message|>' in text:
        return parse_harmony_response(text)

    marker_lower = marker.lower()
    text_lower = text.lower()

    idx = text_lower.find(marker_lower)
    if idx != -1:
        return text[idx + len(marker):].strip()

    return text


def parse_extraction_response(
    response: str,
    confounders: List[Dict[str, Any]],
    reasoning_marker: str = "assistantfinal",
) -> Dict[str, Any]:
    """Parse the LLM JSON response to extract all confounder values."""
    response = strip_reasoning_prefix(response, reasoning_marker)
    response = response.strip()

    json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
    else:
        json_str = response

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        logger.warning(f"Could not parse JSON response: {response[:200]}")
        result = {}
        for conf in confounders:
            name = conf["name"]
            if conf["type"] == "categorical":
                result[name] = "unknown"
            else:
                result[name] = None
        return result

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
    """Parse ground truth values from the patient_prompt column."""
    result = {}

    for conf in confounders:
        name = conf["name"]
        conf_type = conf["type"]
        description = conf.get("description", name.replace("_", " ").title())

        pattern = rf'-\s*{re.escape(description)}:\s*'

        for line in patient_prompt.split('\n'):
            if re.search(pattern, line, re.IGNORECASE):
                match = re.search(pattern + r'(.+?)(?:\s*\(|$)', line, re.IGNORECASE)
                if match:
                    value_str = match.group(1).strip()

                    if conf_type == "continuous":
                        num_match = re.search(r'([\d.]+)', value_str)
                        if num_match:
                            try:
                                result[name] = float(num_match.group(1))
                            except ValueError:
                                result[name] = None
                        else:
                            result[name] = None
                    else:  # categorical
                        quoted_match = re.search(r'"([^"]+)"', value_str)
                        if quoted_match:
                            result[name] = quoted_match.group(1)
                        else:
                            result[name] = value_str.strip()
                    break

        if name not in result:
            if conf_type == "categorical":
                result[name] = "unknown"
            else:
                result[name] = None

    return result


def _make_empty_result(confounders: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return unknown/None for all confounders."""
    result = {}
    for conf in confounders:
        if conf["type"] == "categorical":
            result[conf["name"]] = "unknown"
        else:
            result[conf["name"]] = None
    return result


class _OpenAIExtractionClient:
    """Client for OpenAI-compatible API (including vLLM server)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        confounders: Optional[List[Dict[str, Any]]] = None,
        max_text_tokens: int = 100000,
        reasoning_marker: str = "assistantfinal",
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

        from transformers import AutoTokenizer
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.confounders = confounders or []
        self.max_text_tokens = max_text_tokens
        self.reasoning_marker = reasoning_marker
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        logger.info(f"Initialized OpenAI client: {base_url} / {model}")

    def extract_batch(
        self,
        clinical_texts: List[str],
        batch_size: int = 10,
        temperature: float = 0.0,
        max_tokens: int = 5000,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        results = []
        raw_responses = []

        for i in tqdm(range(0, len(clinical_texts), batch_size), desc="Extracting"):
            batch = clinical_texts[i:i + batch_size]

            for text in batch:
                truncated = truncate_to_last_n_tokens(text, self.tokenizer, self.max_text_tokens)
                prompt = build_extraction_prompt(truncated, self.confounders)

                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    msg = response.choices[0].message
                    content = msg.content

                    if content is None:
                        if hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                            content = msg.reasoning_content
                        elif hasattr(msg, 'model_extra') and msg.model_extra:
                            extra = msg.model_extra
                            if 'reasoning_content' in extra:
                                content = extra['reasoning_content']
                            elif 'reasoning' in extra:
                                content = extra['reasoning']
                        elif hasattr(msg, '__dict__'):
                            for key in ['reasoning_content', 'reasoning', 'text']:
                                if key in msg.__dict__ and msg.__dict__[key]:
                                    content = msg.__dict__[key]
                                    break

                    if content is None:
                        logger.warning(f"LLM returned None content. Message fields: {dir(msg)}")
                        try:
                            msg_dict = msg.model_dump() if hasattr(msg, 'model_dump') else dict(msg)
                            logger.warning(f"Message dict: {msg_dict}")
                        except Exception:
                            pass
                        results.append(_make_empty_result(self.confounders))
                        raw_responses.append("<None>")
                        continue

                    raw_response = content.strip()
                    raw_responses.append(raw_response)
                    extracted = parse_extraction_response(
                        raw_response, self.confounders, self.reasoning_marker
                    )
                    results.append(extracted)
                except Exception as e:
                    logger.error(f"Error extracting from text: {e}")
                    results.append(_make_empty_result(self.confounders))
                    raw_responses.append(f"<Error: {e}>")

        return results, raw_responses


class _VLLMExtractionClient:
    """Direct vLLM inference client for confounder extraction."""

    def __init__(
        self,
        model_name: str = "openai/gpt-oss-120b",
        download_dir: str = "/data1/ken/models",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        confounders: Optional[List[Dict[str, Any]]] = None,
        max_text_tokens: int = 100000,
        reasoning_marker: str = "assistantfinal",
    ):
        try:
            from vllm import LLM, SamplingParams  # noqa: F401
        except ImportError:
            raise ImportError("vllm package required. Install with: pip install vllm")

        logger.info(f"Loading vLLM model: {model_name} with TP={tensor_parallel_size}")

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            download_dir=download_dir,
        )
        self.model_name = model_name
        self.confounders = confounders or []
        self.max_text_tokens = max_text_tokens
        self.reasoning_marker = reasoning_marker
        self.tokenizer = self.llm.get_tokenizer()
        logger.info("vLLM model loaded successfully")

    def extract_batch(
        self,
        clinical_texts: List[str],
        batch_size: int = 100,
        temperature: float = 0.0,
        max_tokens: int = 5000,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        from vllm import SamplingParams

        messages_list = []
        for text in clinical_texts:
            truncated = truncate_to_last_n_tokens(text, self.tokenizer, self.max_text_tokens)
            user_content = build_extraction_prompt(truncated, self.confounders)
            messages_list.append([{"role": "user", "content": user_content}])

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=["User:"],
        )

        logger.info(f"Running vLLM batch chat inference on {len(messages_list)} texts...")
        outputs = self.llm.chat(messages_list, sampling_params=sampling_params)

        results = []
        raw_responses = []
        for output in outputs:
            if output.outputs and len(output.outputs) > 0:
                raw_response = output.outputs[0].text.strip()
                raw_responses.append(raw_response)
                extracted = parse_extraction_response(
                    raw_response, self.confounders, self.reasoning_marker
                )
                results.append(extracted)
            else:
                results.append(_make_empty_result(self.confounders))
                raw_responses.append("<Empty>")

        return results, raw_responses


def evaluate_extraction_accuracy(
    df: pd.DataFrame,
    confounders: List[Dict[str, Any]],
    extracted_prefix: str = "llm_extracted_",
    true_prefix: str = "true_",
) -> Dict[str, Any]:
    """Evaluate extraction accuracy against ground truth for all confounders."""
    metrics: Dict[str, Any] = {
        "per_confounder": {},
        "aggregate": {},
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
                "per_category_accuracy": per_category,
            }

        else:  # continuous
            valid_mask = df[extracted_col].notna() & df[true_col].notna()
            valid_df = df[valid_mask]

            total = len(df)
            total_valid = len(valid_df)
            missing_count = (~valid_mask).sum()

            if total_valid > 1:
                extracted_vals = valid_df[extracted_col].astype(float)
                true_vals = valid_df[true_col].astype(float)

                correlation = np.corrcoef(extracted_vals, true_vals)[0, 1]
                if not np.isnan(correlation):
                    all_continuous_correlations.append(correlation)

                mae = np.abs(extracted_vals - true_vals).mean()
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
                "missing_extractions": missing_count,
            }

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


def _convert_numpy(obj):
    """Convert numpy values to Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy(v) for v in obj]
    return obj


def run_confounder_extraction(
    df: pd.DataFrame,
    confounders: List[Dict[str, Any]],
    output_dir: str,
    use_vllm_direct: bool = False,
    model_name: str = "openai/gpt-oss-120b",
    api_url: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    tensor_parallel_size: int = 2,
    download_dir: str = "/data1/ken/models",
    batch_size: int = 10,
    max_text_tokens: int = 100000,
    max_completion_tokens: int = 5000,
    reasoning_marker: str = "assistantfinal",
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """Run confounder extraction and evaluation on a generated dataset.

    Args:
        df: DataFrame with 'clinical_text' and optionally 'patient_prompt' columns
        confounders: List of confounder definitions from metadata
        output_dir: Directory to save extraction outputs
        use_vllm_direct: Use direct vLLM inference instead of API
        model_name: Model name/path
        api_url: OpenAI-compatible API URL (used when use_vllm_direct=False)
        api_key: API key
        tensor_parallel_size: Tensor parallel size for vLLM
        download_dir: Model download directory for vLLM
        batch_size: Batch size for extraction
        max_text_tokens: Max tokens to keep from end of each clinical text
        max_completion_tokens: Max tokens for model response
        reasoning_marker: Marker to strip reasoning prefix

    Returns:
        Tuple of (augmented DataFrame, metrics dict or None)
    """
    logger.info(f"Extracting {len(confounders)} confounders:")
    for conf in confounders:
        if conf["type"] == "categorical":
            logger.info(f"  - {conf['name']} (categorical): {conf.get('categories', [])}")
        else:
            logger.info(f"  - {conf['name']} (continuous)")

    # Initialize extraction client
    if use_vllm_direct:
        client = _VLLMExtractionClient(
            model_name=model_name,
            download_dir=download_dir,
            tensor_parallel_size=tensor_parallel_size,
            confounders=confounders,
            max_text_tokens=max_text_tokens,
            reasoning_marker=reasoning_marker,
        )
    else:
        client = _OpenAIExtractionClient(
            base_url=api_url,
            model=model_name,
            api_key=api_key,
            confounders=confounders,
            max_text_tokens=max_text_tokens,
            reasoning_marker=reasoning_marker,
        )

    # Extract confounders
    clinical_texts = df["clinical_text"].tolist()
    extracted_values_list, raw_responses = client.extract_batch(
        clinical_texts,
        batch_size=batch_size,
        max_tokens=max_completion_tokens,
    )

    # Add extracted values to dataframe
    for conf in confounders:
        name = conf["name"]
        df[f"llm_extracted_{name}"] = [ev.get(name) for ev in extracted_values_list]

    df["llm_raw_response"] = raw_responses

    # Log extraction statistics
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
                logger.info("    No valid extractions")

    # Parse ground truth and evaluate accuracy
    metrics = None
    if "patient_prompt" in df.columns:
        logger.info("\nParsing ground truth from 'patient_prompt' column...")

        ground_truth_list = []
        for patient_prompt in tqdm(df["patient_prompt"], desc="Parsing ground truth"):
            gt = parse_ground_truth_from_patient_prompt(patient_prompt, confounders)
            ground_truth_list.append(gt)

        for conf in confounders:
            name = conf["name"]
            df[f"true_{name}"] = [gt.get(name) for gt in ground_truth_list]

        metrics = evaluate_extraction_accuracy(df, confounders)

        logger.info("\n" + "=" * 60)
        logger.info("EXTRACTION ACCURACY METRICS")
        logger.info("=" * 60)

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

        agg = metrics["aggregate"]
        logger.info(f"\n{'=' * 60}")
        logger.info("AGGREGATE METRICS:")
        if agg.get("categorical_accuracy") is not None:
            logger.info(f"  Categorical accuracy (all): {agg['categorical_accuracy']:.4f}")
            logger.info(f"  Total categorical samples: {agg['total_categorical_samples']}")
        if agg.get("mean_continuous_correlation") is not None:
            logger.info(f"  Mean continuous correlation: {agg['mean_continuous_correlation']:.4f}")
            logger.info(f"  Num continuous confounders: {agg['num_continuous_confounders']}")

        # Save metrics
        metrics_path = Path(output_dir) / "dataset_with_extraction.metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(_convert_numpy(metrics), f, indent=2)
        logger.info(f"\nSaved metrics to: {metrics_path}")
    else:
        logger.info("\nSkipping accuracy evaluation: 'patient_prompt' column not found")

    # Save augmented dataset
    output_path = Path(output_dir) / "dataset_with_extraction.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved dataset with extractions to: {output_path}")

    return df, metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic clinical datasets with known causal structure using LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with local vLLM server
  python -m synthetic_data.cli --api-url http://localhost:8000/v1 --dataset-size 100

  # Generate with OpenAI API
  python -m synthetic_data.cli --api-url https://api.openai.com/v1 --api-key $OPENAI_API_KEY --model gpt-4

  # Load from config file (CLI args override config file values)
  python -m synthetic_data.cli --config my_config.json --dataset-size 1000

  # Custom clinical question with positivity enforcement
  python -m synthetic_data.cli --clinical-question "Compare pembrolizumab with nivolumab for NSCLC" --enforce-positivity

    # Single GPU (original behavior)
  python -m synthetic_data.cli --use-vllm-batch --tensor-parallel-size 2

  # Multi-GPU with 2 parallel workers (4 GPUs, 2 per worker)
  python -m synthetic_data.cli --use-vllm-batch \
    --gpu-devices 0,1,2,3 --tensor-parallel-size 2 \
    --dataset-size 100 --output-dir ./test_multi_gpu

  # Multi-GPU with 4 parallel workers (8 GPUs, 2 per worker)
  python -m synthetic_data.cli --use-vllm-batch \
    --gpu-devices 0,1,2,3,4,5,6,7 --tensor-parallel-size 2

  # Generate and extract confounders in one step
  python -m synthetic_data.cli --use-vllm-batch --dataset-size 100 --extract-confounders
        """,
    )

    # Config file (loaded first, then CLI args override)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file. CLI arguments override config file values.",
    )

    # Clinical question
    parser.add_argument(
        "--clinical-question",
        type=str,
        default=DEFAULT_CLINICAL_QUESTION,
        help="Comparative effectiveness research question",
    )

    # Dataset parameters
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=500,
        help="Number of patients to generate (default: 500)",
    )
    parser.add_argument(
        "--treatment-effect",
        type=float,
        default=0.20,
        help="Target treatment effect on probability scale (e.g., 0.10 = 10%% increase in outcome probability, default: 0.20)",
    )
    parser.add_argument(
        "--target-treatment-rate",
        type=float,
        default=0.5,
        help="Target proportion of patients receiving treatment=1 (default: 0.5)",
    )
    parser.add_argument(
        "--target-control-outcome-rate",
        type=float,
        default=0.5,
        help="Target outcome rate in control group (treatment=0) (default: 0.5)",
    )
    parser.add_argument(
        "--num-features",
        dest="num_features",
        type=int,
        default=None,
        help="Total number of role-tagged features to generate (default: 8-12, determined by LLM)",
    )
    parser.add_argument(
        "--num-confounders",
        type=int,
        default=None,
        help="Exact number of generated features that should have the confounder role",
    )
    parser.add_argument(
        "--num-effect-modifiers",
        type=int,
        default=None,
        help="Exact number of generated features that should have the effect_modifier role",
    )

    # Positivity enforcement
    parser.add_argument(
        "--enforce-positivity",
        action="store_true",
        help="Enforce minimum treatment/control rates per confounder stratum (avoids positivity violations)",
    )
    parser.add_argument(
        "--min-treatment-rate",
        type=float,
        default=0.1,
        help="Minimum P(T=1|X) per stratum when --enforce-positivity is set (default: 0.1)",
    )
    parser.add_argument(
        "--max-treatment-rate",
        type=float,
        default=0.9,
        help="Maximum P(T=1|X) per stratum when --enforce-positivity is set (default: 0.9)",
    )
    parser.add_argument(
        "--target-logit-std",
        type=float,
        default=2.0,
        help="Target std of logits; lower values compress propensities toward 0.5 (default: 2.0)",
    )

    # Generation mode
    parser.add_argument(
        "--generation-mode",
        type=str,
        choices=["single_document", "two_stage"],
        default="two_stage",
        help="Generation mode: 'single_document' (legacy single blob) or 'two_stage' (event timeline + note expansion, default)",
    )
    parser.add_argument(
        "--min-events",
        type=int,
        default=15,
        help="Minimum events per patient in two-stage mode (default: 15)",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=30,
        help="Maximum events per patient in two-stage mode (default: 30)",
    )
    parser.add_argument(
        "--note-separator",
        type=str,
        default="\n\n<new_note>\n\n",
        help="Separator between concatenated notes in two-stage mode (default: <new_note> tag)",
    )
    parser.add_argument(
        "--drug-perturbation-prob",
        type=float,
        default=0.3,
        help="Probability of generic->brand drug name swapping per note (default: 0.3)",
    )

    # Structured clinical data events
    parser.add_argument(
        "--structured-data",
        action="store_true",
        help="Enable structured clinical data events (encounters, labs, hospitalizations, PROs) in the generated text",
    )
    parser.add_argument(
        "--no-encounters",
        action="store_true",
        help="Disable encounter records (ICD-10/CPT) when --structured-data is enabled",
    )
    parser.add_argument(
        "--no-labs",
        action="store_true",
        help="Disable laboratory results when --structured-data is enabled",
    )
    parser.add_argument(
        "--no-hospitalizations",
        action="store_true",
        help="Disable hospitalization records when --structured-data is enabled",
    )
    parser.add_argument(
        "--no-pros",
        action="store_true",
        help="Disable patient-reported outcomes when --structured-data is enabled",
    )

    # LLM parameters
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000/v1",
        help="OpenAI-compatible API base URL (default: http://localhost:8000/v1 for vLLM)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="API key (can be blank for local models)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-oss-120b",
        help="Model name to use (default: 'openai/gpt-oss-120b')",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature (default: 0.7)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50000,
        help="Max tokens per LLM response (default: 20000)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./synthetic_data/example_synthetic_datasets",
        help="Output directory for generated files (default: ./synthetic_data/example_synthetic_datasets)",
    )

    # Execution
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=40,
        help="Number of parallel workers for patient generation (default: 4)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--use-vllm-batch",
        action="store_true",
        help="Use direct vLLM batch inference (faster, no server needed)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="Tensor parallel size for vLLM (default: 2)",
    )
    parser.add_argument(
        "--gpu-devices",
        type=str,
        default=None,
        help="Comma-separated GPU device IDs for multi-GPU parallelization (e.g., '0,1,2,3'). "
             "When specified with --use-vllm-batch, spawns multiple parallel workers. "
             "Number of workers = len(gpu_devices) / tensor_parallel_size. "
             "If not specified, uses CUDA_VISIBLE_DEVICES or all available GPUs.",
    )
    parser.add_argument(
        "--reasoning-marker",
        type=str,
        default="assistantfinal",
        help="Marker to strip reasoning prefix from clinical text (default: 'assistantfinal'). Set to empty string to disable.",
    )

    parser.add_argument(
        "--vllm-download-dir",
        type=str,
        default="./",
        help="Download directory for vllm model",
    )
    parser.add_argument(
        "--outcome-type",
        type=str,
        choices=["binary", "continuous"],
        default="binary",
        help="Type of outcome: 'binary' (default) or 'continuous'",
    )
    parser.add_argument(
        "--outcome-noise-std",
        type=float,
        default=1.0,
        help="Noise standard deviation for continuous outcomes (default: 1.0)",
    )

    # Explicit feature extraction (post-generation)
    parser.add_argument(
        "--extract-features",
        "--extract-confounders",
        dest="extract_features",
        action="store_true",
        default=True,
        help="After generation, extract explicit features from clinical text using the same LLM and evaluate accuracy",
    )
    parser.add_argument(
        "--extraction-max-text-tokens",
        type=int,
        default=100000,
        help="Max tokens to keep from end of each clinical text for extraction (default: 100000)",
    )
    parser.add_argument(
        "--extraction-max-completion-tokens",
        type=int,
        default=5000,
        help="Max tokens for extraction model response (default: 5000)",
    )
    parser.add_argument(
        "--extraction-batch-size",
        type=int,
        default=10,
        help="Batch size for confounder extraction (default: 10)",
    )

    args = parser.parse_args()

    if args.num_features is None and (
        args.num_confounders is not None or args.num_effect_modifiers is not None
    ):
        args.num_features = (args.num_confounders or 0) + (args.num_effect_modifiers or 0)

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Build configuration: load from file first, then override with CLI args
    if args.config:
        logging.info(f"Loading config from: {args.config}")
        config = SyntheticDataConfig.from_json(args.config)
        # Override with any explicitly provided CLI args
        # We check against defaults to see if user provided a value
        if args.clinical_question != DEFAULT_CLINICAL_QUESTION:
            config.clinical_question = args.clinical_question
        if args.dataset_size != 500:
            config.dataset_size = args.dataset_size
        if args.treatment_effect != 0.20:
            config.treatment_effect_prob = args.treatment_effect
        if args.target_treatment_rate != 0.5:
            config.target_treatment_rate = args.target_treatment_rate
        if args.target_control_outcome_rate != 0.5:
            config.target_control_outcome_rate = args.target_control_outcome_rate
        if args.enforce_positivity:
            config.enforce_positivity = True
        if args.min_treatment_rate != 0.1:
            config.min_treatment_rate_per_stratum = args.min_treatment_rate
        if args.max_treatment_rate != 0.9:
            config.max_treatment_rate_per_stratum = args.max_treatment_rate
        if args.target_logit_std != 2.0:
            config.target_logit_std = args.target_logit_std
        if args.num_features is not None:
            config.num_features = args.num_features
        if args.num_confounders is not None:
            config.num_confounders = args.num_confounders
        if args.num_effect_modifiers is not None:
            config.num_effect_modifiers = args.num_effect_modifiers
        if args.outcome_type != "binary":
            config.outcome_type = args.outcome_type
        if args.outcome_noise_std != 1.0:
            config.outcome_noise_std = args.outcome_noise_std
        # Two-stage generation overrides
        if args.generation_mode != "two_stage":
            config.generation_mode = args.generation_mode
        if args.min_events != 15:
            config.min_events_per_patient = args.min_events
        if args.max_events != 30:
            config.max_events_per_patient = args.max_events
        if args.note_separator != "\n\n<new_note>\n\n":
            config.note_separator = args.note_separator
        if args.drug_perturbation_prob != 0.3:
            config.drug_perturbation_prob = args.drug_perturbation_prob
        # Structured data overrides
        if args.structured_data:
            config.structured_data.enabled = True
        if args.no_encounters:
            config.structured_data.include_encounters = False
        if args.no_labs:
            config.structured_data.include_labs = False
        if args.no_hospitalizations:
            config.structured_data.include_hospitalizations = False
        if args.no_pros:
            config.structured_data.include_pros = False
        if args.output_dir != "./synthetic_output":
            config.output_dir = args.output_dir
        if args.seed != 42:
            config.seed = args.seed
        # LLM overrides
        if args.api_url != "http://localhost:8000/v1":
            config.llm.api_base_url = args.api_url
        if args.api_key != "":
            config.llm.api_key = args.api_key
        if args.model != "openai/gpt-oss-120b":
            config.llm.model_name = args.model
        if args.temperature != 0.7:
            config.llm.temperature = args.temperature
        if args.max_tokens != 50000:
            config.llm.max_tokens = args.max_tokens
    else:
        # Build config from CLI args
        config = SyntheticDataConfig(
            clinical_question=args.clinical_question,
            dataset_size=args.dataset_size,
            treatment_effect_prob=args.treatment_effect,
            target_treatment_rate=args.target_treatment_rate,
            target_control_outcome_rate=args.target_control_outcome_rate,
            enforce_positivity=args.enforce_positivity,
            min_treatment_rate_per_stratum=args.min_treatment_rate,
            max_treatment_rate_per_stratum=args.max_treatment_rate,
            target_logit_std=args.target_logit_std,
            num_features=args.num_features,
            num_confounders=args.num_confounders,
            num_effect_modifiers=args.num_effect_modifiers,
            outcome_type=args.outcome_type,
            outcome_noise_std=args.outcome_noise_std,
            generation_mode=args.generation_mode,
            min_events_per_patient=args.min_events,
            max_events_per_patient=args.max_events,
            note_separator=args.note_separator,
            drug_perturbation_prob=args.drug_perturbation_prob,
            structured_data=StructuredDataConfig(
                enabled=args.structured_data,
                include_encounters=not args.no_encounters,
                include_hospitalizations=not args.no_hospitalizations,
                include_labs=not args.no_labs,
                include_pros=not args.no_pros,
            ),
            output_dir=args.output_dir,
            seed=args.seed,
            llm=LLMConfig(
                api_base_url=args.api_url,
                api_key=args.api_key,
                model_name=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            ),
        )

    # Save config to output directory before generation
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_output_path = output_dir / "generation_config.json"
    config.to_json(str(config_output_path))
    logging.info(f"Config saved to: {config_output_path}")

    # Parse GPU devices (needed for both generation and extraction)
    gpu_device_ids = None
    if args.gpu_devices:
        try:
            gpu_device_ids = [int(d.strip()) for d in args.gpu_devices.split(",")]
        except ValueError:
            logging.error(f"Invalid --gpu-devices format: {args.gpu_devices}. Expected comma-separated integers.")
            sys.exit(1)

        if args.use_vllm_batch:
            if len(gpu_device_ids) % args.tensor_parallel_size != 0:
                logging.error(
                    f"Number of GPU devices ({len(gpu_device_ids)}) must be divisible by "
                    f"tensor_parallel_size ({args.tensor_parallel_size})"
                )
                sys.exit(1)

            num_workers = len(gpu_device_ids) // args.tensor_parallel_size
            logging.info(f"Multi-GPU mode: {len(gpu_device_ids)} GPUs -> {num_workers} parallel workers "
                        f"(tensor_parallel_size={args.tensor_parallel_size})")

    # Run generation
    try:
        if args.use_vllm_batch:
            # Use direct vLLM batch inference (faster)
            from .vllm_batch_client import VLLMConfig

            vllm_config = VLLMConfig(
                model_name=args.model,
                tensor_parallel_size=args.tensor_parallel_size,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                download_dir=args.vllm_download_dir,
                reasoning_marker=args.reasoning_marker if args.reasoning_marker else None,
            )
            df, metadata = generate_synthetic_dataset_batch(
                config=config,
                vllm_config=vllm_config,
                show_progress=True,
                gpu_device_ids=gpu_device_ids,
            )
        else:
            df, metadata = generate_synthetic_dataset(
                config=config,
                num_workers=args.num_workers,
                show_progress=True,
            )

        print(f"\n✓ Generated {len(df)} patients")
        print(f"  - Treatment rate: {df['treatment_indicator'].mean():.1%}")
        if config.outcome_type == "continuous":
            print(f"  - Outcome mean: {df['outcome_indicator'].mean():.2f} (std: {df['outcome_indicator'].std():.2f})")
        else:
            print(f"  - Outcome rate: {df['outcome_indicator'].mean():.1%}")
        if config.enforce_positivity:
            print(f"  - Positivity enforcement: ON (min={config.min_treatment_rate_per_stratum:.0%}, max={config.max_treatment_rate_per_stratum:.0%})")
        print(f"  - Generation mode: {config.generation_mode}")
        if config.generation_mode == "two_stage" and "num_notes" in df.columns:
            print(f"  - Notes per patient: {df['num_notes'].mean():.1f} avg ({df['num_notes'].min()}-{df['num_notes'].max()} range)")
        print(f"  - Config: {config.output_dir}/generation_config.json")
        print(f"  - Dataset: {config.output_dir}/dataset.parquet")
        print(f"  - Metadata: {config.output_dir}/metadata.json")

    except Exception as e:
        logging.error(f"Generation failed: {e}", exc_info=True)
        sys.exit(1)

    # Optional: Extract explicit features from generated text
    if args.extract_features:
        confounders = metadata.get("features", metadata.get("confounders", []))
        if not confounders:
            logging.warning("No explicit features in metadata -- skipping extraction")
        else:
            try:
                print(f"\n--- Explicit Feature Extraction ---")
                print(f"Extracting {len(confounders)} features from clinical text...")

                df_extracted, metrics = run_confounder_extraction(
                    df=df,
                    confounders=confounders,
                    output_dir=config.output_dir,
                    use_vllm_direct=args.use_vllm_batch,
                    model_name=args.model,
                    api_url=args.api_url,
                    api_key=args.api_key if args.api_key else "EMPTY",
                    tensor_parallel_size=args.tensor_parallel_size,
                    download_dir=args.vllm_download_dir,
                    batch_size=args.extraction_batch_size,
                    max_text_tokens=args.extraction_max_text_tokens,
                    max_completion_tokens=args.extraction_max_completion_tokens,
                    reasoning_marker=args.reasoning_marker,
                )

                if metrics:
                    agg = metrics.get("aggregate", {})
                    if agg.get("categorical_accuracy") is not None:
                        print(f"  - Categorical accuracy: {agg['categorical_accuracy']:.4f}")
                    if agg.get("mean_continuous_correlation") is not None:
                        print(f"  - Mean continuous correlation: {agg['mean_continuous_correlation']:.4f}")
                print(f"  - Saved: {config.output_dir}/dataset_with_extraction.parquet")
                print(f"  - Metrics: {config.output_dir}/dataset_with_extraction.metrics.json")

            except Exception as e:
                logging.error(f"Confounder extraction failed: {e}", exc_info=True)
                print(f"\n⚠ Confounder extraction failed (dataset was saved successfully)")


if __name__ == "__main__":
    main()
