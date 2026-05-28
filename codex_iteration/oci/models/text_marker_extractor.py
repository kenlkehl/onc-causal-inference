"""Raw-note clinical marker extractor.

This extractor is intentionally lightweight: it reads high-signal clinical
markers from the text itself and exposes them as tensors that can be consumed by
the usual neural heads and the X/W causal forest split.  It does not read
dataset explicit-feature columns.
"""

import re
from typing import Any, Dict, List

import torch
import torch.nn as nn


class TextMarkerExtractor(nn.Module):
    """Extract age and PD-L1 markers from raw clinical text."""

    output_dim = 6

    _DASHES = {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }
    _SPACES = {
        "\u00a0": " ",
        "\u202f": " ",
    }

    def __init__(
        self,
        device: torch.device | str = "cpu",
        age_center: float = 66.0,
        age_scale: float = 12.0,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.age_center = float(age_center)
        self.age_scale = float(age_scale)
        self._marker_cache: Dict[tuple[int, int], List[float]] = {}

    @staticmethod
    def _texts_from_input(texts_or_batch: Any) -> List[str]:
        if isinstance(texts_or_batch, dict):
            return [str(text) for text in texts_or_batch.get("texts", [])]
        return [str(text) for text in texts_or_batch]

    @classmethod
    def _normalize_text(cls, text: str) -> str:
        for src, dst in cls._DASHES.items():
            text = text.replace(src, dst)
        for src, dst in cls._SPACES.items():
            text = text.replace(src, dst)
        return text

    def _extract_age(self, text: str) -> float | None:
        patterns = [
            r"patient is now\s+(\d+(?:\.\d+)?)\s+years old",
            r"At age\s+(\d+(?:\.\d+)?)\s*,\s*pre-?treatment",
            r"(\d+(?:\.\d+)?)\s*[- ]year[- ]old",
        ]
        matches: List[float] = []
        for pattern in patterns:
            matches.extend(
                float(match)
                for match in re.findall(pattern, text, flags=re.IGNORECASE)
            )
            if matches:
                return matches[-1]

        all_ages = [
            float(match)
            for match in re.findall(r"At age\s+(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
        ]
        return all_ages[-1] if all_ages else None

    @staticmethod
    def _pdl1_category_from_value(raw: str) -> str:
        raw = raw.replace(" ", "")
        if raw.startswith("<"):
            return "<1%"
        if raw.startswith("\u2265"):
            return "\u226550%"
        value = float(raw.replace(">", ""))
        if value < 1:
            return "<1%"
        if value >= 50:
            return "\u226550%"
        return "1-49%"

    def _extract_pdl1(self, text: str) -> str | None:
        patterns = [
            r"PD\s*-?\s*L1[^\n]{0,80}?(?:TPS|tumor proportion score|expression)?[^\d<>\u2265]*([<>\u2265]?\s*\d+(?:\.\d+)?)\s*%",
            r"(?:TPS|PD\s*-?\s*L1)[^\n]{0,80}?([<>\u2265]?\s*\d+(?:\.\d+)?)\s*%",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return self._pdl1_category_from_value(match.group(1))

        for category in ("<1%", "1-49%", "\u226550%"):
            if category in text:
                return category
        return None

    def _marker_row(self, text: str) -> List[float]:
        text = self._normalize_text(text)
        age = self._extract_age(text)
        if age is None:
            age_z = 0.0
            age_missing = 1.0
        else:
            age_z = (age - self.age_center) / self.age_scale
            age_missing = 0.0

        pdl1 = self._extract_pdl1(text)
        pdl1_low = 1.0 if pdl1 == "<1%" else 0.0
        pdl1_mid = 1.0 if pdl1 == "1-49%" else 0.0
        pdl1_high = 1.0 if pdl1 == "\u226550%" else 0.0
        pdl1_missing = 1.0 if pdl1 is None else 0.0
        return [age_z, age_missing, pdl1_low, pdl1_mid, pdl1_high, pdl1_missing]

    def forward(self, texts_or_batch: Any) -> torch.Tensor:
        texts = self._texts_from_input(texts_or_batch)
        rows = []
        for text in texts:
            cache_key = (hash(text), len(text))
            row = self._marker_cache.get(cache_key)
            if row is None:
                row = self._marker_row(text)
                self._marker_cache[cache_key] = row
            rows.append(row)
        return torch.tensor(rows, dtype=torch.float32, device=self.device)

    def extract_role_features(self, texts_or_batch: Any) -> Dict[str, torch.Tensor]:
        markers = self.forward(texts_or_batch)
        return {
            "confounder": markers[:, :2],
            "effect_modifier": markers[:, 2:],
        }

    def get_state(self) -> Dict[str, Any]:
        return {
            "extractor_type": "text_marker",
            "output_dim": self.output_dim,
            "age_center": self.age_center,
            "age_scale": self.age_scale,
        }

    def get_num_parameters(self) -> Dict[str, int]:
        return {"trainable": 0, "frozen": 0}
