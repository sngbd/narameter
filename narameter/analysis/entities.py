"""
Named Entity Recognition (NER) for Narameter pipeline.

Extracts and tracks named entities (people, organizations, locations)
across time periods to identify *who* or *what* triggered narrative shifts.

Uses a multilingual transformer NER model via HuggingFace pipelines,
avoiding the need for spaCy or language-specific models.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    Multilingual named entity extractor using a pre-trained transformer.

    Tracks entity prominence across time periods to reveal which
    actors, organizations, or locations surged or declined at
    narrative shift points.

    Usage::

        extractor = EntityExtractor()
        entity_df = extractor.extract_entities(texts, period_labels)
    """

    # Merge sub-token NER labels into clean categories
    _LABEL_MAP = {
        "PER": "PERSON",
        "B-PER": "PERSON",
        "I-PER": "PERSON",
        "ORG": "ORG",
        "B-ORG": "ORG",
        "I-ORG": "ORG",
        "LOC": "LOCATION",
        "B-LOC": "LOCATION",
        "I-LOC": "LOCATION",
        "GPE": "LOCATION",
        "B-GPE": "LOCATION",
        "I-GPE": "LOCATION",
        "MISC": "MISC",
        "B-MISC": "MISC",
        "I-MISC": "MISC",
    }

    def __init__(
        self,
        model_name: str = "dslim/bert-base-NER",
        device: str = "cpu",
        batch_size: int = 16,
        top_n: int = 15,
    ):
        """
        Initialize the entity extractor.

        Args:
            model_name: HuggingFace NER model. Default is a BERT-based
                        model with strong multilingual entity recognition.
            device: Device to run inference on.
            batch_size: Batch size for NER inference.
            top_n: Number of top entities to track per period.
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.top_n = top_n
        self._pipeline = None

    def _load_pipeline(self):
        """Lazy-load the NER pipeline."""
        if self._pipeline is None:
            from transformers import pipeline as hf_pipeline

            logger.info(f"Loading NER model: {self.model_name}")
            self._pipeline = hf_pipeline(
                "ner",
                model=self.model_name,
                device=self.device,
                aggregation_strategy="first",
                batch_size=self.batch_size,
            )
            logger.info("NER model loaded.")

    def _normalize_entity(self, text: str) -> str:
        """Normalize entity text for deduplication."""
        # Remove sub-word artifacts and clean whitespace
        cleaned = text.replace("##", "").strip()
        # Collapse multiple spaces
        cleaned = " ".join(cleaned.split())
        return cleaned

    def extract_entities(
        self,
        texts: list[str],
        period_labels: list[str],
    ) -> pd.DataFrame:
        """
        Extract named entities and compute frequency per (entity, period).

        Args:
            texts: Flat list of document texts.
            period_labels: Corresponding period label for each document.

        Returns:
            DataFrame with columns:
                [period, entity, entity_type, frequency, normalized_freq]
        """
        self._load_pipeline()

        logger.info(f"Extracting entities from {len(texts)} documents...")

        # Collect entities per period
        period_entities: dict[str, Counter] = {}
        period_entity_types: dict[str, dict[str, str]] = {}
        period_doc_counts: dict[str, int] = Counter()

        for text, period in zip(texts, period_labels):
            period_doc_counts[period] += 1

            if not text or not text.strip():
                continue

            # Truncate long texts to avoid OOM
            truncated = text[:512]

            try:
                ner_results = self._pipeline(truncated)
            except Exception:
                continue

            if period not in period_entities:
                period_entities[period] = Counter()
                period_entity_types[period] = {}

            for ent in ner_results:
                label = self._LABEL_MAP.get(ent.get("entity_group", ""), None)
                if label is None:
                    continue

                entity_text = self._normalize_entity(ent["word"])
                if len(entity_text) < 2:
                    continue  # Skip single-char entities

                period_entities[period][entity_text] += 1
                period_entity_types[period][entity_text] = label

        # Build DataFrame
        rows = []
        for period in sorted(period_entities.keys()):
            doc_count = period_doc_counts[period]
            for entity, freq in period_entities[period].most_common(self.top_n):
                rows.append(
                    {
                        "period": period,
                        "entity": entity,
                        "entity_type": period_entity_types[period].get(entity, "MISC"),
                        "frequency": freq,
                        "normalized_freq": round(freq / doc_count, 4) if doc_count > 0 else 0,
                    }
                )

        df = pd.DataFrame(rows)
        logger.info(f"Extracted {len(df)} entity-period entries across {len(period_entities)} periods.")
        return df

    def get_entity_prominence_matrix(
        self, entity_df: pd.DataFrame, top_n: int = 10
    ) -> pd.DataFrame:
        """
        Build a (entity x period) prominence matrix for heatmap visualization.

        Args:
            entity_df: Output from extract_entities().
            top_n: Number of top global entities to include.

        Returns:
            DataFrame with entities as rows and periods as columns,
            values are normalized frequencies.
        """
        if entity_df.empty:
            return pd.DataFrame()

        # Find top entities globally by total frequency
        global_top = (
            entity_df.groupby("entity")["frequency"]
            .sum()
            .nlargest(top_n)
            .index.tolist()
        )

        filtered = entity_df[entity_df["entity"].isin(global_top)]
        pivot = filtered.pivot_table(
            index="entity",
            columns="period",
            values="normalized_freq",
            fill_value=0,
        )

        # Sort columns chronologically
        pivot = pivot[sorted(pivot.columns)]

        # Sort rows by total prominence
        pivot["_total"] = pivot.sum(axis=1)
        pivot = pivot.sort_values("_total", ascending=False).drop(columns=["_total"])

        return pivot
