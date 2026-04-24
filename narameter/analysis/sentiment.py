"""
Sentiment analysis for Narameter pipeline.

Computes sentiment polarity scores for tweets to distinguish
content shifts (topic changes) from narrative shifts (framing changes).

Uses a multilingual transformer model for zero-shot sentiment classification.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Multilingual sentiment analyzer using a pre-trained transformer model.

    Computes sentiment polarity scores in the range [-1, 1] where:
        -1 = very negative
         0 = neutral
        +1 = very positive

    Usage::

        analyzer = SentimentAnalyzer()
        scores = analyzer.compute_sentiment(["I love this!", "This is terrible."])
    """

    # Map star ratings (1-5) to polarity range [-1, 1]
    _STAR_TO_POLARITY = {1: -1.0, 2: -0.5, 3: 0.0, 4: 0.5, 5: 1.0}

    def __init__(
        self,
        model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment",
        device: str = "cpu",
        batch_size: int = 32,
    ):
        """
        Initialize the sentiment analyzer.

        Args:
            model_name: HuggingFace model name for sentiment classification.
            device: Device to run inference on ("cpu" or "cuda").
            batch_size: Batch size for inference.
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._pipeline = None

    def _load_pipeline(self):
        """Lazy-load the sentiment pipeline."""
        if self._pipeline is None:
            from transformers import pipeline as hf_pipeline

            logger.info(f"Loading sentiment model: {self.model_name}")
            self._pipeline = hf_pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=self.device,
                truncation=True,
                max_length=512,
            )
            logger.info("Sentiment model loaded.")

    def compute_sentiment(self, texts: list[str]) -> list[float]:
        """
        Compute sentiment polarity scores for a list of texts.

        Args:
            texts: List of text strings to analyze.

        Returns:
            List of polarity scores in [-1, 1].
        """
        self._load_pipeline()

        if not texts:
            return []

        scores = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            # Ensure no empty strings (causes errors)
            batch = [t if t.strip() else "neutral" for t in batch]
            results = self._pipeline(batch)
            for r in results:
                # nlptown model outputs "1 star" to "5 stars"
                star_label = r["label"]
                try:
                    stars = int(star_label.split()[0])
                except (ValueError, IndexError):
                    stars = 3  # fallback to neutral
                scores.append(self._STAR_TO_POLARITY.get(stars, 0.0))

        return scores

    def compute_topic_sentiment(
        self,
        texts: list[str],
        doc_topics: np.ndarray,
        period_labels: list[str],
    ) -> pd.DataFrame:
        """
        Compute mean sentiment per (topic, period) pair.

        Args:
            texts: Flat list of all texts (matching doc_topics order).
            doc_topics: 1-D array of topic IDs per document.
            period_labels: List of period labels per document.

        Returns:
            DataFrame with columns:
                [period, topic, mean_sentiment, doc_count]
        """
        logger.info(f"Computing sentiment for {len(texts)} documents...")
        sentiments = self.compute_sentiment(texts)

        df = pd.DataFrame(
            {
                "period": period_labels,
                "topic": doc_topics,
                "sentiment": sentiments,
            }
        )

        # Filter out outlier topic (-1)
        df = df[df["topic"] != -1]

        # Aggregate per (period, topic)
        grouped = (
            df.groupby(["period", "topic"])
            .agg(mean_sentiment=("sentiment", "mean"), doc_count=("sentiment", "count"))
            .reset_index()
        )

        # Also compute corpus-level sentiment per period
        corpus_sentiment = (
            df.groupby("period")
            .agg(
                corpus_mean_sentiment=("sentiment", "mean"),
                corpus_doc_count=("sentiment", "count"),
            )
            .reset_index()
        )

        return grouped, corpus_sentiment
