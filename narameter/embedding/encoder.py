"""
Sentence embedding encoder for Narameter pipeline.

Wraps sentence-transformers for batch encoding with caching support.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

from narameter.config import EmbeddingConfig

logger = logging.getLogger(__name__)


class SentenceEncoder:
    """
    Encodes texts into sentence embeddings using Sentence-BERT models.

    Supports multilingual models for Indonesian text.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize the encoder.

        Args:
            config: Embedding configuration. Uses defaults if None.
        """
        self.config = config or EmbeddingConfig()
        self._model = None
        self._embedding_dim = None

    @property
    def model(self):
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading model: {self.config.model_name}")
            logger.info(f"Device: {self.config.device}")

            self._model = SentenceTransformer(
                self.config.model_name,
                device=self.config.device,
            )
            self._embedding_dim = self._model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Embedding dimension: {self._embedding_dim}")

        return self._model

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension of the loaded model."""
        if self._embedding_dim is None:
            _ = self.model  # Force load
        return self._embedding_dim

    def encode(
        self,
        texts: list[str],
        batch_size: Optional[int] = None,
        show_progress: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of texts into embeddings.

        Args:
            texts: List of text strings to encode.
            batch_size: Batch size for encoding. Uses config default if None.
            show_progress: Whether to show a progress bar.
            normalize: Whether to L2-normalize embeddings.

        Returns:
            numpy array of shape (n_texts, embedding_dim).
        """
        if not texts:
            return np.array([])

        batch_size = batch_size or self.config.batch_size

        logger.info(f"Encoding {len(texts)} texts (batch_size={batch_size})")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )

        logger.info(f"Encoded shape: {embeddings.shape}")
        return embeddings

    def encode_periods(
        self,
        period_texts: dict[str, list[str]],
        cache: bool = True,
        cache_dir: Optional[str] = None,
    ) -> dict[str, np.ndarray]:
        """
        Encode texts grouped by time period, with optional caching.

        Args:
            period_texts: Dictionary mapping period labels to lists of texts.
            cache: Whether to cache embeddings to disk.
            cache_dir: Directory for cached embeddings.

        Returns:
            Dictionary mapping period labels to embedding arrays.
        """
        cache_dir = cache_dir or self.config.cache_dir
        period_embeddings = {}

        for period_label, texts in period_texts.items():
            if not texts:
                logger.warning(f"Period '{period_label}' has no texts, skipping")
                continue

            # Try loading from cache
            if cache:
                cached = self._load_cached(period_label, texts, cache_dir)
                if cached is not None:
                    period_embeddings[period_label] = cached
                    logger.info(
                        f"Period '{period_label}': loaded {len(cached)} cached embeddings"
                    )
                    continue

            # Encode
            logger.info(
                f"Period '{period_label}': encoding {len(texts)} texts"
            )
            embeddings = self.encode(texts, show_progress=False)
            period_embeddings[period_label] = embeddings

            # Save to cache
            if cache:
                self._save_cache(period_label, texts, embeddings, cache_dir)

        return period_embeddings

    def _cache_key(self, period_label: str, texts: list[str]) -> str:
        """Generate a cache key based on period label and content hash."""
        content_hash = hashlib.md5(
            "\n".join(texts).encode("utf-8")
        ).hexdigest()[:12]
        safe_label = period_label.replace(" ", "_").replace("/", "-")
        return f"{safe_label}_{content_hash}"

    def _cache_path(self, period_label: str, texts: list[str], cache_dir: str) -> Path:
        """Get the cache file path for a period."""
        key = self._cache_key(period_label, texts)
        return Path(cache_dir) / f"{key}.npy"

    def _load_cached(
        self, period_label: str, texts: list[str], cache_dir: str
    ) -> Optional[np.ndarray]:
        """Load cached embeddings if available."""
        path = self._cache_path(period_label, texts, cache_dir)
        if path.exists():
            try:
                return np.load(path)
            except Exception as e:
                logger.warning(f"Failed to load cache {path}: {e}")
        return None

    def _save_cache(
        self,
        period_label: str,
        texts: list[str],
        embeddings: np.ndarray,
        cache_dir: str,
    ) -> None:
        """Save embeddings to cache."""
        os.makedirs(cache_dir, exist_ok=True)
        path = self._cache_path(period_label, texts, cache_dir)
        try:
            np.save(path, embeddings)
            logger.debug(f"Cached embeddings to {path}")
        except Exception as e:
            logger.warning(f"Failed to save cache {path}: {e}")

    def clear_cache(self, cache_dir: Optional[str] = None) -> int:
        """
        Clear all cached embeddings.

        Args:
            cache_dir: Cache directory. Uses config default if None.

        Returns:
            Number of files removed.
        """
        cache_dir = cache_dir or self.config.cache_dir
        cache_path = Path(cache_dir)
        if not cache_path.exists():
            return 0

        files = list(cache_path.glob("*.npy"))
        for f in files:
            f.unlink()

        logger.info(f"Cleared {len(files)} cached embeddings")
        return len(files)
