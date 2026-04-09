"""
Wasserstein distance computation for Narameter pipeline.

Computes distances between embedding distributions using three methods:
- Exact EMD (Earth Mover's Distance) via ot.emd2
- Sinkhorn (entropy-regularized OT) via ot.sinkhorn2
- Sliced Wasserstein Distance via ot.sliced_wasserstein_distance

References:
    - Peyré & Cuturi (2019). Computational Optimal Transport.
    - Otao & Yamada (2023). A linear time approximation of Wasserstein distance.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import ot
import pandas as pd

from narameter.config import DistanceConfig

logger = logging.getLogger(__name__)


class WassersteinCalculator:
    """
    Computes Wasserstein distances between sets of embeddings.

    Supports three methods with different speed/accuracy trade-offs:
    - "emd": Exact Earth Mover's Distance — O(n³), most accurate
    - "sinkhorn": Entropy-regularized OT — O(n²), good trade-off
    - "sliced": Sliced Wasserstein Distance — O(n log n), fastest
    """

    VALID_METHODS = {"emd", "sinkhorn", "sliced"}

    def __init__(self, config: Optional[DistanceConfig] = None):
        """
        Initialize the Wasserstein calculator.

        Args:
            config: Distance configuration. Uses defaults if None.
        """
        self.config = config or DistanceConfig()

        if self.config.method not in self.VALID_METHODS:
            raise ValueError(
                f"Invalid method '{self.config.method}'. "
                f"Choose from: {self.VALID_METHODS}"
            )

    def compute_distance(
        self,
        X_s: np.ndarray,
        X_t: np.ndarray,
        method: Optional[str] = None,
    ) -> float:
        """
        Compute Wasserstein distance between two sets of embeddings.

        Args:
            X_s: Source embeddings, shape (n_s, dim).
            X_t: Target embeddings, shape (n_t, dim).
            method: Distance method. Uses config default if None.

        Returns:
            Wasserstein distance (float, non-negative).
        """
        method = method or self.config.method

        if X_s.ndim != 2 or X_t.ndim != 2:
            raise ValueError("Embeddings must be 2D arrays (n_samples, dim)")
        if X_s.shape[1] != X_t.shape[1]:
            raise ValueError(
                f"Embedding dimensions must match: {X_s.shape[1]} vs {X_t.shape[1]}"
            )

        start_time = time.time()

        if method == "emd":
            dist = self._compute_emd(X_s, X_t)
        elif method == "sinkhorn":
            dist = self._compute_sinkhorn(X_s, X_t)
        elif method == "sliced":
            dist = self._compute_sliced(X_s, X_t)
        else:
            raise ValueError(f"Unknown method: {method}")

        elapsed = time.time() - start_time
        logger.debug(
            f"WD ({method}): {dist:.6f} "
            f"({X_s.shape[0]}×{X_t.shape[0]} samples, {elapsed:.2f}s)"
        )

        return float(dist)

    def _compute_emd(self, X_s: np.ndarray, X_t: np.ndarray) -> float:
        """Compute exact Earth Mover's Distance."""
        n_s, n_t = len(X_s), len(X_t)

        # Uniform weights
        a = np.ones(n_s) / n_s
        b = np.ones(n_t) / n_t

        # Cost matrix (squared Euclidean distance)
        M = ot.dist(X_s, X_t, metric="sqeuclidean")

        # Normalize cost matrix to avoid numerical issues
        if M.max() > 0:
            M /= M.max()

        # Compute EMD
        dist = ot.emd2(a, b, M)

        if self.config.p == 2:
            dist = np.sqrt(max(dist, 0))

        return dist

    def _compute_sinkhorn(self, X_s: np.ndarray, X_t: np.ndarray) -> float:
        """Compute Sinkhorn (entropy-regularized) distance."""
        n_s, n_t = len(X_s), len(X_t)

        # Uniform weights
        a = np.ones(n_s) / n_s
        b = np.ones(n_t) / n_t

        # Cost matrix
        M = ot.dist(X_s, X_t, metric="sqeuclidean")

        if M.max() > 0:
            M /= M.max()

        # Compute Sinkhorn distance
        dist = ot.sinkhorn2(
            a, b, M,
            reg=self.config.regularization,
            numItermax=1000,
            warn=False,
        )

        # sinkhorn2 may return an array
        if hasattr(dist, "__len__"):
            dist = dist[0]

        if self.config.p == 2:
            dist = np.sqrt(max(dist, 0))

        return dist

    def _compute_sliced(self, X_s: np.ndarray, X_t: np.ndarray) -> float:
        """Compute Sliced Wasserstein Distance."""
        dist = ot.sliced_wasserstein_distance(
            X_s, X_t,
            n_projections=self.config.n_projections,
            p=self.config.p,
            seed=self.config.seed,
        )
        return dist

    def compute_consecutive(
        self,
        period_embeddings: dict[str, np.ndarray],
        method: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Compute Wasserstein distances between consecutive time periods.

        Args:
            period_embeddings: Ordered dict mapping period labels to embeddings.
            method: Distance method. Uses config default if None.

        Returns:
            DataFrame with columns:
            ["period_a", "period_b", "distance", "n_docs_a", "n_docs_b", "method"]
        """
        labels = list(period_embeddings.keys())
        results = []

        for i in range(len(labels) - 1):
            label_a = labels[i]
            label_b = labels[i + 1]

            X_a = period_embeddings[label_a]
            X_b = period_embeddings[label_b]

            dist = self.compute_distance(X_a, X_b, method=method)

            results.append({
                "period_a": label_a,
                "period_b": label_b,
                "distance": dist,
                "n_docs_a": len(X_a),
                "n_docs_b": len(X_b),
                "method": method or self.config.method,
            })

            logger.info(
                f"{label_a} → {label_b}: WD = {dist:.6f} "
                f"({len(X_a)} vs {len(X_b)} docs)"
            )

        return pd.DataFrame(results)

    def compute_distance_matrix(
        self,
        period_embeddings: dict[str, np.ndarray],
        method: Optional[str] = None,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Compute full pairwise distance matrix between all periods.

        Args:
            period_embeddings: Dict mapping period labels to embeddings.
            method: Distance method.

        Returns:
            Tuple of (distance_matrix, labels).
            distance_matrix: Square symmetric ndarray of shape (n_periods, n_periods).
            labels: List of period labels.
        """
        labels = list(period_embeddings.keys())
        n = len(labels)
        D = np.zeros((n, n))

        total_pairs = n * (n - 1) // 2
        computed = 0

        for i in range(n):
            for j in range(i + 1, n):
                dist = self.compute_distance(
                    period_embeddings[labels[i]],
                    period_embeddings[labels[j]],
                    method=method,
                )
                D[i, j] = dist
                D[j, i] = dist
                computed += 1

                if computed % 5 == 0 or computed == total_pairs:
                    logger.info(f"Distance matrix: {computed}/{total_pairs} pairs computed")

        return D, labels

    def compare_methods(
        self,
        X_s: np.ndarray,
        X_t: np.ndarray,
    ) -> dict[str, float]:
        """
        Compute distance using all available methods for comparison.

        Useful for validating that Sliced WD approximates well.

        Args:
            X_s: Source embeddings.
            X_t: Target embeddings.

        Returns:
            Dict mapping method name to distance value.
        """
        results = {}
        for method in self.VALID_METHODS:
            try:
                dist = self.compute_distance(X_s, X_t, method=method)
                results[method] = dist
                logger.info(f"  {method}: {dist:.6f}")
            except Exception as e:
                logger.error(f"  {method}: FAILED - {e}")
                results[method] = float("nan")

        return results
