"""
Topic modeling for Narameter pipeline using BERTopic.

Discovers sub-topics within collected tweets and tracks how topic
proportions change across time periods.  When combined with
Wasserstein-based shift detection this reveals *what* changed,
not just *that* a shift occurred.

References:
    - Grootendorst (2022). BERTopic: Neural topic modeling with a
      class-based TF-IDF procedure.  arXiv:2203.05794.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TopicModeler:
    """
    Wraps BERTopic for discovering and tracking sub-topics.

    Usage::

        modeler = TopicModeler(n_topics=10)
        topic_info, doc_topics = modeler.fit(texts, embeddings)
        proportions = modeler.get_period_proportions(doc_topics, period_labels)
        shift_df = modeler.explain_shifts(proportions, shifts_df)
    """

    def __init__(
        self,
        n_topics: Optional[int] = None,
        min_topic_size: int = 5,
        n_top_words: int = 8,
        language: str = "multilingual",
        seed: int = 42,
        representation_model: str = "keyword",
        llm_model_name: str = "llama-3.3-70b-versatile",
        llm_base_url: Optional[str] = None,
        llm_prompt: str = "Summarize the following topic into a concise 3-word phrase: [DOCUMENTS]",
    ):
        """
        Initialize the topic modeler.

        Args:
            n_topics: Target number of topics.  ``None`` = auto (HDBSCAN).
            min_topic_size: Minimum cluster size for HDBSCAN.
            n_top_words: Number of representative words per topic.
            language: Language hint for BERTopic tokenizer.
            seed: Random seed for reproducibility.
        """
        self.n_topics = n_topics
        self.min_topic_size = min_topic_size
        self.n_top_words = n_top_words
        self.language = language
        self.seed = seed
        self.representation_model = representation_model
        self.llm_model_name = llm_model_name
        self.llm_base_url = llm_base_url
        self.llm_prompt = llm_prompt

        self._model = None
        self._topic_info = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        embedding_model=None,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Fit BERTopic on the full corpus.

        We pass the *already-computed* sentence-transformer embeddings
        so BERTopic skips its own embedding step (much faster).

        Args:
            texts: Flat list of all preprocessed texts.
            embeddings: Corresponding (N, dim) embedding matrix.

        Returns:
            topic_info: DataFrame with columns [Topic, Count, Name,
                        Representation, …].
            doc_topics: 1-D array of topic IDs, one per document.
                        ``-1`` = outlier / unclassified.
        """
        from bertopic import BERTopic
        from sklearn.cluster import MiniBatchKMeans
        from hdbscan import HDBSCAN

        logger.info(
            "Fitting BERTopic on %d documents (n_topics=%s, min_topic_size=%d)",
            len(texts),
            self.n_topics or "auto",
            self.min_topic_size,
        )

        # Choose clustering model
        if self.n_topics is not None and self.n_topics > 0:
            # Fixed k → use KMeans
            cluster_model = MiniBatchKMeans(
                n_clusters=self.n_topics,
                random_state=self.seed,
            )
        else:
            # Auto → HDBSCAN
            cluster_model = HDBSCAN(
                min_cluster_size=self.min_topic_size,
                min_samples=max(1, self.min_topic_size // 2),
                metric="euclidean",
                prediction_data=True,
            )

        # Reduce dimensionality before clustering (UMAP)
        from umap import UMAP
        umap_model = UMAP(
            n_components=5,
            n_neighbors=min(15, max(2, len(texts) - 1)),
            min_dist=0.0,
            metric="cosine",
            random_state=self.seed,
        )

        # Check if we should use an LLM for representation
        rep_model = None
        if self.representation_model == "openai":
            import os
            try:
                import openai
                from bertopic.representation import OpenAI
                client_kwargs = {"api_key": os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY")}
                if self.llm_base_url:
                    client_kwargs["base_url"] = self.llm_base_url
                client = openai.OpenAI(**client_kwargs)
                rep_model = OpenAI(
                    client, 
                    model=self.llm_model_name, 
                    chat=True, 
                    prompt=self.llm_prompt,
                    doc_length=100,  # TRUNCATE TWEETS TO 100 CHARS
                    tokenizer="char", # Required when doc_length is specified
                    nr_docs=2,       # ONLY PASS TOP 2 TWEETS (default is usually 4/5)
                    delay_in_seconds=2
                )
            except ImportError:
                logger.warning("openai package not found. Using keyword representation.")
        elif self.representation_model == "gemini":
            try:
                from bertopic.representation import Google
                rep_model = Google(self.llm_model_name)
            except ImportError:
                logger.warning("google-genai package not found. Using keyword representation.")
        elif self.representation_model == "keybert":
            from bertopic.representation import KeyBERTInspired
            rep_model = KeyBERTInspired()

        model = BERTopic(
            language=self.language,
            top_n_words=self.n_top_words,
            umap_model=umap_model,
            hdbscan_model=cluster_model,
            representation_model={"LLM": rep_model} if rep_model else None,
            embedding_model=embedding_model,
            verbose=False,
        )

        # Fit — pass pre-computed embeddings
        topics, _probs = model.fit_transform(texts, embeddings=embeddings)

        self._model = model
        self._topic_info = model.get_topic_info()

        doc_topics = np.array(topics)

        n_topics_found = len(set(doc_topics)) - (1 if -1 in doc_topics else 0)
        n_outliers = int((doc_topics == -1).sum())
        logger.info(
            "BERTopic found %d topics (%d outliers / %d docs)",
            n_topics_found,
            n_outliers,
            len(texts),
        )

        return self._topic_info, doc_topics

    def get_topic_label(self, topic_id: int, prefer_llm: bool = True) -> str:
        """Return a human-readable label for a topic ID."""
        if self._model is None:
            return f"Topic {topic_id}"

        if topic_id == -1:
            return "Outlier"

        # If we have an LLM representation and we prefer it, use it! Otherwise fallback.
        topic_info = self._model.get_topic_info(topic_id)
        if prefer_llm and "LLM" in topic_info.columns:
            llm_rep = topic_info["LLM"].iloc[0]
            if isinstance(llm_rep, list):
                label_str = str(llm_rep[0])
            else:
                label_str = str(llm_rep)
        else:
            words = self._model.get_topic(topic_id)
            if not words:
                return f"Topic {topic_id}"
            label_str = str(words[0][0])
            
        label = label_str.strip(' "\'*')
        
        # If it's a short word, title case it. If it's a full phrase, just capitalize first letter.
        if " " not in label:
            return label.title()
        else:
            return label[0].upper() + label[1:]

    # ------------------------------------------------------------------
    # Period-level analysis
    # ------------------------------------------------------------------

    def get_period_proportions(
        self,
        doc_topics: np.ndarray,
        doc_periods: list[str],
        include_outliers: bool = False,
        prefer_llm: bool = True,
    ) -> pd.DataFrame:
        """
        Compute topic proportion per time period.

        Args:
            doc_topics: Topic IDs per document (from ``fit``).
            doc_periods: Period label per document (same length).
            include_outliers: Whether to count topic -1.

        Returns:
            DataFrame with periods as rows and topics as columns.
            Values are proportions (0–1) summing to 1 per row.
        """
        assert len(doc_topics) == len(doc_periods), (
            f"Length mismatch: {len(doc_topics)} topics vs {len(doc_periods)} periods"
        )

        records = []
        for period in sorted(set(doc_periods)):
            mask = [p == period for p in doc_periods]
            period_topics = doc_topics[mask]

            if not include_outliers:
                period_topics = period_topics[period_topics != -1]

            if len(period_topics) == 0:
                continue

            counts = Counter(period_topics)
            total = sum(counts.values())
            record = {"period": period}
            for tid, cnt in counts.items():
                label = self.get_topic_label(tid, prefer_llm=prefer_llm)
                record[label] = cnt / total
            records.append(record)

        df = pd.DataFrame(records).fillna(0.0)
        df = df.set_index("period")

        # Sort columns by mean proportion descending
        col_means = df.mean().sort_values(ascending=False)
        df = df[col_means.index]

        return df

    def explain_shifts(
        self,
        proportions: pd.DataFrame,
        results: pd.DataFrame,
        top_n: int = 3,
        output_suffix: str = "",
    ) -> pd.DataFrame:
        """
        For each detected shift explain which topics drove the change.

        Args:
            proportions: Period × topic proportion matrix.
            results: Detection results (with ``period_a``, ``period_b``,
                     ``is_shift`` columns).
            top_n: Number of top changing topics to report.

        Returns:
            Copy of *results* with added columns:
            ``topic_changes``  — JSON string of top topic deltas
            ``shift_summary``  — human-readable one-liner
        """
        explanations = []
        summaries = []

        for _, row in results.iterrows():
            pa, pb = row["period_a"], row["period_b"]

            if pa not in proportions.index or pb not in proportions.index:
                explanations.append("")
                summaries.append("")
                continue

            prop_a = proportions.loc[pa]
            prop_b = proportions.loc[pb]

            # Compute deltas
            all_topics = set(prop_a.index) | set(prop_b.index)
            deltas = {}
            for t in all_topics:
                va = prop_a.get(t, 0.0)
                vb = prop_b.get(t, 0.0)
                deltas[t] = round(vb - va, 4)

            # Sort by absolute change
            sorted_deltas = sorted(
                deltas.items(), key=lambda x: abs(x[1]), reverse=True
            )[:top_n]

            explanations.append(json.dumps(dict(sorted_deltas), ensure_ascii=False))

            # Build summary
            parts = []
            for topic, delta in sorted_deltas:
                if abs(delta) < 0.01:
                    continue
                direction = "↑" if delta > 0 else "↓"
                parts.append(f"{topic} {direction}{abs(delta)*100:.0f}%")

            summaries.append("; ".join(parts) if parts else "stable")

        out = results.copy()
        out[f"topic_changes{output_suffix}"] = explanations
        out[f"shift_summary{output_suffix}"] = summaries
        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the fitted BERTopic model."""
        if self._model is not None:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self._model.save(path)
            logger.info("Topic model saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "TopicModeler":
        """Load a previously saved BERTopic model."""
        from bertopic import BERTopic

        instance = cls()
        instance._model = BERTopic.load(path)
        instance._topic_info = instance._model.get_topic_info()
        return instance
