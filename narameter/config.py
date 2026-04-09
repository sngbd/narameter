"""
Configuration loader for Narameter pipeline.

Loads YAML configuration and provides typed access to settings with defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class EmbeddingConfig:
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    batch_size: int = 64
    device: str = "cpu"
    cache_dir: str = "data/processed/embeddings"


@dataclass
class DistanceConfig:
    method: str = "sliced"  # "emd", "sinkhorn", "sliced"
    n_projections: int = 500
    regularization: float = 0.05
    p: int = 2
    seed: int = 42


@dataclass
class DetectionConfig:
    method: str = "otsu"  # "zscore", "iqr", "otsu", "threshold"
    zscore_threshold: float = 2.0
    fixed_threshold: Optional[float] = None
    window_size: int = 3
    min_period_size: int = 10


@dataclass
class PreprocessingConfig:
    language: str = "id"
    remove_urls: bool = True
    remove_mentions: bool = True
    remove_hashtag_symbol: bool = True
    remove_numbers: bool = False
    remove_punctuation: bool = True
    lowercase: bool = True
    stem: bool = True
    remove_stopwords: bool = True
    min_length: int = 5
    deduplicate: bool = True


@dataclass
class TimePeriodConfig:
    granularity: str = "week"  # "day", "week", "month"
    date_format: Optional[str] = None


@dataclass
class VisualizationConfig:
    style: str = "seaborn-v0_8-whitegrid"
    dpi: int = 150
    figsize: list = field(default_factory=lambda: [15, 8])
    colormap: str = "viridis"
    format: str = "png"
    interactive: bool = True


@dataclass
class TopicConfig:
    enabled: bool = True
    n_topics: int = 0  # 0 = auto (HDBSCAN), >0 = fixed (KMeans)
    min_topic_size: int = 5
    n_top_words: int = 8
    language: str = "multilingual"
    seed: int = 42
    
    # LLM Summarization options
    representation_model: str = "keyword"  # "keyword", "openai", "gemini", "keybert"
    llm_model_name: str = "llama-3.1-8b-instant"
    llm_base_url: Optional[str] = None
    llm_prompt: str = "Summarize the following topic into a concise 3-word phrase: [DOCUMENTS]"


@dataclass
class OutputConfig:
    dir: str = "output"
    save_intermediate: bool = True


@dataclass
class NarameterConfig:
    """Main configuration container for the Narameter pipeline."""

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    distance: DistanceConfig = field(default_factory=DistanceConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    time_period: TimePeriodConfig = field(default_factory=TimePeriodConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    topic: TopicConfig = field(default_factory=TopicConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "NarameterConfig":
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        config = cls()

        if "embedding" in raw:
            config.embedding = EmbeddingConfig(**raw["embedding"])
        if "distance" in raw:
            config.distance = DistanceConfig(**raw["distance"])
        if "detection" in raw:
            config.detection = DetectionConfig(**raw["detection"])
        if "preprocessing" in raw:
            config.preprocessing = PreprocessingConfig(**raw["preprocessing"])
        if "time_period" in raw:
            config.time_period = TimePeriodConfig(**raw["time_period"])
        if "visualization" in raw:
            config.visualization = VisualizationConfig(**raw["visualization"])
        if "topic" in raw:
            config.topic = TopicConfig(**raw["topic"])
        if "output" in raw:
            config.output = OutputConfig(**raw["output"])

        return config

    @classmethod
    def default(cls) -> "NarameterConfig":
        """Return configuration with all defaults."""
        return cls()

    def ensure_dirs(self) -> None:
        """Create necessary output directories."""
        os.makedirs(self.output.dir, exist_ok=True)
        os.makedirs(self.embedding.cache_dir, exist_ok=True)
        os.makedirs("data/raw", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)


def load_config(path: Optional[str | Path] = None) -> NarameterConfig:
    """
    Load configuration from YAML file, falling back to defaults.

    Args:
        path: Path to YAML config file. If None, looks for 'config.yaml'
              in the current directory, then uses defaults.

    Returns:
        NarameterConfig instance.
    """
    if path is not None:
        return NarameterConfig.from_yaml(path)

    default_path = Path("config.yaml")
    if default_path.exists():
        return NarameterConfig.from_yaml(default_path)

    return NarameterConfig.default()
