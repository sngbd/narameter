# 📊 Narameter

**Deteksi Perubahan Narasi Publik Menggunakan Distribusi Embedding Teks dan Wasserstein Distance**

*Public Narrative Shift Detection using Text Embedding Distributions and Wasserstein Distance*

---

## Overview

Narameter is a Python pipeline that detects shifts in public narratives over time. It:

1. **Collects** text data from Twitter/X or CSV/JSON files
2. **Preprocesses** Indonesian text (cleaning, stemming, stopword removal)
3. **Encodes** texts into sentence embeddings using multilingual Sentence-BERT
4. **Constructs** empirical distributions per time period
5. **Computes** Wasserstein Distance between consecutive period distributions
6. **Detects** statistically significant narrative shifts using **Otsu's Thresholding Methodology** or the **PELT (Pruned Exact Linear Time)** statistical algorithm.
7. **Extracts & Summarizes** topics from shifted periods using BERTopic and **Generative AI** (Gemini, Llama 3 via Groq, OpenAI) or local phrasing (KeyBERT).
8. **Analyzes Sentiment & Entities** by calculating polarity `[-1, +1]` scores per topic and running Named Entity Recognition (NER) to track actors driving the narrative.
9. **Visualizes** results with dynamically formatted multi-axis time-series plots, UMAP projections, distance matrices, and entity prominence heatmaps.

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd narameter

# Install in development mode
pip install -e ".[dev]"
```

## Quick Start

```bash
# Run full pipeline on a CSV file
narameter run --input data.csv --text-col "text" --date-col "date"

# With custom config
narameter run --input data.csv --text-col "text" --date-col "date" --config config.yaml

# Generate sample data and run
narameter sample --output data/sample/synthetic.csv
narameter run --input data/sample/synthetic.csv --text-col "text" --date-col "timestamp"
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `narameter run` | Run the full pipeline |
| `narameter scrape` | Scrape Twitter/X data |
| `narameter scrape-run` | Scrape Twitter/X data and immediately run pipeline |
| `narameter embed` | Generate embeddings only |
| `narameter sample` | Generate synthetic sample data |

## Architecture

```
Data Teks (Twitter/X & CSV/JSON)
   ↓
Preprocessing (cleaning, tokenisasi, normalisasi)
   ↓
Sentence Embedding (Sentence-BERT multilingual)
   ↓
Distribusi Embedding per Periode Waktu
   ↓
Wasserstein Distance (EMD / Sinkhorn / Sliced)
   ↓
Deteksi Perubahan Narasi (Otsu's Method algorithm / PELT algorithm)
   ↓
Identifikasi Topik & Anotasi GenAI (BERTopic / c-TF-IDF / Gemini / Llama 3 / OpenAI)
   ↓
Sentiment Tracking & Named Entity/Actor Extraction (multilingual BERT / NER)
   ↓
Visualisasi & Laporan Dinamis (auto-timestamped directories, dual-axis timelines, heatmaps)
```

## Configuration

Edit `config.yaml` to customize pipeline behavior. Key settings:

- **`embedding.model_name`**: SBERT model (default: `paraphrase-multilingual-MiniLM-L12-v2`)
- **`distance.method`**: `"sliced"` (fast), `"sinkhorn"` (regularized), or `"emd"` (exact)
- **`detection.method`**: `"otsu"` (variance maximization), `"zscore"`, or `"pelt"` (change point detection, supports `pelt_penalty` param)
- **`time_period.granularity`**: `"day"`, `"week"`, or `"month"`
- **`topic.representation_model`** (in `fast_config.yaml`): The model for parsing topic geometries. Supports `"gemini"`, `"openai"`, `"keybert"`, or Groq APIs (Llama 3) to circumvent free-tier token limitations. 

💡 **Note on Environment Variables:** You can drop a `.env` file directly into the root folder containing `GOOGLE_API_KEY=AIza...`, `GROQ_API_KEY=gsk_...`, or `OPENAI_API_KEY=sk_...`. The pipeline natively loads `python-dotenv` prior to execution, so `source .env` isn't required.

## Outputs
Data exports securely route to dynamically timestamped directories (e.g. `output/[filename]_[timestamp]/`) containing visualization PNGs alongside dual-layered topic matrices:
*   `results.csv` (Annotated shifts over LLM titles, including tracking of top detected entities)
*   `topic_proportions.csv` and `topic_proportions_keywords.csv` (Provides the underlying c-TF-IDF raw keywords for reality-checking the LLM translations).
*   `sentiment_by_topic.csv` & `corpus_sentiment.csv` (Topic and corpus level sentiment polarity mapped to `[-1, 1]`)
*   `entity_tracking.csv` (Named entity recognition frequency lists per period)

## References

- Reimers & Gurevych (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
- Peyré & Cuturi (2019). Computational Optimal Transport
- Colombo et al. (2021). Automatic Text Evaluation through the Lens of Wasserstein Barycenters
- Otao & Yamada (2023). A linear time approximation of Wasserstein distance

## License

MIT
