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
6. **Detects** statistically significant narrative shifts using **Otsu's Thresholding Methodology** 
7. **Extracts & Summarizes** topics from shifted periods using BERTopic and **Llama 3** Generative AI via a zero-cost Groq pipeline.
8. **Visualizes** results with dynamically formatted time-series plots, UMAP projections, and heatmaps.

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
| `narameter embed` | Generate embeddings only |
| `narameter distance` | Compute Wasserstein distances |
| `narameter detect` | Detect narrative shifts |
| `narameter visualize` | Generate visualizations |
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
Deteksi Perubahan Narasi (Otsu's Method Algorithm)
   ↓
Identifikasi Topik (BERTopic / c-TF-IDF / Llama 3 API)
   ↓
Visualisasi & Laporan Dinamis (auto-timestamped directories)
```

## Configuration

Edit `config.yaml` to customize pipeline behavior. Key settings:

- **`embedding.model_name`**: SBERT model (default: `paraphrase-multilingual-MiniLM-L12-v2`)
- **`distance.method`**: `"sliced"` (fast), `"sinkhorn"` (regularized), or `"emd"` (exact)
- **`detection.method`**: `"otsu"` (variance maximization) or `"zscore"`
- **`time_period.granularity`**: `"day"`, `"week"`, or `"month"`
- **`topic.llm_model_name`**: The LLM parsing topic geometries. Defaults safely to **`llama-3.1-8b-instant`** over Groq APIs to circumvent free-tier token limitations. 

💡 **Note on Environment Variables:** You can drop a `.env` file directly directly into the root folder containing `GROQ_API_KEY=gsk_...` or `OPENAI_API_KEY=sk_...`. The pipeline natively loads `python-dotenv` prior to execution, so `source .env` isn't required.

## Outputs
Data exports securely route to dynamically timestamped directories (e.g. `output/[filename]_[timestamp]/`) containing visualization PNGs alongside dual-layered topic matrices:
*   `results.csv` (Annotated shifts over LLM titles)
*   `topic_proportions.csv` 
*   `topic_proportions_keywords.csv` (Provides the underlying c-TF-IDF raw keywords for reality-checking the LLM translations).

## References

- Reimers & Gurevych (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
- Peyré & Cuturi (2019). Computational Optimal Transport
- Colombo et al. (2021). Automatic Text Evaluation through the Lens of Wasserstein Barycenters
- Otao & Yamada (2023). A linear time approximation of Wasserstein distance

## License

MIT
