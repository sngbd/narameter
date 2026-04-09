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
6. **Detects** statistically significant narrative shifts (z-score spike detection)
7. **Visualizes** results with time-series plots, UMAP projections, and heatmaps

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
Deteksi Perubahan Narasi (z-score spike detection)
   ↓
Visualisasi (time-series, UMAP, heatmap)
```

## Configuration

Edit `config.yaml` to customize pipeline behavior. Key settings:

- **`embedding.model_name`**: SBERT model (default: `paraphrase-multilingual-MiniLM-L12-v2`)
- **`distance.method`**: `"sliced"` (fast), `"sinkhorn"` (regularized), or `"emd"` (exact)
- **`detection.zscore_threshold`**: How many standard deviations above mean to flag as shift
- **`time_period.granularity`**: `"day"`, `"week"`, or `"month"`

## References

- Reimers & Gurevych (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
- Peyré & Cuturi (2019). Computational Optimal Transport
- Colombo et al. (2021). Automatic Text Evaluation through the Lens of Wasserstein Barycenters
- Otao & Yamada (2023). A linear time approximation of Wasserstein distance

## License

MIT
