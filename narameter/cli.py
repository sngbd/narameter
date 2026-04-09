"""
CLI entry point for Narameter pipeline.

Commands:
    narameter run        — Run the full pipeline
    narameter scrape     — Scrape Twitter/X data
    narameter scrape-run — Scrape + run pipeline in one step
    narameter embed      — Generate embeddings only
    narameter sample     — Generate synthetic sample data
"""

from __future__ import annotations

import logging
import sys

import click
from rich.console import Console
from rich.logging import RichHandler

console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=False, show_path=False)],
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(verbose: bool) -> None:
    """
    Narameter — Narrative Shift Detection Pipeline.

    Deteksi Perubahan Narasi Publik Menggunakan Distribusi Embedding Teks
    dan Wasserstein Distance.
    """
    setup_logging(verbose)


# ─── run ────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--input", "-i", "input_path", required=True, help="Input data file (CSV/JSON)")
@click.option("--text-col", "-t", default="text", help="Text column name")
@click.option("--date-col", "-d", default="date", help="Date column name")
@click.option("--output", "-o", "output_dir", default="output", help="Output directory")
@click.option("--config", "-c", "config_path", default=None, help="Config YAML file")
def run(
    input_path: str,
    text_col: str,
    date_col: str,
    output_dir: str,
    config_path: str,
) -> None:
    """Run the full narrative shift detection pipeline."""
    from narameter.pipeline import NarameterPipeline

    pipeline = NarameterPipeline(config_path=config_path)
    pipeline.run(
        input_path=input_path,
        text_col=text_col,
        date_col=date_col,
        output_dir=output_dir,
    )


# ─── scraper helpers ────────────────────────────────────────────────────────

def _init_scraper(auth_token, cookies_file):
    """Initialize TwitterScraper from auth_token or cookies file."""
    from narameter.data.twitter_scraper import TwitterScraper

    # Priority: --auth-token > --cookies > saved cookies
    if auth_token:
        console.print("[green]✓ Using provided auth_token.[/green]")
        return TwitterScraper(auth_token=auth_token)

    if cookies_file:
        console.print(f"[yellow]Loading cookies from {cookies_file}...[/yellow]")
        scraper = TwitterScraper.from_cookies(cookies_file)
        console.print("[green]✓ Cookies loaded.[/green]")
        return scraper

    scraper = TwitterScraper.from_default_cookies()
    if scraper:
        console.print("[green]✓ Loaded saved session cookies.[/green]")
        return scraper

    console.print(
        "[red]No auth method available.[/red]\n\n"
        "[cyan]Option 1 — auth_token (easiest):[/cyan]\n"
        "  1. Log in to x.com → F12 → Application → Cookies → copy auth_token\n"
        "  2. narameter scrape -q QUERY --auth-token YOUR_TOKEN\n\n"
        "[cyan]Option 2 — cookies file:[/cyan]\n"
        "  1. Install 'Cookie-Editor' browser extension\n"
        "  2. Log in to x.com → click extension → Export → save as cookies.json\n"
        "  3. narameter scrape -q QUERY --cookies cookies.json\n"
    )
    return None


# ─── scrape ─────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--query", "-q", required=True, help="Twitter search query/keyword")
@click.option("--auth-token", default=None, help="auth_token cookie from x.com")
@click.option("--cookies", "cookies_file", default=None, help="Path to cookies JSON (browser export)")
@click.option("--max-results", "-n", default=500, help="Max tweets to collect")
@click.option("--start-date", default=None, help="Start date (YYYY-MM-DD)")
@click.option("--end-date", default=None, help="End date (YYYY-MM-DD)")
@click.option("--output", "-o", default="data/raw", help="Output directory")
def scrape(
    query: str,
    auth_token: str,
    cookies_file: str,
    max_results: int,
    start_date: str,
    end_date: str,
    output: str,
) -> None:
    """Scrape tweets from Twitter/X and save to JSON.

    \b
    Auth (pick one):
      --auth-token TOKEN    (from F12 → Application → Cookies → auth_token)
      --cookies FILE        (browser export via Cookie-Editor)
      (auto)                uses saved data/raw/twitter_cookies.json
    """
    import os
    from datetime import datetime

    scraper = _init_scraper(auth_token, cookies_file)
    if not scraper:
        return

    df = scraper.scrape_topic(
        topic=query,
        start_date=start_date,
        end_date=end_date,
        max_results=max_results,
    )

    if df.empty:
        console.print("[red]No tweets found for the query.[/red]")
        return

    os.makedirs(output, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = query.replace(" ", "_")[:30]
    filepath = os.path.join(output, f"twitter_{safe_query}_{ts}.json")
    df.to_json(filepath, orient="records", date_format="iso", indent=2)

    console.print(f"[green]✅ Scraped {len(df)} tweets → {filepath}[/green]")
    console.print(f"   Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    console.print(f"\n[cyan]To run the pipeline:[/cyan]")
    console.print(f"   narameter run -i {filepath} -t text -d timestamp")


# ─── scrape-run ─────────────────────────────────────────────────────────────

@cli.command("scrape-run")
@click.option("--query", "-q", required=True, help="Twitter search query/keyword")
@click.option("--auth-token", default=None, help="auth_token cookie from x.com")
@click.option("--cookies", "cookies_file", default=None, help="Path to cookies JSON (browser export)")
@click.option("--max-results", "-n", default=500, help="Max tweets to collect")
@click.option("--start-date", default=None, help="Start date (YYYY-MM-DD)")
@click.option("--end-date", default=None, help="End date (YYYY-MM-DD)")
@click.option("--output", "-o", "output_dir", default="output", help="Output directory for results")
@click.option("--config", "-c", "config_path", default=None, help="Config YAML file")
def scrape_run(
    query: str,
    auth_token: str,
    cookies_file: str,
    max_results: int,
    start_date: str,
    end_date: str,
    output_dir: str,
    config_path: str,
) -> None:
    """Scrape tweets, then run the full narrative shift detection pipeline.

    \b
    Auth (pick one):
      --auth-token TOKEN    (from F12 → Application → Cookies → auth_token)
      --cookies FILE        (browser export via Cookie-Editor)
      (auto)                uses saved data/raw/twitter_cookies.json
    """
    import os
    from datetime import datetime
    from narameter.pipeline import NarameterPipeline

    scraper = _init_scraper(auth_token, cookies_file)
    if not scraper:
        return

    # Step 1: Scrape
    console.print(f"[bold]🐦 Scraping tweets for: '{query}'[/bold]")
    df = scraper.scrape_topic(
        topic=query,
        start_date=start_date,
        end_date=end_date,
        max_results=max_results,
    )

    if df.empty:
        console.print("[red]No tweets found. Cannot run pipeline.[/red]")
        return

    # Save raw data
    os.makedirs("data/raw", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = query.replace(" ", "_")[:30]
    raw_path = f"data/raw/twitter_{safe_query}_{ts}.json"
    df.to_json(raw_path, orient="records", date_format="iso", indent=2)
    console.print(f"[green]Scraped {len(df)} tweets → {raw_path}[/green]")

    # Step 2: Run pipeline
    console.print(f"\n[bold]🚀 Running narrative shift detection...[/bold]")
    pipeline = NarameterPipeline(config_path=config_path)
    results = pipeline.run_from_dataframe(df, output_dir=output_dir)

    n_shifts = results["is_shift"].sum() if "is_shift" in results.columns else 0
    console.print(f"\n[bold green]Done! {n_shifts} narrative shifts detected.[/bold green]")


# ─── embed ──────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--input", "-i", "input_path", required=True, help="Input data file")
@click.option("--text-col", "-t", default="text", help="Text column name")
@click.option("--date-col", "-d", default="date", help="Date column name")
@click.option("--output", "-o", default="data/processed/embeddings", help="Output directory")
@click.option("--config", "-c", "config_path", default=None, help="Config YAML file")
def embed(
    input_path: str,
    text_col: str,
    date_col: str,
    output: str,
    config_path: str,
) -> None:
    """Generate embeddings from input data."""
    import numpy as np
    from narameter.config import load_config
    from narameter.data.loader import DataLoader
    from narameter.data.preprocessor import TextPreprocessor
    from narameter.embedding.encoder import SentenceEncoder

    config = load_config(config_path)

    loader = DataLoader()
    df = loader.load_auto(input_path, text_col=text_col, date_col=date_col)

    preprocessor = TextPreprocessor(config.preprocessing)
    df = preprocessor.preprocess_dataframe(df)

    encoder = SentenceEncoder(config.embedding)
    texts = df["text_clean"].tolist() if "text_clean" in df.columns else df["text"].tolist()
    embeddings = encoder.encode(texts)

    import os
    os.makedirs(output, exist_ok=True)
    output_path = os.path.join(output, "embeddings.npy")
    np.save(output_path, embeddings)
    console.print(f"[green]Embeddings saved to {output_path} — shape: {embeddings.shape}[/green]")


# ─── sample ─────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--output", "-o", default="data/sample/synthetic.csv", help="Output CSV path")
@click.option("--periods", "-p", default=6, help="Number of time periods")
@click.option("--docs-per-period", "-n", default=50, help="Documents per period")
def sample(output: str, periods: int, docs_per_period: int) -> None:
    """Generate synthetic sample data for testing."""
    sys.path.insert(0, ".")

    from data.sample.sample_generator import generate_synthetic_data

    df = generate_synthetic_data(
        n_periods=periods,
        docs_per_period=docs_per_period,
        output_path=output,
    )
    console.print(f"[green]Generated {len(df)} synthetic records → {output}[/green]")


if __name__ == "__main__":
    cli()
