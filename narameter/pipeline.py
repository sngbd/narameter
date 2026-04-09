"""
End-to-end pipeline orchestrator for Narameter.

Chains all components: load → preprocess → embed → distance → detect → visualize.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from narameter.config import NarameterConfig, load_config
from narameter.data.loader import DataLoader
from narameter.data.preprocessor import TextPreprocessor
from narameter.detection.detector import NarrativeShiftDetector
from narameter.distance.wasserstein import WassersteinCalculator
from narameter.embedding.encoder import SentenceEncoder
from narameter.topic.modeler import TopicModeler
from narameter.visualization.plots import NarrativeVisualizer

logger = logging.getLogger(__name__)
console = Console()


class NarameterPipeline:
    """
    End-to-end pipeline for narrative shift detection.

    Usage:
        pipeline = NarameterPipeline("config.yaml")
        results = pipeline.run("data.csv", text_col="text", date_col="date")
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the pipeline.

        Args:
            config_path: Path to YAML config file. Uses defaults if None.
        """
        self.config = load_config(config_path)
        self.config.ensure_dirs()

        # Initialize components
        self.loader = DataLoader()
        self.preprocessor = TextPreprocessor(self.config.preprocessing)
        self.encoder = SentenceEncoder(self.config.embedding)
        self.calculator = WassersteinCalculator(self.config.distance)
        self.detector = NarrativeShiftDetector(self.config.detection)
        self.visualizer = NarrativeVisualizer(self.config.visualization)
        
        self.topic_modeler = None
        if getattr(self.config, "topic", None) and self.config.topic.enabled:
            topic_kwargs = {k: v for k, v in self.config.topic.__dict__.items() if k != "enabled"}
            self.topic_modeler = TopicModeler(**topic_kwargs)

    def run(
        self,
        input_path: str,
        text_col: str = "text",
        date_col: str = "date",
        output_dir: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Run the full pipeline from data loading to visualization.

        Args:
            input_path: Path to input data (CSV, JSON, or directory).
            text_col: Column name for text data.
            date_col: Column name for timestamps.
            output_dir: Output directory for results and plots.

        Returns:
            Detection results DataFrame.
        """
        import datetime
        base_dir = output_dir or self.config.output.dir
        input_basename = os.path.splitext(os.path.basename(input_path))[0]
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_dir, f"{timestamp}_{input_basename}")
        
        os.makedirs(output_dir, exist_ok=True)

        start_time = time.time()

        console.print(Panel(
            "[bold cyan]Narameter — Narrative Shift Detection Pipeline[/bold cyan]\n"
            "[dim]Deteksi Perubahan Narasi Publik[/dim]",
            border_style="cyan",
        ))

        # Step 1: Load data
        console.print("\n[bold]📥 Step 1/6: Loading Data[/bold]")
        df = self.loader.load_auto(input_path, text_col=text_col, date_col=date_col)
        console.print(f"  Loaded [green]{len(df)}[/green] documents")
        console.print(f"  Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")

        # Step 2: Preprocess
        console.print("\n[bold]🧹 Step 2/6: Preprocessing[/bold]")
        df = self.preprocessor.preprocess_dataframe(df, text_col="text", output_col="text_clean")
        console.print(f"  After preprocessing: [green]{len(df)}[/green] documents")

        # Step 3: Group by time period
        console.print("\n[bold]📅 Step 3/6: Grouping by Time Period[/bold]")
        period_texts = self._group_by_period(df)
        self._print_period_summary(period_texts)

        # Filter periods with too few documents
        min_size = self.config.detection.min_period_size
        period_texts = {
            k: v for k, v in period_texts.items()
            if len(v) >= min_size
        }

        if len(period_texts) < 2:
            console.print(
                f"[red]Error: Need at least 2 periods with ≥{min_size} documents.[/red]"
            )
            console.print(
                f"[yellow]Insufficient data: only {len(period_texts)} valid periods found. "
                f"Try scraping more data or changing granularity.[/yellow]"
            )
            return pd.DataFrame()

        # Step 4: Encode embeddings
        console.print("\n[bold]🧠 Step 4/6: Generating Embeddings[/bold]")
        console.print(f"  Model: [cyan]{self.config.embedding.model_name}[/cyan]")
        period_embeddings = self.encoder.encode_periods(period_texts)

        total_embeddings = sum(len(v) for v in period_embeddings.values())
        dim = next(iter(period_embeddings.values())).shape[1]
        console.print(f"  Encoded [green]{total_embeddings}[/green] texts → {dim}D embeddings")

        # Step 5: Compute Wasserstein distances
        console.print("\n[bold]📏 Step 5/6: Computing Wasserstein Distances[/bold]")
        console.print(f"  Method: [cyan]{self.config.distance.method}[/cyan]")

        # Consecutive distances
        distances_df = self.calculator.compute_consecutive(period_embeddings)

        # Full distance matrix (for heatmap)
        distance_matrix, matrix_labels = self.calculator.compute_distance_matrix(
            period_embeddings
        )

        self._print_distance_table(distances_df)

        # Step 6: Detect narrative shifts
        console.print("\n[bold]⚡ Step 6/6: Detecting Narrative Shifts[/bold]")
        results = self.detector.detect(distances_df)
        
        # Topic Modeling
        if self.topic_modeler is not None:
            console.print("\n[bold]🧠 Topic Modeling (BERTopic)[/bold]")
            
            # Flatten texts and embeddings
            flat_texts = []
            flat_embs = []
            flat_periods = []
            for period, txts in period_texts.items():
                flat_texts.extend(txts)
                flat_embs.append(period_embeddings[period])
                flat_periods.extend([period] * len(txts))
                
            X_emb = np.vstack(flat_embs)
            
            # Fit topic model
            topic_info, doc_topics = self.topic_modeler.fit(
                flat_texts, X_emb, embedding_model=self.encoder.model
            )
            
            # Get proportions and explain shifts
            proportions = self.topic_modeler.get_period_proportions(doc_topics, flat_periods, prefer_llm=True)
            results = self.topic_modeler.explain_shifts(proportions, results)
            proportions.to_csv(os.path.join(output_dir, "topic_proportions.csv"))
            
            # If an LLM was used, also save separated non-LLM (keyword) versions for reality check
            if "LLM" in topic_info.columns:
                proportions_keywords = self.topic_modeler.get_period_proportions(doc_topics, flat_periods, prefer_llm=False)
                results = self.topic_modeler.explain_shifts(proportions_keywords, results, output_suffix="_keywords")
                proportions_keywords.to_csv(os.path.join(output_dir, "topic_proportions_keywords.csv"))
            
            # Save topic info
            topic_info.to_csv(os.path.join(output_dir, "topic_info.csv"), index=False)
            
            console.print(f"  Found [green]{len(topic_info)}[/green] topics")

        summary = self.detector.summary(results)

        self._print_detection_results(results, summary)

        # Generate visualizations
        console.print("\n[bold]📊 Generating Visualizations[/bold]")
        generated_files = self.visualizer.generate_report(
            results=results,
            period_embeddings=period_embeddings,
            output_dir=output_dir,
            distance_matrix=distance_matrix,
            distance_labels=matrix_labels,
        )

        for f in generated_files:
            console.print(f"  📄 {f}")

        # Save results
        results_path = os.path.join(output_dir, "results.csv")
        results.to_csv(results_path, index=False)

        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        elapsed = time.time() - start_time

        console.print(Panel(
            f"[bold green]✅ Pipeline Complete[/bold green]\n"
            f"Total time: {elapsed:.1f}s\n"
            f"Results saved to: {output_dir}/\n"
            f"Shifts detected: {summary['total_shifts']}/{summary['total_transitions']}",
            border_style="green",
        ))

        return results

    def run_from_dataframe(
        self,
        df: pd.DataFrame,
        output_dir: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Run pipeline from an already-loaded DataFrame.

        DataFrame must have columns: ["text", "timestamp"].
        """
        output_dir = output_dir or self.config.output.dir
        os.makedirs(output_dir, exist_ok=True)

        # Preprocess
        df = self.preprocessor.preprocess_dataframe(df)

        # Group by period
        period_texts = self._group_by_period(df)

        # Filter small periods
        min_size = self.config.detection.min_period_size
        period_texts = {
            k: v for k, v in period_texts.items()
            if len(v) >= min_size
        }

        if len(period_texts) < 2:
            console.print(
                f"[red]Error: Need at least 2 periods with ≥{min_size} documents.[/red]"
            )
            console.print(
                f"[yellow]Insufficient data: only {len(period_texts)} valid periods found. "
                f"Try scraping more data or changing granularity.[/yellow]"
            )
            return pd.DataFrame()

        # Encode
        period_embeddings = self.encoder.encode_periods(period_texts)

        # Compute distances
        distances_df = self.calculator.compute_consecutive(period_embeddings)
        distance_matrix, matrix_labels = self.calculator.compute_distance_matrix(
            period_embeddings
        )

        # Detect shifts
        results = self.detector.detect(distances_df)

        # Topic Modeling
        if self.topic_modeler is not None:
            console.print("\n[bold]🧠 Topic Modeling (BERTopic)[/bold]")
            
            # Flatten texts and embeddings
            flat_texts = []
            flat_embs = []
            flat_periods = []
            for period, txts in period_texts.items():
                flat_texts.extend(txts)
                flat_embs.append(period_embeddings[period])
                flat_periods.extend([period] * len(txts))
                
            X_emb = np.vstack(flat_embs)
            
            # Fit topic model
            topic_info, doc_topics = self.topic_modeler.fit(
                flat_texts, X_emb, embedding_model=self.encoder.model
            )
            
            # Get proportions and explain shifts
            proportions = self.topic_modeler.get_period_proportions(doc_topics, flat_periods, prefer_llm=True)
            results = self.topic_modeler.explain_shifts(proportions, results)
            proportions.to_csv(os.path.join(output_dir, "topic_proportions.csv"))
            
            # If an LLM was used, also save separated non-LLM (keyword) versions for reality check
            if "LLM" in topic_info.columns:
                proportions_keywords = self.topic_modeler.get_period_proportions(doc_topics, flat_periods, prefer_llm=False)
                results = self.topic_modeler.explain_shifts(proportions_keywords, results, output_suffix="_keywords")
                proportions_keywords.to_csv(os.path.join(output_dir, "topic_proportions_keywords.csv"))
            
            # Save topic info
            topic_info.to_csv(os.path.join(output_dir, "topic_info.csv"), index=False)
            
            console.print(f"  Found [green]{len(topic_info)}[/green] topics")

        # Visualize
        self.visualizer.generate_report(
            results, period_embeddings, output_dir,
            distance_matrix, matrix_labels,
        )

        # Save
        results.to_csv(os.path.join(output_dir, "results.csv"), index=False)

        return results

    def _group_by_period(self, df: pd.DataFrame) -> dict[str, list[str]]:
        """Group texts by time period based on configuration."""
        granularity = self.config.time_period.granularity
        text_col = "text_clean" if "text_clean" in df.columns else "text"

        if granularity == "day":
            df["period"] = df["timestamp"].dt.strftime("%Y-%m-%d")
        elif granularity in ["week", "month"]:
            # Drop timezone info to avoid pandas UserWarning before converting to period
            ts_no_tz = df["timestamp"].dt.tz_localize(None)
            if granularity == "week":
                df["period"] = ts_no_tz.dt.to_period("W").astype(str)
            else:
                df["period"] = ts_no_tz.dt.to_period("M").astype(str)
        else:
            raise ValueError(f"Unknown granularity: {granularity}")

        # Build ordered dict from sorted periods
        period_texts = {}
        for period in sorted(df["period"].unique()):
            texts = df[df["period"] == period][text_col].tolist()
            period_texts[period] = texts

        return period_texts

    def _print_period_summary(self, period_texts: dict[str, list[str]]) -> None:
        """Print a summary table of periods."""
        table = Table(title="Time Periods", show_header=True, header_style="bold cyan")
        table.add_column("Period", style="cyan")
        table.add_column("Documents", justify="right", style="green")

        for period, texts in period_texts.items():
            table.add_row(period, str(len(texts)))

        table.add_row(
            "[bold]Total[/bold]",
            f"[bold]{sum(len(v) for v in period_texts.values())}[/bold]",
        )
        console.print(table)

    def _print_distance_table(self, distances_df: pd.DataFrame) -> None:
        """Print distance results as a table."""
        table = Table(title="Wasserstein Distances", show_header=True, header_style="bold cyan")
        table.add_column("Transition", style="cyan")
        table.add_column("Distance", justify="right", style="yellow")
        table.add_column("Docs A", justify="right")
        table.add_column("Docs B", justify="right")

        for _, row in distances_df.iterrows():
            table.add_row(
                f"{row['period_a']} → {row['period_b']}",
                f"{row['distance']:.6f}",
                str(int(row["n_docs_a"])),
                str(int(row["n_docs_b"])),
            )

        console.print(table)

    def _print_detection_results(
        self, results: pd.DataFrame, summary: dict
    ) -> None:
        """Print detection results."""
        table = Table(
            title="Narrative Shift Detection Results",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Transition", style="cyan")
        table.add_column("Distance", justify="right", style="yellow")
        table.add_column("Z-Score", justify="right")
        table.add_column("Shift?", justify="center")
        table.add_column("Magnitude", justify="center")

        for _, row in results.iterrows():
            is_shift = row.get("is_shift", False)
            magnitude = row.get("shift_magnitude", "none")

            shift_marker = "⚡ YES" if is_shift else "—"
            shift_style = "bold red" if is_shift else "dim"

            mag_style = {
                "major": "bold red",
                "moderate": "yellow",
                "minor": "dim yellow",
                "none": "dim",
            }.get(magnitude, "dim")

            table.add_row(
                f"{row['period_a']} → {row['period_b']}",
                f"{row['distance']:.6f}",
                f"{row.get('z_score', 0):.2f}",
                f"[{shift_style}]{shift_marker}[/{shift_style}]",
                f"[{mag_style}]{magnitude}[/{mag_style}]",
            )

        console.print(table)

        # Summary stats
        console.print(f"\n  Detection method: [cyan]{self.config.detection.method}[/cyan]")
        console.print(f"  Mean distance: {summary['distance_mean']:.6f}")
        console.print(f"  Std distance: {summary['distance_std']:.6f}")
        console.print(
            f"  Shifts found: [{'red' if summary['total_shifts'] > 0 else 'green'}]"
            f"{summary['total_shifts']}/{summary['total_transitions']}"
            f"[/{'red' if summary['total_shifts'] > 0 else 'green'}]"
        )
