"""
Visualization module for Narameter pipeline.

Generates publication-quality plots for narrative shift analysis:
- Time-series distance plots with shift markers
- UMAP embedding space projections
- Pairwise distance heatmaps
- Distribution comparison plots
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from narameter.config import VisualizationConfig

logger = logging.getLogger(__name__)

# Use non-interactive backend for saving files
matplotlib.use("Agg")


# --- Color Palette ---
COLORS = {
    "primary": "#4F46E5",       # Indigo
    "secondary": "#7C3AED",     # Violet
    "accent": "#EC4899",        # Pink
    "success": "#10B981",       # Emerald
    "warning": "#F59E0B",       # Amber
    "danger": "#EF4444",        # Red
    "neutral": "#6B7280",       # Gray
    "background": "#1F2937",    # Dark gray
    "surface": "#374151",       # Medium gray
    "text": "#F9FAFB",          # Light gray
}

# Period colors for embedding plots
PERIOD_COLORS = [
    "#4F46E5", "#7C3AED", "#EC4899", "#F59E0B", "#10B981",
    "#06B6D4", "#8B5CF6", "#F97316", "#14B8A6", "#E11D48",
    "#3B82F6", "#A855F7", "#EF4444", "#22C55E", "#0EA5E9",
]


class NarrativeVisualizer:
    """
    Generates visualizations for narrative shift detection results.

    Supports both matplotlib (static, publication-quality) and
    plotly (interactive HTML) outputs.
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize the visualizer.

        Args:
            config: Visualization configuration. Uses defaults if None.
        """
        self.config = config or VisualizationConfig()

    def plot_timeline(
        self,
        results: pd.DataFrame,
        title: str = "Narrative Shift Detection — Wasserstein Distance Over Time",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot time-series of Wasserstein distances with shift markers.

        Args:
            results: DataFrame from NarrativeShiftDetector.detect().
            title: Plot title.
            save_path: Path to save the figure.
        """
        fig, ax = plt.subplots(
            figsize=tuple(self.config.figsize),
            facecolor=COLORS["background"],
        )
        ax.set_facecolor(COLORS["background"])

        # X-axis: transition labels
        x_labels = [
            f"{row['period_a']}\n→\n{row['period_b']}"
            for _, row in results.iterrows()
        ]
        x = np.arange(len(x_labels))
        distances = results["distance"].values

        # Plot distance line
        ax.plot(
            x, distances,
            color=COLORS["primary"],
            linewidth=2.5,
            marker="o",
            markersize=8,
            markerfacecolor=COLORS["primary"],
            markeredgecolor="white",
            markeredgewidth=1.5,
            zorder=3,
            label="Wasserstein Distance",
        )

        # Fill under curve
        ax.fill_between(
            x, distances, alpha=0.15, color=COLORS["primary"],
        )

        # Plot threshold line
        if "threshold" in results.columns:
            threshold = results["threshold"].iloc[0]
            ax.axhline(
                y=threshold,
                color=COLORS["warning"],
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
                label=f"Threshold ({threshold:.4f})",
                zorder=2,
            )

        # Mark detected shifts
        if "is_shift" in results.columns:
            shift_mask = results["is_shift"].values
            if shift_mask.any():
                shift_x = x[shift_mask]
                shift_y = distances[shift_mask]
                ax.scatter(
                    shift_x, shift_y,
                    color=COLORS["danger"],
                    s=200,
                    zorder=5,
                    marker="*",
                    edgecolors="white",
                    linewidth=1,
                    label="Narrative Shift Detected",
                )

                # Add magnitude and topic change annotations
                for idx, (sx, sy) in enumerate(zip(shift_x, shift_y)):
                    shift_row = results[results["is_shift"]].iloc[idx]
                    magnitude = shift_row.get("shift_magnitude", "")
                    summary = shift_row.get("shift_summary", "")
                    
                    annotation_text = "⚡"
                    if magnitude and magnitude != "none":
                        annotation_text += f" {magnitude}"
                    if summary and summary != "stable":
                        # Wrap long summaries or take first topic
                        first_topic = summary.split(";")[0] if ";" in summary else summary
                        # Truncate if too long
                        if len(first_topic) > 30:
                            first_topic = first_topic[:27] + "..."
                        annotation_text += f"\n{first_topic}"
                        
                    if len(annotation_text) > 1: # more than just the lightning bolt
                        ax.annotate(
                            annotation_text,
                            (sx, sy),
                            textcoords="offset points",
                            xytext=(0, 15),
                            ha="center",
                            fontsize=8,
                            fontweight="bold",
                            color=COLORS["danger"],
                            bbox=dict(boxstyle="round,pad=0.2", fc=COLORS["surface"], ec="none", alpha=0.8),
                        )

        # Rolling mean line
        if "rolling_mean" in results.columns and len(results) > 2:
            ax.plot(
                x, results["rolling_mean"].values,
                color=COLORS["success"],
                linewidth=1.5,
                linestyle=":",
                alpha=0.7,
                label="Rolling Mean",
                zorder=2,
            )

        # Styling
        ax.set_xlabel("Time Period Transition", fontsize=12, color=COLORS["text"], labelpad=15)
        ax.set_ylabel("Wasserstein Distance", fontsize=12, color=COLORS["text"], labelpad=15)
        ax.set_title(title, fontsize=14, fontweight="bold", color=COLORS["text"], pad=30)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=8, color=COLORS["text"], rotation=0)
        ax.tick_params(axis='x', pad=10)
        ax.tick_params(colors=COLORS["text"])
        legend = ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.30),
            ncol=3,
            fontsize=9,
            facecolor=COLORS["surface"],
            edgecolor=COLORS["neutral"],
            labelcolor=COLORS["text"],
        )
        ax.grid(True, alpha=0.2, color=COLORS["neutral"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(COLORS["neutral"])
        ax.spines["bottom"].set_color(COLORS["neutral"])

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(
                save_path,
                dpi=self.config.dpi,
                bbox_extra_artists=(legend,),
                bbox_inches="tight",
                facecolor=fig.get_facecolor(),
            )
            logger.info(f"Timeline plot saved to {save_path}")

        plt.close(fig)

    def plot_embedding_space(
        self,
        period_embeddings: dict[str, np.ndarray],
        title: str = "Embedding Space — UMAP Projection by Period",
        save_path: Optional[str] = None,
        method: str = "umap",
    ) -> None:
        """
        Plot 2D projection of embeddings colored by time period.

        Args:
            period_embeddings: Dict mapping period labels to embedding arrays.
            title: Plot title.
            save_path: Path to save the figure.
            method: Dimensionality reduction method ("umap" or "pca").
        """
        # Combine all embeddings
        all_embeddings = []
        labels = []
        period_labels = list(period_embeddings.keys())

        for period, embs in period_embeddings.items():
            all_embeddings.append(embs)
            labels.extend([period] * len(embs))

        combined = np.vstack(all_embeddings)
        labels = np.array(labels)

        # Dimensionality reduction
        if method == "umap":
            try:
                import umap
                reducer = umap.UMAP(
                    n_components=2,
                    random_state=42,
                    n_neighbors=min(15, len(combined) - 1),
                    min_dist=0.1,
                )
                coords = reducer.fit_transform(combined)
                method_label = "UMAP"
            except ImportError:
                logger.warning("umap-learn not installed, falling back to PCA")
                method = "pca"

        if method == "pca":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=42)
            coords = reducer.fit_transform(combined)
            method_label = "PCA"

        # Plot
        fig, ax = plt.subplots(
            figsize=(10, 8),
            facecolor=COLORS["background"],
        )
        ax.set_facecolor(COLORS["background"])

        for i, period in enumerate(period_labels):
            mask = labels == period
            color = PERIOD_COLORS[i % len(PERIOD_COLORS)]
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=color,
                label=period,
                alpha=0.6,
                s=25,
                edgecolors="white",
                linewidth=0.3,
                zorder=3,
            )

        # Styling
        ax.set_xlabel(f"{method_label} Dimension 1", fontsize=12, color=COLORS["text"])
        ax.set_ylabel(f"{method_label} Dimension 2", fontsize=12, color=COLORS["text"])
        ax.set_title(title, fontsize=14, fontweight="bold", color=COLORS["text"], pad=15)
        ax.tick_params(colors=COLORS["text"])
        ax.legend(
            title="Time Period",
            loc="best",
            fontsize=8,
            title_fontsize=10,
            facecolor=COLORS["surface"],
            edgecolor=COLORS["neutral"],
            labelcolor=COLORS["text"],
            markerscale=1.5,
        )
        ax.grid(True, alpha=0.15, color=COLORS["neutral"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(COLORS["neutral"])
        ax.spines["bottom"].set_color(COLORS["neutral"])

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(
                save_path,
                dpi=self.config.dpi,
                bbox_inches="tight",
                facecolor=fig.get_facecolor(),
            )
            logger.info(f"Embedding space plot saved to {save_path}")

        plt.close(fig)

    def plot_distance_matrix(
        self,
        distance_matrix: np.ndarray,
        labels: list[str],
        title: str = "Pairwise Wasserstein Distance Matrix",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot heatmap of pairwise distances between periods.

        Args:
            distance_matrix: Square matrix of distances.
            labels: Period labels.
            title: Plot title.
            save_path: Path to save the figure.
        """
        fig, ax = plt.subplots(
            figsize=(max(8, len(labels) * 0.8), max(6, len(labels) * 0.7)),
            facecolor=COLORS["background"],
        )
        ax.set_facecolor(COLORS["background"])

        # Heatmap
        im = ax.imshow(
            distance_matrix,
            cmap="magma_r",
            aspect="auto",
            interpolation="nearest",
        )

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(
            "Wasserstein Distance",
            fontsize=11,
            color=COLORS["text"],
        )
        cbar.ax.tick_params(colors=COLORS["text"])

        # Labels
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, fontsize=8, color=COLORS["text"], rotation=45, ha="right")
        ax.set_yticklabels(labels, fontsize=8, color=COLORS["text"])

        # Annotate cells with values
        for i in range(len(labels)):
            for j in range(len(labels)):
                value = distance_matrix[i, j]
                text_color = "white" if value > distance_matrix.max() * 0.5 else "black"
                ax.text(
                    j, i, f"{value:.3f}",
                    ha="center", va="center",
                    fontsize=7, color=text_color,
                )

        ax.set_title(title, fontsize=14, fontweight="bold", color=COLORS["text"], pad=15)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(
                save_path,
                dpi=self.config.dpi,
                bbox_inches="tight",
                facecolor=fig.get_facecolor(),
            )
            logger.info(f"Distance matrix plot saved to {save_path}")

        plt.close(fig)

    def plot_distribution_comparison(
        self,
        emb_a: np.ndarray,
        emb_b: np.ndarray,
        label_a: str,
        label_b: str,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot side-by-side distribution comparison using PCA projections.

        Shows how two period distributions differ in the principal component space.

        Args:
            emb_a: Embeddings from period A.
            emb_b: Embeddings from period B.
            label_a: Label for period A.
            label_b: Label for period B.
            title: Plot title.
            save_path: Path to save the figure.
        """
        from sklearn.decomposition import PCA

        # Fit PCA on combined data
        combined = np.vstack([emb_a, emb_b])
        pca = PCA(n_components=2, random_state=42)
        combined_pca = pca.fit_transform(combined)

        pca_a = combined_pca[: len(emb_a)]
        pca_b = combined_pca[len(emb_a):]

        title = title or f"Distribution Comparison: {label_a} vs {label_b}"

        fig, axes = plt.subplots(
            1, 3,
            figsize=(18, 5),
            facecolor=COLORS["background"],
        )

        for ax in axes:
            ax.set_facecolor(COLORS["background"])

        # --- Subplot 1: 2D scatter ---
        axes[0].scatter(
            pca_a[:, 0], pca_a[:, 1],
            c=COLORS["primary"], alpha=0.5, s=20, label=label_a,
            edgecolors="white", linewidth=0.3,
        )
        axes[0].scatter(
            pca_b[:, 0], pca_b[:, 1],
            c=COLORS["accent"], alpha=0.5, s=20, label=label_b,
            edgecolors="white", linewidth=0.3,
        )
        axes[0].set_xlabel("PC1", fontsize=10, color=COLORS["text"])
        axes[0].set_ylabel("PC2", fontsize=10, color=COLORS["text"])
        axes[0].set_title("2D Scatter", fontsize=11, color=COLORS["text"])
        axes[0].legend(
            fontsize=8, facecolor=COLORS["surface"],
            edgecolor=COLORS["neutral"], labelcolor=COLORS["text"],
        )
        axes[0].tick_params(colors=COLORS["text"])

        # --- Subplot 2: PC1 histogram ---
        axes[1].hist(
            pca_a[:, 0], bins=30, alpha=0.6, color=COLORS["primary"],
            label=label_a, density=True, edgecolor="white", linewidth=0.5,
        )
        axes[1].hist(
            pca_b[:, 0], bins=30, alpha=0.6, color=COLORS["accent"],
            label=label_b, density=True, edgecolor="white", linewidth=0.5,
        )
        axes[1].set_xlabel("PC1", fontsize=10, color=COLORS["text"])
        axes[1].set_ylabel("Density", fontsize=10, color=COLORS["text"])
        axes[1].set_title("PC1 Distribution", fontsize=11, color=COLORS["text"])
        axes[1].legend(
            fontsize=8, facecolor=COLORS["surface"],
            edgecolor=COLORS["neutral"], labelcolor=COLORS["text"],
        )
        axes[1].tick_params(colors=COLORS["text"])

        # --- Subplot 3: PC2 histogram ---
        axes[2].hist(
            pca_a[:, 1], bins=30, alpha=0.6, color=COLORS["primary"],
            label=label_a, density=True, edgecolor="white", linewidth=0.5,
        )
        axes[2].hist(
            pca_b[:, 1], bins=30, alpha=0.6, color=COLORS["accent"],
            label=label_b, density=True, edgecolor="white", linewidth=0.5,
        )
        axes[2].set_xlabel("PC2", fontsize=10, color=COLORS["text"])
        axes[2].set_ylabel("Density", fontsize=10, color=COLORS["text"])
        axes[2].set_title("PC2 Distribution", fontsize=11, color=COLORS["text"])
        axes[2].legend(
            fontsize=8, facecolor=COLORS["surface"],
            edgecolor=COLORS["neutral"], labelcolor=COLORS["text"],
        )
        axes[2].tick_params(colors=COLORS["text"])

        # Common styling
        for ax in axes:
            ax.grid(True, alpha=0.15, color=COLORS["neutral"])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color(COLORS["neutral"])
            ax.spines["bottom"].set_color(COLORS["neutral"])

        fig.suptitle(title, fontsize=14, fontweight="bold", color=COLORS["text"], y=1.02)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(
                save_path,
                dpi=self.config.dpi,
                bbox_inches="tight",
                facecolor=fig.get_facecolor(),
            )
            logger.info(f"Distribution comparison plot saved to {save_path}")

        plt.close(fig)

    def plot_timeline_interactive(
        self,
        results: pd.DataFrame,
        title: str = "Narrative Shift Detection — Interactive",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Generate interactive plotly timeline plot.

        Args:
            results: DataFrame from NarrativeShiftDetector.detect().
            title: Plot title.
            save_path: Path to save the HTML file.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            logger.warning("plotly not installed, skipping interactive plot")
            return

        x_labels = [
            f"{row['period_a']} → {row['period_b']}"
            for _, row in results.iterrows()
        ]

        fig = go.Figure()

        # Distance line
        fig.add_trace(go.Scatter(
            x=x_labels,
            y=results["distance"],
            mode="lines+markers",
            name="Wasserstein Distance",
            line=dict(color=COLORS["primary"], width=3),
            marker=dict(size=10, color=COLORS["primary"], line=dict(width=2, color="white")),
            hovertemplate="<b>%{x}</b><br>Distance: %{y:.4f}<extra></extra>",
        ))

        # Threshold line
        if "threshold" in results.columns:
            fig.add_trace(go.Scatter(
                x=x_labels,
                y=[results["threshold"].iloc[0]] * len(x_labels),
                mode="lines",
                name="Threshold",
                line=dict(color=COLORS["warning"], width=2, dash="dash"),
            ))

        # Shift markers
        if "is_shift" in results.columns:
            shifts = results[results["is_shift"]]
            if not shifts.empty:
                shift_labels = [
                    f"{row['period_a']} → {row['period_b']}"
                    for _, row in shifts.iterrows()
                ]
                
                # Extract summaries for hover text
                custom_data = []
                for _, row in shifts.iterrows():
                    summary = row.get("shift_summary", "")
                    if summary and summary != "stable":
                        # format as bullet points
                        formatted = "<br> • " + "<br> • ".join(summary.split("; "))
                    else:
                        formatted = "<br> • No identifiable topic shift"
                    custom_data.append(formatted)

                fig.add_trace(go.Scatter(
                    x=shift_labels,
                    y=shifts["distance"],
                    mode="markers",
                    name="Narrative Shift",
                    customdata=custom_data,
                    marker=dict(
                        size=18,
                        color=COLORS["danger"],
                        symbol="star",
                        line=dict(width=2, color="white"),
                    ),
                    hovertemplate=(
                        "<b>⚡ SHIFT DETECTED</b><br>"
                        "%{x}<br>"
                        "Distance: %{y:.4f}<br>"
                        "<b>Top Topic Changes:</b>%{customdata}<br>"
                        "<extra></extra>"
                    ),
                ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=16, color=COLORS["text"])),
            plot_bgcolor=COLORS["background"],
            paper_bgcolor=COLORS["background"],
            font=dict(color=COLORS["text"]),
            xaxis=dict(title="Time Period Transition", gridcolor=COLORS["surface"]),
            yaxis=dict(title="Wasserstein Distance", gridcolor=COLORS["surface"]),
            legend=dict(
                bgcolor=COLORS["surface"],
                bordercolor=COLORS["neutral"],
            ),
            hovermode="x unified",
        )

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.write_html(save_path)
            logger.info(f"Interactive plot saved to {save_path}")

    def generate_report(
        self,
        results: pd.DataFrame,
        period_embeddings: dict[str, np.ndarray],
        output_dir: str = "output",
        distance_matrix: Optional[np.ndarray] = None,
        distance_labels: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Generate a complete set of visualizations.

        Args:
            results: Detection results DataFrame.
            period_embeddings: Period-to-embeddings mapping.
            output_dir: Output directory for plots.
            distance_matrix: Optional precomputed distance matrix.
            distance_labels: Labels for distance matrix.

        Returns:
            List of generated file paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        generated_files = []

        # 1. Timeline plot (static)
        timeline_path = os.path.join(output_dir, f"timeline.{self.config.format}")
        self.plot_timeline(results, save_path=timeline_path)
        generated_files.append(timeline_path)

        # 2. Timeline plot (interactive)
        if self.config.interactive:
            interactive_path = os.path.join(output_dir, "timeline_interactive.html")
            self.plot_timeline_interactive(results, save_path=interactive_path)
            generated_files.append(interactive_path)

        # 3. Embedding space
        if period_embeddings:
            embedding_path = os.path.join(output_dir, f"embedding_space.{self.config.format}")
            self.plot_embedding_space(period_embeddings, save_path=embedding_path)
            generated_files.append(embedding_path)

        # 4. Distance matrix heatmap
        if distance_matrix is not None and distance_labels is not None:
            matrix_path = os.path.join(output_dir, f"distance_matrix.{self.config.format}")
            self.plot_distance_matrix(
                distance_matrix, distance_labels, save_path=matrix_path
            )
            generated_files.append(matrix_path)

        # 5. Distribution comparison for detected shifts
        if "is_shift" in results.columns and results["is_shift"].any():
            shifts = results[results["is_shift"]]
            for idx, row in shifts.iterrows():
                pa, pb = row["period_a"], row["period_b"]
                if pa in period_embeddings and pb in period_embeddings:
                    safe_name = f"comparison_{pa}_{pb}".replace(" ", "_").replace("/", "-")
                    comp_path = os.path.join(
                        output_dir, f"{safe_name}.{self.config.format}"
                    )
                    self.plot_distribution_comparison(
                        period_embeddings[pa],
                        period_embeddings[pb],
                        pa, pb,
                        save_path=comp_path,
                    )
                    generated_files.append(comp_path)

        logger.info(f"Generated {len(generated_files)} visualization files in {output_dir}")
        return generated_files
