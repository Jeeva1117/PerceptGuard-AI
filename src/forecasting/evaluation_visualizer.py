"""PerceptGuard AI - Evaluation Visualizer.

Generates publication-grade Matplotlib diagnostic charts for multi-horizon
forecasting evaluation:
1. Actual vs Predicted Reliability (actual_vs_predicted_reliability.png)
2. MAE by Horizon (mae_by_horizon.png)
3. RMSE by Horizon (rmse_by_horizon.png)
4. Prediction Error Distribution (prediction_error_distribution.png)
5. Baseline vs LSTM Comparison (baseline_vs_lstm_comparison.png)
6. Residual Analysis (residual_analysis.png)

Strictly uses non-interactive backend (matplotlib.use('Agg')) for headless
CPU execution and automated CI/CD pipelines.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
# Enforce headless Agg backend before importing pyplot
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.forecasting.model_evaluator import (
    AlignedEvaluationPair,
    HorizonEvaluationMetrics,
)

logger = logging.getLogger(__name__)


class VisualizationError(Exception):
    """Raised when evaluation plot generation fails."""
    pass


class ForecastingEvaluationVisualizer:
    """Generates publication-quality diagnostic charts for forecasting evaluation."""

    def __init__(
        self,
        style: str = "default",
        dpi: int = 300,
        figsize_standard: tuple[int, int] = (10, 6),
        figsize_wide: tuple[int, int] = (12, 6),
    ) -> None:
        """Initialize visualizer with rendering settings.

        Args:
            style: Matplotlib style name.
            dpi: Output image resolution (DPI).
            figsize_standard: Figure size for single-plot charts.
            figsize_wide: Figure size for comparison or multi-panel charts.
        """
        self.dpi = dpi
        self.figsize_standard = figsize_standard
        self.figsize_wide = figsize_wide
        self.style = style

        # Color palette for horizons and models
        self.horizon_colors = {
            1: "#1f77b4",  # Blue
            3: "#ff7f0e",  # Orange
            5: "#2ca02c",  # Green
        }
        self.model_colors = {
            "LSTMForecaster": "#1f77b4",
            "Baseline_Persistence": "#d62728",
            "Baseline_Mean": "#9467bd",
            "Baseline_Trend": "#8c564b",
        }

    def _setup_style(self) -> None:
        """Apply clean visual styling."""
        plt.style.use(self.style)
        plt.rcParams.update({
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.edgecolor": "#cccccc",
            "axes.linewidth": 0.8,
            "grid.color": "#e0e0e0",
            "grid.linestyle": "--",
            "grid.alpha": 0.7,
        })

    def plot_actual_vs_predicted(
        self,
        pairs: List[AlignedEvaluationPair],
        output_path: Union[str, Path],
    ) -> Path:
        """Generate actual vs. predicted reliability scatter plot with identity reference line.

        Args:
            pairs: List of aligned evaluation pairs.
            output_path: File destination for the PNG plot.

        Returns:
            Path to saved figure.
        """
        if not pairs:
            raise VisualizationError("Cannot generate actual vs predicted plot: pairs list is empty.")

        self._setup_style()
        fig, ax = plt.subplots(figsize=self.figsize_standard, dpi=self.dpi)

        # Group by horizon for color coding
        horizons = sorted(list(set(p.horizon for p in pairs)))
        for h in horizons:
            h_pairs = [p for p in pairs if p.horizon == h]
            actuals = [p.ground_truth_reliability for p in h_pairs]
            preds = [p.predicted_reliability for p in h_pairs]
            color = self.horizon_colors.get(h, "#333333")
            ax.scatter(
                actuals,
                preds,
                c=color,
                label=f"Horizon {h} (N={len(h_pairs)})",
                alpha=0.65,
                edgecolors="none",
                s=35,
            )

        # Plot 45-degree identity reference line
        all_actuals = [p.ground_truth_reliability for p in pairs]
        all_preds = [p.predicted_reliability for p in pairs]
        min_val = min(min(all_actuals), min(all_preds), 0.0)
        max_val = max(max(all_actuals), max(all_preds), 1.0)
        margin = max((max_val - min_val) * 0.05, 0.05)
        line_range = np.linspace(min_val - margin, max_val + margin, 100)

        ax.plot(
            line_range,
            line_range,
            color="#d62728",
            linestyle="--",
            linewidth=1.5,
            label="Ideal Alignment (y = x)",
            zorder=4,
        )

        # Annotate overall metrics
        overall_mae = float(np.mean([p.absolute_error for p in pairs]))
        overall_rmse = float(np.sqrt(np.mean([p.error ** 2 for p in pairs])))
        r_corr = (
            np.corrcoef(all_actuals, all_preds)[0, 1]
            if len(pairs) > 1 and np.std(all_actuals) > 1e-6 and np.std(all_preds) > 1e-6
            else 0.0
        )

        stats_text = (
            f"Total Pairs: {len(pairs)}\n"
            f"Overall MAE: {overall_mae:.4f}\n"
            f"Overall RMSE: {overall_rmse:.4f}\n"
            f"Pearson r: {r_corr:.3f}"
        )
        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.9),
            fontsize=9,
        )

        ax.set_title("Actual vs. Predicted Detection Reliability", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Ground-Truth Reliability (Observed)", fontsize=11, labelpad=8)
        ax.set_ylabel("Predicted Reliability (LSTM)", fontsize=11, labelpad=8)
        ax.set_xlim(min_val - margin, max_val + margin)
        ax.set_ylim(min_val - margin, max_val + margin)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="lower right", framealpha=0.9)

        plt.tight_layout()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=self.dpi)
        plt.close(fig)
        return out

    def plot_mae_by_horizon(
        self,
        metrics: Dict[str, HorizonEvaluationMetrics],
        output_path: Union[str, Path],
    ) -> Path:
        """Generate MAE bar chart across forward horizons.

        Args:
            metrics: Mapping of horizon key to HorizonEvaluationMetrics.
            output_path: Destination path for the plot.

        Returns:
            Path to saved figure.
        """
        if not metrics:
            raise VisualizationError("Cannot generate MAE by horizon plot: metrics dictionary is empty.")

        self._setup_style()
        fig, ax = plt.subplots(figsize=self.figsize_standard, dpi=self.dpi)

        sorted_metrics = sorted(metrics.values(), key=lambda m: m.horizon)
        horizons = [f"Horizon {m.horizon} ({m.horizon}f)" for m in sorted_metrics]
        mae_values = [m.mae for m in sorted_metrics]
        colors = [self.horizon_colors.get(m.horizon, "#1f77b4") for m in sorted_metrics]

        bars = ax.bar(horizons, mae_values, color=colors, width=0.45, edgecolor="#333333", linewidth=0.8, alpha=0.85)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )

        ax.set_title("Mean Absolute Error (MAE) by Forecasting Horizon", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Forecasting Horizon (Frames Ahead)", fontsize=11, labelpad=8)
        ax.set_ylabel("Mean Absolute Error (MAE)", fontsize=11, labelpad=8)
        max_y = max(mae_values) if mae_values else 0.1
        ax.set_ylim(0, max_y * 1.25 + 0.02)
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        plt.tight_layout()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=self.dpi)
        plt.close(fig)
        return out

    def plot_rmse_by_horizon(
        self,
        metrics: Dict[str, HorizonEvaluationMetrics],
        output_path: Union[str, Path],
    ) -> Path:
        """Generate RMSE bar chart across forward horizons.

        Args:
            metrics: Mapping of horizon key to HorizonEvaluationMetrics.
            output_path: Destination path for the plot.

        Returns:
            Path to saved figure.
        """
        if not metrics:
            raise VisualizationError("Cannot generate RMSE by horizon plot: metrics dictionary is empty.")

        self._setup_style()
        fig, ax = plt.subplots(figsize=self.figsize_standard, dpi=self.dpi)

        sorted_metrics = sorted(metrics.values(), key=lambda m: m.horizon)
        horizons = [f"Horizon {m.horizon} ({m.horizon}f)" for m in sorted_metrics]
        rmse_values = [m.rmse for m in sorted_metrics]
        colors = [self.horizon_colors.get(m.horizon, "#2ca02c") for m in sorted_metrics]

        bars = ax.bar(horizons, rmse_values, color=colors, width=0.45, edgecolor="#333333", linewidth=0.8, alpha=0.85)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )

        ax.set_title("Root Mean Squared Error (RMSE) by Forecasting Horizon", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Forecasting Horizon (Frames Ahead)", fontsize=11, labelpad=8)
        ax.set_ylabel("Root Mean Squared Error (RMSE)", fontsize=11, labelpad=8)
        max_y = max(rmse_values) if rmse_values else 0.1
        ax.set_ylim(0, max_y * 1.25 + 0.02)
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        plt.tight_layout()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=self.dpi)
        plt.close(fig)
        return out

    def plot_error_distribution(
        self,
        pairs: List[AlignedEvaluationPair],
        output_path: Union[str, Path],
    ) -> Path:
        """Generate prediction error distribution histogram with statistical indicators.

        Args:
            pairs: List of aligned evaluation pairs.
            output_path: Destination path for the plot.

        Returns:
            Path to saved figure.
        """
        if not pairs:
            raise VisualizationError("Cannot generate error distribution plot: pairs list is empty.")

        self._setup_style()
        fig, ax = plt.subplots(figsize=self.figsize_standard, dpi=self.dpi)

        errors = np.array([p.error for p in pairs])
        mean_err = float(np.mean(errors))
        median_err = float(np.median(errors))
        std_err = float(np.std(errors))

        n_bins = min(max(int(np.sqrt(len(errors)) * 2), 15), 50)
        counts, bins, patches = ax.hist(
            errors,
            bins=n_bins,
            density=True,
            color="#3470a3",
            edgecolor="white",
            alpha=0.75,
            label="Error Distribution",
        )

        ax.axvline(0.0, color="#d62728", linestyle="--", linewidth=1.8, label="Zero Error (e = 0)")
        ax.axvline(mean_err, color="#ff7f0e", linestyle="-", linewidth=1.5, label=f"Mean Error (Bias: {mean_err:+.4f})")
        ax.axvline(median_err, color="#2ca02c", linestyle=":", linewidth=1.5, label=f"Median Error: {median_err:+.4f}")

        stats_text = (
            f"Sample Count: {len(errors)}\n"
            f"Mean (Bias): {mean_err:+.4f}\n"
            f"Std Dev: {std_err:.4f}\n"
            f"Median: {median_err:+.4f}\n"
            f"Min Error: {np.min(errors):+.4f}\n"
            f"Max Error: {np.max(errors):+.4f}"
        )
        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.9),
            fontsize=9,
        )

        ax.set_title("Forecasting Error Distribution ($e = \\hat{y} - y$)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Prediction Error (Residual)", fontsize=11, labelpad=8)
        ax.set_ylabel("Probability Density", fontsize=11, labelpad=8)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper right", framealpha=0.9)

        plt.tight_layout()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=self.dpi)
        plt.close(fig)
        return out

    def plot_baseline_comparison(
        self,
        comparison_records: List[Dict[str, Any]],
        output_path: Union[str, Path],
    ) -> Path:
        """Generate grouped bar chart comparing LSTM against baseline forecasting models.

        Args:
            comparison_records: Records from compare_against_baselines().
            output_path: Destination path for the plot.

        Returns:
            Path to saved figure.
        """
        if not comparison_records:
            raise VisualizationError("Cannot generate baseline comparison plot: comparison records list is empty.")

        self._setup_style()
        fig, ax = plt.subplots(figsize=self.figsize_wide, dpi=self.dpi)

        models = []
        for r in comparison_records:
            m_name = r.get("model_name", "Unknown")
            if m_name not in models:
                models.append(m_name)

        horizons = sorted(list(set(r.get("horizon", 1) for r in comparison_records)))

        x = np.arange(len(horizons))
        n_models = len(models)
        width = 0.8 / max(n_models, 1)

        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        for i, m_name in enumerate(models):
            mae_vals = []
            for h in horizons:
                rec = next((r for r in comparison_records if r.get("model_name") == m_name and r.get("horizon") == h), None)
                mae_vals.append(rec.get("mae", 0.0) if rec and rec.get("mae") is not None else 0.0)

            offset = (i - n_models / 2 + 0.5) * width
            color = palette[i % len(palette)]
            bars = ax.bar(
                x + offset,
                mae_vals,
                width,
                label=m_name.replace("Baseline_", "Baseline: "),
                color=color,
                edgecolor="#333333",
                linewidth=0.6,
                alpha=0.85,
            )

            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(
                        f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        rotation=0,
                    )

        ax.set_title("Forecast Accuracy Comparison: LSTM vs Baseline Models (MAE)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Forecasting Horizon (Frames Ahead)", fontsize=11, labelpad=8)
        ax.set_ylabel("Mean Absolute Error (Lower is Better)", fontsize=11, labelpad=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Horizon {h} ({h}f)" for h in horizons])
        ax.legend(loc="upper left", framealpha=0.9)
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        plt.tight_layout()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=self.dpi)
        plt.close(fig)
        return out

    def plot_residuals(
        self,
        pairs: List[AlignedEvaluationPair],
        output_path: Union[str, Path],
    ) -> Path:
        """Generate residual analysis scatter plot ($e_i$ vs $\\hat{y}_i$).

        Args:
            pairs: List of aligned evaluation pairs.
            output_path: Destination path for the plot.

        Returns:
            Path to saved figure.
        """
        if not pairs:
            raise VisualizationError("Cannot generate residual plot: pairs list is empty.")

        self._setup_style()
        fig, ax = plt.subplots(figsize=self.figsize_standard, dpi=self.dpi)

        preds = [p.predicted_reliability for p in pairs]
        errors = [p.error for p in pairs]
        horizons = sorted(list(set(p.horizon for p in pairs)))

        for h in horizons:
            h_pairs = [p for p in pairs if p.horizon == h]
            h_preds = [p.predicted_reliability for p in h_pairs]
            h_errs = [p.error for p in h_pairs]
            color = self.horizon_colors.get(h, "#1f77b4")
            ax.scatter(
                h_preds,
                h_errs,
                c=color,
                label=f"Horizon {h}",
                alpha=0.65,
                s=35,
                edgecolors="none",
            )

        min_x = min(preds) if preds else 0.0
        max_x = max(preds) if preds else 1.0
        margin_x = max((max_x - min_x) * 0.05, 0.05)
        ax.axhline(0.0, color="#d62728", linestyle="--", linewidth=1.5, label="Zero Residual (e = 0)")

        ax.set_title("Residual Analysis: Prediction Error vs. Predicted Reliability", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Predicted Detection Reliability ($\\hat{y}$)", fontsize=11, labelpad=8)
        ax.set_ylabel("Residual Error ($e = \\hat{y} - y$)", fontsize=11, labelpad=8)
        ax.set_xlim(min_x - margin_x, max_x + margin_x)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper right", framealpha=0.9)

        plt.tight_layout()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=self.dpi)
        plt.close(fig)
        return out

    def plot_actual_vs_predicted_horizon(
        self,
        pairs: List[AlignedEvaluationPair],
        horizon: int,
        output_path: Union[str, Path],
    ) -> Path:
        """Generate actual vs predicted scatter plot specifically for a single horizon."""
        h_pairs = [p for p in pairs if p.horizon == horizon]
        if not h_pairs:
            raise VisualizationError(f"Cannot generate plot: no pairs for horizon {horizon}")

        self._setup_style()
        fig, ax = plt.subplots(figsize=self.figsize_standard, dpi=self.dpi)

        actuals = np.array([p.ground_truth_reliability for p in h_pairs])
        preds = np.array([p.predicted_reliability for p in h_pairs])
        color = self.horizon_colors.get(horizon, "#1f77b4")

        ax.scatter(actuals, preds, c=color, label=f"Horizon {horizon} (N={len(h_pairs)})", alpha=0.75, s=40, edgecolors="none")

        min_val = min(float(np.min(actuals)), float(np.min(preds)), 0.0)
        max_val = max(float(np.max(actuals)), float(np.max(preds)), 0.04)
        margin = max((max_val - min_val) * 0.1, 0.005)
        line_range = np.linspace(min_val - margin, max_val + margin, 100)
        ax.plot(line_range, line_range, color="#d62728", linestyle="--", linewidth=1.5, label="Ideal (y = x)")

        h_mae = float(np.mean(np.abs(actuals - preds)))
        h_rmse = float(np.sqrt(np.mean((actuals - preds) ** 2)))
        stats_text = f"Horizon {horizon} ({horizon}f ahead)\nSamples: {len(h_pairs)}\nMAE: {h_mae:.4f}\nRMSE: {h_rmse:.4f}"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.9), fontsize=9)

        ax.set_title(f"Actual vs. Predicted Reliability — Horizon {horizon} (t + {horizon} frames)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Ground-Truth Reliability (Observed)", fontsize=11, labelpad=8)
        ax.set_ylabel(f"Predicted Reliability (LSTM H{horizon})", fontsize=11, labelpad=8)
        ax.set_xlim(min_val - margin, max_val + margin)
        ax.set_ylim(min_val - margin, max_val + margin)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="lower right", framealpha=0.9)

        plt.tight_layout()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=self.dpi)
        plt.close(fig)
        return out

    def plot_rmse_comparison_across_models(
        self,
        comparison_records: List[Dict[str, Any]],
        output_path: Union[str, Path],
    ) -> Path:
        """Generate grouped bar chart comparing RMSE across models."""
        if not comparison_records:
            raise VisualizationError("Cannot generate RMSE comparison plot: comparison records list is empty.")

        self._setup_style()
        fig, ax = plt.subplots(figsize=self.figsize_wide, dpi=self.dpi)

        models = []
        for r in comparison_records:
            m_name = r.get("model_name", "Unknown")
            if m_name not in models:
                models.append(m_name)

        horizons = sorted(list(set(r.get("horizon", 1) for r in comparison_records)))
        x = np.arange(len(horizons))
        n_models = len(models)
        width = 0.8 / max(n_models, 1)
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        for i, m_name in enumerate(models):
            rmse_vals = []
            for h in horizons:
                rec = next((r for r in comparison_records if r.get("model_name") == m_name and r.get("horizon") == h), None)
                rmse_vals.append(rec.get("rmse", 0.0) if rec and rec.get("rmse") is not None else 0.0)

            offset = (i - n_models / 2 + 0.5) * width
            color = palette[i % len(palette)]
            bars = ax.bar(x + offset, rmse_vals, width, label=m_name.replace("Baseline_", "Baseline: "), color=color, edgecolor="#333333", linewidth=0.6, alpha=0.85)
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f"{height:.4f}", xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

        ax.set_title("Forecast Accuracy Comparison: RMSE Across Models", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Forecasting Horizon (Frames Ahead)", fontsize=11, labelpad=8)
        ax.set_ylabel("Root Mean Squared Error (RMSE) (Lower is Better)", fontsize=11, labelpad=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Horizon {h} ({h}f)" for h in horizons])
        ax.legend(loc="upper left", framealpha=0.9)
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        plt.tight_layout()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=self.dpi)
        plt.close(fig)
        return out

    def plot_reliability_forecasting_timeline(
        self,
        pairs: List[AlignedEvaluationPair],
        output_path: Union[str, Path],
    ) -> Path:
        """Generate sequential timeline plot showing actual vs predicted reliability across time."""
        if not pairs:
            raise VisualizationError("Cannot generate timeline plot: pairs list is empty.")

        self._setup_style()
        fig, ax = plt.subplots(figsize=self.figsize_wide, dpi=self.dpi)

        h1_pairs = sorted([p for p in pairs if p.horizon == 1], key=lambda p: p.target_frame_index)
        h3_pairs = sorted([p for p in pairs if p.horizon == 3], key=lambda p: p.target_frame_index)
        h5_pairs = sorted([p for p in pairs if p.horizon == 5], key=lambda p: p.target_frame_index)

        if h1_pairs:
            x1 = np.arange(len(h1_pairs))
            actuals = [p.ground_truth_reliability for p in h1_pairs]
            preds_h1 = [p.predicted_reliability for p in h1_pairs]
            ax.plot(x1, actuals, color="#d62728", linestyle="-", linewidth=2.0, marker="o", markersize=4, label="Actual Ground Truth", zorder=5)
            ax.plot(x1, preds_h1, color="#1f77b4", linestyle="--", linewidth=1.5, marker="s", markersize=3, label="LSTM Forecast H1 (t+1)", alpha=0.85)

        if h3_pairs:
            x3 = np.arange(len(h3_pairs))
            preds_h3 = [p.predicted_reliability for p in h3_pairs]
            ax.plot(x3, preds_h3, color="#ff7f0e", linestyle=":", linewidth=1.5, marker="^", markersize=3, label="LSTM Forecast H3 (t+3)", alpha=0.85)

        if h5_pairs:
            x5 = np.arange(len(h5_pairs))
            preds_h5 = [p.predicted_reliability for p in h5_pairs]
            ax.plot(x5, preds_h5, color="#2ca02c", linestyle="-.", linewidth=1.5, marker="v", markersize=3, label="LSTM Forecast H5 (t+5)", alpha=0.85)

        ax.set_title("Reliability Forecasting Timeline: Actual vs Multi-Horizon Forecasts", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Test Evaluation Sequence Step", fontsize=11, labelpad=8)
        ax.set_ylabel("Perception Reliability ([0.0, 1.0])", fontsize=11, labelpad=8)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper right", framealpha=0.9)

        plt.tight_layout()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=self.dpi)
        plt.close(fig)
        return out

    def plot_horizon_wise_performance(
        self,
        metrics: Dict[str, HorizonEvaluationMetrics],
        output_path: Union[str, Path],
    ) -> Path:
        """Generate combined MAE and RMSE performance curve across horizons."""
        if not metrics:
            raise VisualizationError("Cannot generate horizon performance plot: metrics dictionary is empty.")

        self._setup_style()
        fig, ax = plt.subplots(figsize=self.figsize_standard, dpi=self.dpi)

        sorted_m = sorted(metrics.values(), key=lambda m: m.horizon)
        horizons = [m.horizon for m in sorted_m]
        maes = [m.mae for m in sorted_m]
        rmses = [m.rmse for m in sorted_m]

        ax.plot(horizons, maes, color="#1f77b4", marker="o", linewidth=2.0, markersize=7, label="MAE (Mean Absolute Error)")
        ax.plot(horizons, rmses, color="#ff7f0e", marker="s", linewidth=2.0, markersize=7, linestyle="--", label="RMSE (Root Mean Squared Error)")

        for h, m, r in zip(horizons, maes, rmses):
            ax.annotate(f"{m:.4f}", xy=(h, m), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=9, fontweight="bold", color="#1f77b4")
            ax.annotate(f"{r:.4f}", xy=(h, r), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=9, fontweight="bold", color="#ff7f0e")

        ax.set_title("Horizon-Wise Performance & Error Degradation", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Forecast Horizon (Frames Ahead: H1=1, H3=3, H5=5)", fontsize=11, labelpad=8)
        ax.set_ylabel("Error Metric Value", fontsize=11, labelpad=8)
        ax.set_xticks(horizons)
        ax.set_xticklabels([f"H{h} ({h}f)" for h in horizons])
        max_y = max(max(maes), max(rmses)) if maes and rmses else 0.05
        ax.set_ylim(0.0, max_y * 1.3)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper left", framealpha=0.9)

        plt.tight_layout()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=self.dpi)
        plt.close(fig)
        return out

    def generate_all_plots(
        self,
        pairs: List[AlignedEvaluationPair],
        metrics: Dict[str, HorizonEvaluationMetrics],
        comparison_records: Optional[List[Dict[str, Any]]] = None,
        output_dir: Union[str, Path] = "artifacts/evaluation/plots",
    ) -> Dict[str, Path]:
        """Generate the standard suite of 6 evaluation diagnostic plots.

        Args:
            pairs: List of aligned evaluation pairs.
            metrics: Mapping of horizon keys to HorizonEvaluationMetrics.
            comparison_records: Optional baseline comparison records.
            output_dir: Directory where figures will be saved.

        Returns:
            Dictionary mapping plot name to saved Path (5 or 6 plots).
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        generated_plots: Dict[str, Path] = {}

        if not pairs:
            logger.warning("Empty pairs provided to generate_all_plots. Skipping scatter/distribution plots.")
            return generated_plots

        try:
            p1 = self.plot_actual_vs_predicted(
                pairs=pairs,
                output_path=out_dir / "actual_vs_predicted_reliability.png",
            )
            generated_plots["actual_vs_predicted"] = p1
        except Exception as e:
            logger.warning("Failed to generate actual_vs_predicted plot: %s", e)

        try:
            p2 = self.plot_mae_by_horizon(
                metrics=metrics,
                output_path=out_dir / "mae_by_horizon.png",
            )
            generated_plots["mae_by_horizon"] = p2
        except Exception as e:
            logger.warning("Failed to generate mae_by_horizon plot: %s", e)

        try:
            p3 = self.plot_rmse_by_horizon(
                metrics=metrics,
                output_path=out_dir / "rmse_by_horizon.png",
            )
            generated_plots["rmse_by_horizon"] = p3
        except Exception as e:
            logger.warning("Failed to generate rmse_by_horizon plot: %s", e)

        try:
            p4 = self.plot_error_distribution(
                pairs=pairs,
                output_path=out_dir / "prediction_error_distribution.png",
            )
            generated_plots["error_distribution"] = p4
        except Exception as e:
            logger.warning("Failed to generate error_distribution plot: %s", e)

        try:
            p5 = self.plot_residuals(
                pairs=pairs,
                output_path=out_dir / "residual_analysis.png",
            )
            generated_plots["residual_analysis"] = p5
        except Exception as e:
            logger.warning("Failed to generate residual_analysis plot: %s", e)

        if comparison_records:
            try:
                p6 = self.plot_baseline_comparison(
                    comparison_records=comparison_records,
                    output_path=out_dir / "baseline_vs_lstm_comparison.png",
                )
                generated_plots["baseline_comparison"] = p6
            except Exception as e:
                logger.warning("Failed to generate baseline_comparison plot: %s", e)

        return generated_plots

    def generate_task6_visualizations(
        self,
        pairs: List[AlignedEvaluationPair],
        metrics: Dict[str, HorizonEvaluationMetrics],
        comparison_records: Optional[List[Dict[str, Any]]] = None,
        output_dir: Union[str, Path] = "reports/day9/visualizations",
    ) -> Dict[str, Path]:
        """Generate the complete set of 8 Task 6 evaluation visualizations.

        1. Actual vs Predicted Reliability — H1 (actual_vs_predicted_h1.png)
        2. Actual vs Predicted Reliability — H3 (actual_vs_predicted_h3.png)
        3. Actual vs Predicted Reliability — H5 (actual_vs_predicted_h5.png)
        4. MAE comparison across models (mae_comparison_across_models.png)
        5. RMSE comparison across models (rmse_comparison_across_models.png)
        6. Prediction error distribution (prediction_error_distribution.png)
        7. Reliability forecasting timeline (reliability_forecasting_timeline.png)
        8. Horizon-wise performance comparison (horizon_wise_performance_comparison.png)
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        plots: Dict[str, Path] = {}

        if not pairs:
            return plots

        # 1-3. Actual vs Predicted for H1, H3, H5
        for h in [1, 3, 5]:
            try:
                p_h = self.plot_actual_vs_predicted_horizon(
                    pairs=pairs,
                    horizon=h,
                    output_path=out_dir / f"actual_vs_predicted_h{h}.png",
                )
                plots[f"actual_vs_predicted_h{h}"] = p_h
            except Exception as e:
                logger.warning("Failed to generate actual_vs_predicted_h%d: %s", h, e)

        # 4. MAE comparison across models
        if comparison_records:
            try:
                p_mae = self.plot_baseline_comparison(
                    comparison_records=comparison_records,
                    output_path=out_dir / "mae_comparison_across_models.png",
                )
                plots["mae_comparison_across_models"] = p_mae
            except Exception as e:
                logger.warning("Failed to generate mae_comparison_across_models: %s", e)

            # 5. RMSE comparison across models
            try:
                p_rmse = self.plot_rmse_comparison_across_models(
                    comparison_records=comparison_records,
                    output_path=out_dir / "rmse_comparison_across_models.png",
                )
                plots["rmse_comparison_across_models"] = p_rmse
            except Exception as e:
                logger.warning("Failed to generate rmse_comparison_across_models: %s", e)

        # 6. Prediction error distribution
        try:
            p_dist = self.plot_error_distribution(
                pairs=pairs,
                output_path=out_dir / "prediction_error_distribution.png",
            )
            plots["prediction_error_distribution"] = p_dist
        except Exception as e:
            logger.warning("Failed to generate prediction_error_distribution: %s", e)

        # 7. Reliability forecasting timeline
        try:
            p_time = self.plot_reliability_forecasting_timeline(
                pairs=pairs,
                output_path=out_dir / "reliability_forecasting_timeline.png",
            )
            plots["reliability_forecasting_timeline"] = p_time
        except Exception as e:
            logger.warning("Failed to generate reliability_forecasting_timeline: %s", e)

        # 8. Horizon-wise performance comparison
        try:
            p_hw = self.plot_horizon_wise_performance(
                metrics=metrics,
                output_path=out_dir / "horizon_wise_performance_comparison.png",
            )
            plots["horizon_wise_performance_comparison"] = p_hw
        except Exception as e:
            logger.warning("Failed to generate horizon_wise_performance_comparison: %s", e)

        return plots
