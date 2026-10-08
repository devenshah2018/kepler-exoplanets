"""Plotting helpers with a single consistent style for every figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from . import config

NEGATIVE_COLOR = "#C44E52"
POSITIVE_COLOR = "#4C72B0"
CLASS_COLORS = (NEGATIVE_COLOR, POSITIVE_COLOR)
MODEL_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#8172B3", "#937860"]
GRID_KWARGS = {"alpha": 0.3, "linewidth": 0.6}

# Curve legends sit on top of plotted lines, so they get an opaque backing.
LEGEND_ON_PLOT = {
    "frameon": True,
    "framealpha": 0.92,
    "facecolor": "white",
    "edgecolor": "none",
    "fontsize": 8.5,
}


def set_plot_style() -> None:
    """Apply the shared matplotlib settings."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "font.size": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def save_figure(fig: Figure, name: str, directory: Path | None = None) -> Path:
    """Write a figure to the project figure directory and return its path."""
    directory = Path(directory or config.FIGURE_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path)
    return path


# ---------------------------------------------------------------------------
# Exploratory figures
# ---------------------------------------------------------------------------


def plot_class_balance(labels: pd.Series, ax=None) -> Figure:
    """Bar chart of the two dispositions with counts and percentages."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 3.5))

    counts = labels.value_counts().sort_index()
    names = list(config.CLASS_NAMES)
    bars = ax.bar(names, counts.values, color=CLASS_COLORS, width=0.6)

    total = counts.sum()
    for bar, value in zip(bars, counts.values):
        ax.annotate(
            f"{value:,}\n{100 * value / total:.1f}%",
            (bar.get_x() + bar.get_width() / 2, value),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel("Kepler objects of interest")
    ax.set_title("Class balance")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.grid(axis="x", visible=False)
    return ax.figure


def plot_feature_distributions(
    frame: pd.DataFrame,
    features: list[str],
    label_column: str = "label",
    n_cols: int = 3,
    bins: int = 45,
) -> Figure:
    """Overlay per-class histograms for a set of features."""
    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 2.9 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, feature in zip(axes, features):
        values = frame[feature].replace([np.inf, -np.inf], np.nan)
        finite = values.dropna()
        if finite.empty:
            ax.set_visible(False)
            continue

        low, high = np.percentile(finite, [0.5, 99.5])
        edges = np.linspace(low, high, bins)
        for label, color, name in zip((0, 1), CLASS_COLORS, config.CLASS_NAMES):
            subset = values[frame[label_column] == label].dropna()
            ax.hist(
                subset,
                bins=edges,
                color=color,
                alpha=0.55,
                density=True,
                label=name,
            )
        ax.set_xlabel(feature)
        ax.set_yticks([])
        ax.set_ylabel("density")

    for ax in axes[len(features):]:
        ax.set_visible(False)

    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Feature distributions by disposition", y=1.0, fontsize=13)
    fig.tight_layout()
    return fig


def plot_period_depth(frame: pd.DataFrame, label_column: str = "label") -> Figure:
    """Scatter the period-depth plane, the classic view of a transit survey."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for label, color, name in zip((0, 1), CLASS_COLORS, config.CLASS_NAMES):
        subset = frame[frame[label_column] == label]
        ax.scatter(
            subset["koi_period"],
            subset["koi_depth"],
            s=7,
            alpha=0.35,
            color=color,
            label=name,
            edgecolors="none",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Orbital period (days)")
    ax.set_ylabel("Transit depth (ppm)")
    ax.set_title("Transit depth against orbital period")
    ax.legend(markerscale=2.5, loc="upper left")
    return fig


# ---------------------------------------------------------------------------
# Model comparison figures
# ---------------------------------------------------------------------------


def plot_cv_comparison(
    summary: pd.DataFrame,
    metric: str = "roc_auc",
    error_column: str | None = "roc_auc_std",
) -> Figure:
    """Horizontal bars of cross-validated scores with fold-to-fold spread."""
    ordered = summary.sort_values(metric)
    errors = ordered[error_column] if error_column in ordered else None

    fig, ax = plt.subplots(figsize=(7, 0.75 * len(ordered) + 1.6))
    positions = np.arange(len(ordered))
    ax.barh(
        positions,
        ordered[metric],
        xerr=errors,
        color=MODEL_COLORS[: len(ordered)],
        height=0.6,
        capsize=4,
    )

    ax.set_yticks(positions)
    ax.set_yticklabels(ordered.index)
    ax.set_xlabel(metric.replace("_", " ").upper())
    ax.set_title(f"Cross-validated {metric.replace('_', ' ')} by model")

    lower = max(0.5, ordered[metric].min() - 0.06)
    ax.set_xlim(lower, 1.0)
    for position, value in zip(positions, ordered[metric]):
        ax.annotate(
            f"{value:.4f}",
            (value, position),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )
    ax.grid(axis="y", visible=False)
    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    score_map: dict[str, np.ndarray],
    ax=None,
) -> Figure:
    """ROC curves for several models on the same axes."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.6, 5))

    for (name, scores), color in zip(score_map.items(), MODEL_COLORS):
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        ax.plot(fpr, tpr, color=color, linewidth=1.8, label=f"{name} (AUC {auc:.4f})")

    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves on the held-out test set")
    ax.legend(loc="lower right", **LEGEND_ON_PLOT)
    return ax.figure


def plot_pr_curves(
    y_true: np.ndarray,
    score_map: dict[str, np.ndarray],
    ax=None,
) -> Figure:
    """Precision-recall curves for several models on the same axes."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.6, 5))

    for (name, scores), color in zip(score_map.items(), MODEL_COLORS):
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        ax.plot(recall, precision, color=color, linewidth=1.8, label=f"{name} (AP {ap:.4f})")

    baseline = float(np.mean(y_true))
    ax.axhline(baseline, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves on the held-out test set")
    ax.legend(loc="lower left", **LEGEND_ON_PLOT)
    return ax.figure


def plot_confusion_matrices(
    y_true: np.ndarray,
    prediction_map: dict[str, np.ndarray],
    n_cols: int = 4,
) -> Figure:
    """Row-normalized confusion matrices, one panel per model."""
    n_cols = min(n_cols, len(prediction_map))
    n_rows = int(np.ceil(len(prediction_map) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 3.4 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (name, predictions) in zip(axes, prediction_map.items()):
        matrix = confusion_matrix(y_true, predictions, normalize="true")
        display = ConfusionMatrixDisplay(matrix, display_labels=config.CLASS_NAMES)
        display.plot(ax=ax, cmap="Blues", colorbar=False, values_format=".3f")
        ax.set_title(name, fontsize=10)
        ax.grid(visible=False)
        if ax is not axes[0]:
            ax.set_ylabel("")

    for ax in axes[len(prediction_map):]:
        ax.set_visible(False)

    fig.suptitle("Confusion matrices, normalized by true class", y=1.02, fontsize=13)
    fig.tight_layout()
    return fig


def plot_importances(
    importances: pd.Series,
    errors: pd.Series | None = None,
    title: str = "Permutation importance",
    xlabel: str = "Decrease in ROC AUC",
) -> Figure:
    """Horizontal bar chart of feature importances, largest at the top."""
    ordered = importances.sort_values()
    error_values = errors.reindex(ordered.index) if errors is not None else None

    fig, ax = plt.subplots(figsize=(7, 0.32 * len(ordered) + 1.6))
    ax.barh(
        np.arange(len(ordered)),
        ordered.values,
        xerr=error_values,
        color=POSITIVE_COLOR,
        height=0.7,
        capsize=2.5,
    )
    ax.set_yticks(np.arange(len(ordered)))
    ax.set_yticklabels(ordered.index, fontsize=8.5)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def plot_threshold_tradeoff(
    y_true: np.ndarray,
    scores: np.ndarray,
    chosen_threshold: float | None = None,
) -> Figure:
    """Precision, recall, and F1 as a function of the decision threshold."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    precision, recall = precision[:-1], recall[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.nan_to_num(2 * precision * recall / (precision + recall))

    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(thresholds, precision, color=POSITIVE_COLOR, label="Precision")
    ax.plot(thresholds, recall, color=NEGATIVE_COLOR, label="Recall")
    ax.plot(thresholds, f1, color="#55A868", label="F1")

    if chosen_threshold is not None:
        ax.axvline(chosen_threshold, color="grey", linestyle="--", linewidth=1)
        ax.annotate(
            f"threshold = {chosen_threshold:.3f}",
            (chosen_threshold, 0.05),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9,
            color="grey",
        )

    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Operating point selection on out-of-fold predictions")
    ax.legend(loc="lower center", ncols=3)
    return fig


# ---------------------------------------------------------------------------
# Light curve figures
# ---------------------------------------------------------------------------


def plot_light_curve(
    time: np.ndarray,
    raw_flux: np.ndarray,
    flat_flux: np.ndarray,
    title: str,
    window_days: float = 40.0,
) -> Figure:
    """Raw and detrended flux over a short window, stacked for comparison."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.4), sharex=True)

    end = time.min() + window_days
    window = time <= end

    axes[0].plot(time[window], raw_flux[window], color="#444444", linewidth=0.7)
    axes[0].set_ylabel("Normalized flux")
    axes[0].set_title(title)

    axes[1].plot(time[window], flat_flux[window], color=POSITIVE_COLOR, linewidth=0.7)
    axes[1].set_ylabel("Detrended flux")
    axes[1].set_xlabel("Time (BKJD, days)")

    fig.tight_layout()
    return fig


def plot_periodogram(
    period_grid: np.ndarray,
    power: np.ndarray,
    best_period: float,
    catalog_period: float | None = None,
    title: str = "BLS periodogram",
) -> Figure:
    """Periodogram with the recovered peak and its harmonics marked."""
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(period_grid, power, color="#444444", linewidth=0.7)

    ax.axvline(best_period, color=POSITIVE_COLOR, alpha=0.8, linewidth=1.4,
               label=f"BLS peak {best_period:.5f} d")
    for multiple in (2, 3):
        if best_period * multiple <= period_grid.max():
            ax.axvline(best_period * multiple, color=POSITIVE_COLOR, alpha=0.3,
                       linestyle=":", linewidth=1.2)
    if catalog_period is not None:
        ax.axvline(catalog_period, color=NEGATIVE_COLOR, alpha=0.7, linestyle="--",
                   linewidth=1.2, label=f"Archive {catalog_period:.5f} d")

    ax.set_xlabel("Trial period (days)")
    ax.set_ylabel("BLS power")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    return fig


def plot_phase_fold(
    phase: np.ndarray,
    flux: np.ndarray,
    binned_phase: np.ndarray,
    binned_flux: np.ndarray,
    title: str,
    ax=None,
    zoom: float | None = None,
) -> Figure:
    """Folded light curve with binned means drawn over the individual cadences."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4))

    if zoom is not None:
        keep = np.abs(phase) <= zoom
        phase, flux = phase[keep], flux[keep]
        keep_binned = np.abs(binned_phase) <= zoom
        binned_phase, binned_flux = binned_phase[keep_binned], binned_flux[keep_binned]

    ax.scatter(phase, flux, s=1.2, color="#B0B0B0", alpha=0.5, edgecolors="none")
    ax.plot(binned_phase, binned_flux, color=POSITIVE_COLOR, linewidth=1.6)
    ax.set_xlabel("Orbital phase")
    ax.set_ylabel("Detrended flux")
    ax.set_title(title)
    return ax.figure
