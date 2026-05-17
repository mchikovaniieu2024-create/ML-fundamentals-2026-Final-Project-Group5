import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

from config import RESULTS_DIR

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.1)

PALETTE = {
    "Baseline (Zero-Rule)":   "#9e9e9e",
    "Education — LR":         "#42a5f5",
    "Education — RF":         "#1565c0",
    "Experience — LR":        "#66bb6a",
    "Experience — RF":        "#2e7d32",
    "Skills — LR":            "#ffa726",
    "Skills — RF":            "#e65100",
    "Skills — Embeddings LR": "#ff7043",
    "Skills — Embeddings RF": "#bf360c",
    "Combined — XGBoost":     "#ab47bc",
    "Combined — LR":          "#6a1b9a",
}

DEFAULT_COLOR = "#607d8b"


def evaluate_model(pipeline, X_test, y_test, model_name: str) -> dict:
    y_pred = pipeline.predict(X_test)

    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    elif hasattr(pipeline, "decision_function"):
        y_proba = pipeline.decision_function(X_test)
    else:
        y_proba = y_pred.astype(float)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred,    zero_division=0)
    f1        = f1_score(y_test, y_pred,        zero_division=0)
    roc_auc   = roc_auc_score(y_test, y_proba)

    print(
        f"[evaluate_model] {model_name:<30} | "
        f"Precision: {precision:.3f} | "
        f"Recall: {recall:.3f} | "
        f"F1: {f1:.3f} | "
        f"ROC-AUC: {roc_auc:.3f}"
    )

    return {
        "model_name": model_name,
        "precision":  precision,
        "recall":     recall,
        "f1":         f1,
        "roc_auc":    roc_auc,
        "y_pred":     y_pred,
        "y_proba":    y_proba,
    }


def build_comparison_table(results: dict) -> pd.DataFrame:
    rows = []
    for info in results.values():
        rows.append({
            "Model":     info["model_name"],
            "Precision": round(info["precision"], 4),
            "Recall":    round(info["recall"],    4),
            "F1":        round(info["f1"],        4),
            "ROC-AUC":   round(info["roc_auc"],   4),
        })

    table = (
        pd.DataFrame(rows)
        .sort_values("ROC-AUC", ascending=False)
        .reset_index(drop=True)
    )

    csv_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    table.to_csv(csv_path, index=False)
    print(f"[build_comparison_table] Saved → {csv_path}")

    print("\n" + "=" * 62)
    print("MODEL COMPARISON TABLE (sorted by ROC-AUC)")
    print("=" * 62)
    print(table.to_string(index=False))
    print("=" * 62 + "\n")

    return table


def _save_and_show(fig, filename: str) -> None:
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved → {path}")
    plt.show()
    plt.close(fig)


def plot_roc_curves(results: dict, y_test) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))

    ax.plot([0, 1], [0, 1], linestyle="--", color="#bdbdbd",
            linewidth=1.2, label="Random chance (AUC = 0.50)")

    for info in results.values():
        name    = info["model_name"]
        y_proba = info["y_proba"]
        auc     = info["roc_auc"]
        color   = PALETTE.get(name, DEFAULT_COLOR)

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name}  (AUC = {auc:.3f})")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    _save_and_show(fig, "roc_curves.png")


def plot_precision_recall_curves(results: dict, y_test) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))

    baseline_ap = float(np.mean(y_test))
    ax.axhline(y=baseline_ap, linestyle="--", color="#bdbdbd",
               linewidth=1.2, label=f"No-skill baseline (AP = {baseline_ap:.3f})")

    for info in results.values():
        name    = info["model_name"]
        y_proba = info["y_proba"]
        color   = PALETTE.get(name, DEFAULT_COLOR)

        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)

        ax.plot(recall_vals, precision_vals, color=color, linewidth=2,
                label=f"{name}  (AP = {ap:.3f})")

    ax.set_xlabel("Recall",    fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    _save_and_show(fig, "precision_recall_curves.png")


def plot_confusion_matrices(results: dict, y_test) -> None:
    n_models = len(results)
    n_cols   = min(3, n_models)
    n_rows   = int(np.ceil(n_models / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.5 * n_cols, 4.5 * n_rows),
        squeeze=False,
    )

    for idx, info in enumerate(results.values()):
        row = idx // n_cols
        col = idx % n_cols
        ax  = axes[row][col]

        name   = info["model_name"]
        y_pred = info["y_pred"]
        cm     = confusion_matrix(y_test, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        sns.heatmap(cm_norm, annot=False, cmap="Blues",
                    vmin=0, vmax=1, linewidths=0.5, ax=ax, cbar=False)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j + 0.5, i + 0.5,
                    f"{cm_norm[i, j]:.2f}\n({cm[i, j]:,})",
                    ha="center", va="center", fontsize=10,
                    color="white" if cm_norm[i, j] > 0.6 else "black",
                )

        ax.set_title(name, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Predicted label", fontsize=10)
        ax.set_ylabel("True label",      fontsize=10)
        ax.set_xticklabels(["Not Employed (0)", "Employed (1)"], fontsize=9)
        ax.set_yticklabels(["Not Employed (0)", "Employed (1)"], fontsize=9, rotation=0)

    for idx in range(n_models, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle("Confusion Matrices — All Models (row-normalised)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    _save_and_show(fig, "confusion_matrices.png")


def plot_feature_importance(pipeline, model_name: str, feature_names: list) -> None:
    model = pipeline.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        print(f"[plot_feature_importance] Skipped '{model_name}' — no feature_importances_")
        return

    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1]
    top_n       = min(20, len(importances))
    indices     = indices[:top_n]

    labels = [feature_names[i] if i < len(feature_names)
               else f"feature_{i}" for i in indices]
    values = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 0.4 * top_n + 2))

    color = PALETTE.get(model_name, DEFAULT_COLOR)
    ax.barh(range(top_n), values[::-1], color=color, edgecolor="white", height=0.7)

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels[::-1], fontsize=9)
    ax.set_xlabel("Feature Importance (Gini)", fontsize=11)
    ax.set_title(f"Feature Importance — {model_name} (Top {top_n})",
                 fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_").replace("—", "").replace("/", "_")
    _save_and_show(fig, f"feature_importance_{safe_name}.png")


def plot_metric_comparison_bar(results: dict) -> None:
    model_names = [info["model_name"] for info in results.values()]
    f1_scores   = [info["f1"]         for info in results.values()]
    auc_scores  = [info["roc_auc"]    for info in results.values()]

    x     = np.arange(len(model_names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(10, len(model_names) * 1.4), 6))

    bars_f1  = ax.bar(x - width / 2, f1_scores,  width, label="F1",      color="#42a5f5", edgecolor="white")
    bars_auc = ax.bar(x + width / 2, auc_scores, width, label="ROC-AUC", color="#ab47bc", edgecolor="white")

    for bar in bars_f1:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=8)

    for bar in bars_auc:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim([0, 1.12])
    ax.set_title("F1 and ROC-AUC Comparison — All Models",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    plt.tight_layout()
    _save_and_show(fig, "metric_comparison_bar.png")


def plot_all(results: dict, y_test) -> None:
    print("\n[plot_all] Generating all evaluation plots...")
    plot_roc_curves(results, y_test)
    plot_precision_recall_curves(results, y_test)
    plot_confusion_matrices(results, y_test)
    plot_metric_comparison_bar(results)
    print("[plot_all] Done. All plots saved to results/\n")