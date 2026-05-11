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

from skills import run_skills_experiment
from config import RESULTS_DIR

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.1)

PALETTE = {
    "binary_lr":     "#42a5f5",
    "binary_rf":     "#1565c0",
    "counts_lr":     "#66bb6a",
    "counts_rf":     "#2e7d32",
    "grouped_lr":    "#ffa726",
    "grouped_rf":    "#e65100",
    "embeddings_lr": "#ab47bc",
    "embeddings_rf": "#6a1b9a",
}

DISPLAY_NAMES = {
    "binary_lr":     "Binary — LR",
    "binary_rf":     "Binary — RF",
    "counts_lr":     "Counts — LR",
    "counts_rf":     "Counts — RF",
    "grouped_lr":    "Grouped — LR",
    "grouped_rf":    "Grouped — RF",
    "embeddings_lr": "Embeddings — LR",
    "embeddings_rf": "Embeddings — RF",
}


# 1.EVALUATE

def evaluate_skills_models(results: dict) -> dict:
    """
    Compute Precision, Recall, F1, and ROC-AUC for all skills models.

    Parameters
    ----------
    results : dict returned by run_skills_experiment()

    Returns
    -------
    dict mapping model_key → metrics dict
    """
    y_test      = results["splits"]["y_test"]
    predictions = results["predictions"]
    evaluated   = {}

    for model_key, preds in predictions.items():
        y_pred  = preds["test_pred"]
        y_proba = preds["test_prob"]

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score(y_test, y_pred,    zero_division=0)
        f1        = f1_score(y_test, y_pred,        zero_division=0)
        roc_auc   = roc_auc_score(y_test, y_proba)

        display = DISPLAY_NAMES.get(model_key, model_key)

        evaluated[model_key] = {
            "model_key":  model_key,
            "model_name": display,
            "precision":  precision,
            "recall":     recall,
            "f1":         f1,
            "roc_auc":    roc_auc,
            "y_pred":     y_pred,
            "y_proba":    y_proba,
        }

        print(
            f"[evaluate] {display:<22} | "
            f"Precision: {precision:.3f} | "
            f"Recall: {recall:.3f} | "
            f"F1: {f1:.3f} | "
            f"ROC-AUC: {roc_auc:.3f}"
        )

    return evaluated


# 2. COMPARISON TABLE

def build_comparison_table(evaluated: dict) -> pd.DataFrame:
    """
    Build and save a model comparison table sorted by ROC-AUC.
    Saved to results/skills_model_comparison.csv.
    """
    rows = []
    for info in evaluated.values():
        rows.append({
            "Model":          info["model_name"],
            "Representation": info["model_key"].split("_")[0].capitalize(),
            "Classifier":     info["model_key"].split("_")[1].upper(),
            "Precision":      round(info["precision"], 4),
            "Recall":         round(info["recall"],    4),
            "F1":             round(info["f1"],        4),
            "ROC-AUC":        round(info["roc_auc"],   4),
        })

    table = (
        pd.DataFrame(rows)
        .sort_values("ROC-AUC", ascending=False)
        .reset_index(drop=True)
    )

    csv_path = os.path.join(RESULTS_DIR, "skills_model_comparison.csv")
    table.to_csv(csv_path, index=False)
    print(f"\n[table] Saved → {csv_path}")

    print("\n" + "=" * 72)
    print("SKILLS MODEL COMPARISON (sorted by ROC-AUC)")
    print("=" * 72)
    print(table.to_string(index=False))
    print("=" * 72 + "\n")

    return table


# 3. PLOTS

def _save_and_show(fig, filename: str) -> None:
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved → {path}")
    plt.show()
    plt.close(fig)


def plot_roc_curves(evaluated: dict, y_test) -> None:
    """ROC curves for all skills models on one plot."""
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot([0, 1], [0, 1], linestyle="--", color="#bdbdbd",
            linewidth=1.2, label="Random chance (AUC = 0.50)")

    for model_key, info in evaluated.items():
        fpr, tpr, _ = roc_curve(y_test, info["y_proba"])
        color = PALETTE.get(model_key, "#607d8b")
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{info['model_name']}  (AUC = {info['roc_auc']:.3f})")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curves — Skills Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    _save_and_show(fig, "skills_roc_curves.png")


def plot_precision_recall_curves(evaluated: dict, y_test) -> None:
    """Precision-Recall curves for all skills models."""
    fig, ax = plt.subplots(figsize=(10, 7))

    baseline_ap = float(np.mean(y_test))
    ax.axhline(y=baseline_ap, linestyle="--", color="#bdbdbd",
               linewidth=1.2, label=f"No-skill baseline (AP = {baseline_ap:.3f})")

    for model_key, info in evaluated.items():
        prec_vals, rec_vals, _ = precision_recall_curve(y_test, info["y_proba"])
        ap    = average_precision_score(y_test, info["y_proba"])
        color = PALETTE.get(model_key, "#607d8b")
        ax.plot(rec_vals, prec_vals, color=color, linewidth=2,
                label=f"{info['model_name']}  (AP = {ap:.3f})")

    ax.set_xlabel("Recall",    fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — Skills Models", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    _save_and_show(fig, "skills_pr_curves.png")


def plot_confusion_matrices(evaluated: dict, y_test) -> None:
    """Grid of confusion matrices for all skills models."""
    n_models = len(evaluated)
    n_cols   = 4
    n_rows   = int(np.ceil(n_models / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.5 * n_cols, 4.5 * n_rows),
        squeeze=False,
    )

    for idx, (model_key, info) in enumerate(evaluated.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax  = axes[row][col]

        cm      = confusion_matrix(y_test, info["y_pred"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        sns.heatmap(cm_norm, annot=False, cmap="Blues",
                    vmin=0, vmax=1, linewidths=0.5, ax=ax, cbar=False)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j + 0.5, i + 0.5,
                    f"{cm_norm[i,j]:.2f}\n({cm[i,j]:,})",
                    ha="center", va="center", fontsize=9,
                    color="white" if cm_norm[i, j] > 0.6 else "black",
                )

        ax.set_title(info["model_name"], fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True",      fontsize=9)
        ax.set_xticklabels(["Not Emp.", "Emp."], fontsize=8)
        ax.set_yticklabels(["Not Emp.", "Emp."], fontsize=8, rotation=0)

    for idx in range(n_models, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle("Confusion Matrices — Skills Models (row-normalised)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_and_show(fig, "skills_confusion_matrices.png")


def plot_metric_comparison_bar(evaluated: dict) -> None:
    """Grouped bar chart comparing F1 and ROC-AUC across all skills models."""
    model_names = [info["model_name"] for info in evaluated.values()]
    f1_scores   = [info["f1"]         for info in evaluated.values()]
    auc_scores  = [info["roc_auc"]    for info in evaluated.values()]

    x     = np.arange(len(model_names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 6))

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
    ax.set_xticklabels(model_names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim([0, 1.12])
    ax.set_title("F1 and ROC-AUC — Skills Models Comparison",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    plt.tight_layout()
    _save_and_show(fig, "skills_metric_bar.png")


# 4. MAIN

if __name__ == "__main__":
    print("=" * 60)
    print("SKILLS EXPERIMENT EVALUATION")
    print("=" * 60)

    # Step 1: Train all skills models
    print("\n[Step 1] Running skills experiment...")
    results = run_skills_experiment()
    y_test  = results["splits"]["y_test"]

    # Step 2: Evaluate all models
    print("\n[Step 2] Evaluating all skills models on test set...")
    evaluated = evaluate_skills_models(results)

    # Step 3: Build comparison table
    print("\n[Step 3] Building comparison table...")
    table = build_comparison_table(evaluated)

    # Step 4: Generate plots
    print("\n[Step 4] Generating plots...")
    plot_roc_curves(evaluated, y_test)
    plot_precision_recall_curves(evaluated, y_test)
    plot_confusion_matrices(evaluated, y_test)
    plot_metric_comparison_bar(evaluated)

    print("\nDone! All results saved to results/")