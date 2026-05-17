import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.model_selection import learning_curve
from sklearn.base import clone

from sklearn.metrics import (
    accuracy_score,
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

_EDA = {
    "pos":  "#4C72B0",
    "neg":  "#DD8452",
    "miss": "#E84040",
    "grid": "#E5E5E5",
}


def _prefixed_filename(filename: str, experiment_name: str | None = None) -> str:
    if not experiment_name:
        return filename
    safe = experiment_name.lower().replace(" ", "_").replace("—", "-")
    return f"{safe}_{filename}"


def _save_and_show(fig: plt.Figure, filename: str, experiment_name: str | None = None) -> None:
    """Save to RESULTS_DIR and display."""
    path = os.path.join(RESULTS_DIR, _prefixed_filename(filename, experiment_name))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved → {path}")
    plt.show()
    plt.close(fig)


def _save_or_show(fig: plt.Figure, save_path: str | None) -> None:
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[eda_plots] Saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


def _style_ax(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(axis="y", color=_EDA["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def evaluate_model(pipeline, X_test, y_test, model_name: str) -> dict:
    y_pred = pipeline.predict(X_test)

    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    elif hasattr(pipeline, "decision_function"):
        y_proba = pipeline.decision_function(X_test)
    else:
        y_proba = y_pred.astype(float)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(
        f"[evaluate_model] {model_name:<30} | "
        f"Accuracy: {accuracy:.3f} | "
        f"Precision: {precision:.3f} | "
        f"Recall: {recall:.3f} | "
        f"F1: {f1:.3f} | "
        f"ROC-AUC: {roc_auc:.3f}"
    )

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def build_comparison_table(results: dict, experiment_name: str | None = None) -> pd.DataFrame:
    rows = []
    for info in results.values():
        rows.append({
            "Model": info["model_name"],
            "Accuracy": round(info["accuracy"], 4),
            "Precision": round(info["precision"], 4),
            "Recall": round(info["recall"], 4),
            "F1": round(info["f1"], 4),
            "ROC-AUC": round(info["roc_auc"], 4),
        })

    table = (
        pd.DataFrame(rows)
        .sort_values("ROC-AUC", ascending=False)
        .reset_index(drop=True)
    )

    csv_name = _prefixed_filename("model_comparison.csv", experiment_name)
    csv_path = os.path.join(RESULTS_DIR, csv_name)
    table.to_csv(csv_path, index=False)
    print(f"[build_comparison_table] Saved → {csv_path}")

    print("\n" + "=" * 62)
    print("MODEL COMPARISON TABLE (sorted by ROC-AUC)")
    print("=" * 62)
    print(table.to_string(index=False))
    print("=" * 62 + "\n")

    return table


def build_summary_table(results: dict, experiment_name: str) -> pd.DataFrame:
    y_val = results["splits"]["y_val"]
    y_test = results["splits"]["y_test"]

    rows = []
    for model_key, preds in results["predictions"].items():
        rows.append({
            "Experiment": experiment_name.capitalize(),
            "Model": model_key,

            "Val Acc": round(accuracy_score(y_val, preds["val_pred"]), 3),
            "Val Prec": round(precision_score(y_val, preds["val_pred"], zero_division=0), 3),
            "Val Rec": round(recall_score(y_val, preds["val_pred"], zero_division=0), 3),
            "Val F1": round(f1_score(y_val, preds["val_pred"], zero_division=0), 3),
            "Val AUC": round(roc_auc_score(y_val, preds["val_prob"]), 3),

            "Test Acc": round(accuracy_score(y_test, preds["test_pred"]), 3),
            "Test Prec": round(precision_score(y_test, preds["test_pred"], zero_division=0), 3),
            "Test Rec": round(recall_score(y_test, preds["test_pred"], zero_division=0), 3),
            "Test F1": round(f1_score(y_test, preds["test_pred"], zero_division=0), 3),
            "Test AUC": round(roc_auc_score(y_test, preds["test_prob"]), 3),
        })

    table = (
        pd.DataFrame(rows)
        .sort_values(["Test AUC", "Val AUC"], ascending=False)
        .reset_index(drop=True)
    )

    csv_path = os.path.join(RESULTS_DIR, f"{experiment_name}_summary.csv")
    table.to_csv(csv_path, index=False)
    print(f"[build_summary_table] Saved → {csv_path}")

    print(table.to_string(index=False))
    return table



def plot_roc_curves(results: dict, y_test, experiment_name: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))

    ax.plot([0, 1], [0, 1], linestyle="--", color="#bdbdbd",
            linewidth=1.2, label="Random chance (AUC = 0.50)")

    for info in results.values():
        name = info["model_name"]
        y_proba = info["y_proba"]
        auc = info["roc_auc"]
        color = PALETTE.get(name, DEFAULT_COLOR)

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name}  (AUC = {auc:.3f})")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    _save_and_show(fig, "roc_curves.png", experiment_name)


def plot_precision_recall_curves(results: dict, y_test, experiment_name: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))

    baseline_ap = float(np.mean(y_test))
    ax.axhline(y=baseline_ap, linestyle="--", color="#bdbdbd",
               linewidth=1.2, label=f"No-skill baseline (AP = {baseline_ap:.3f})")

    for info in results.values():
        name = info["model_name"]
        y_proba = info["y_proba"]
        color = PALETTE.get(name, DEFAULT_COLOR)

        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)

        ax.plot(recall_vals, precision_vals, color=color, linewidth=2,
                label=f"{name}  (AP = {ap:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    _save_and_show(fig, "precision_recall_curves.png", experiment_name)


def plot_confusion_matrices(results: dict, y_test, experiment_name: str | None = None) -> None:
    n_models = len(results)
    n_cols = min(3, n_models)
    n_rows = int(np.ceil(n_models / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.5 * n_cols, 4.5 * n_rows),
        squeeze=False,
    )

    for idx, info in enumerate(results.values()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]

        name = info["model_name"]
        y_pred = info["y_pred"]
        cm = confusion_matrix(y_test, y_pred)
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
        ax.set_ylabel("True label", fontsize=10)
        ax.set_xticklabels(["Not Employed (0)", "Employed (1)"], fontsize=9)
        ax.set_yticklabels(["Not Employed (0)", "Employed (1)"], fontsize=9, rotation=0)

    for idx in range(n_models, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle("Confusion Matrices (row-normalised)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    _save_and_show(fig, "confusion_matrices.png", experiment_name)

def plot_metric_comparison_bar(results: dict, experiment_name: str | None = None) -> None:
    model_names = [info["model_name"] for info in results.values()]
    f1_scores = [info["f1"] for info in results.values()]
    auc_scores = [info["roc_auc"] for info in results.values()]

    x = np.arange(len(model_names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(10, len(model_names) * 1.4), 6))

    bars_f1 = ax.bar(x - width / 2, f1_scores, width, label="F1", color="#42a5f5", edgecolor="white")
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
    ax.set_title("F1 and ROC-AUC Comparison",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    plt.tight_layout()
    _save_and_show(fig, "metric_comparison_bar.png", experiment_name)


def plot_target_distribution(
    y,
    labels: dict | None = None,
    title: str = "Target Distribution",
    save_path: str | None = None,
) -> None:
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    total = counts.sum()

    if labels is None:
        labels = {c: f"Class {c}" for c in classes}

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [_EDA["neg"], _EDA["pos"]]

    bars = ax.bar(
        [labels.get(c, str(c)) for c in classes],
        counts,
        color=colors[: len(classes)],
        width=0.5,
        edgecolor="white",
        linewidth=1.2,
        zorder=3,
    )

    for bar, count in zip(bars, counts):
        pct = 100 * count / total
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.005,
            f"{count:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    _style_ax(ax, title, ylabel="Count")
    ax.set_ylim(0, max(counts) * 1.18)
    _save_or_show(fig, save_path)

def plot_correlation_heatmap(
    df,
    method: str = "pearson",
    title: str = "Correlation Heatmap",
    max_features: int = 30,
    save_path: str | None = None,
) -> None:
    numeric = df.select_dtypes(include=[np.number])

    if numeric.shape[1] > max_features:
        top_cols = numeric.var().nlargest(max_features).index
        numeric = numeric[top_cols]

    corr = numeric.corr(method=method)
    n = corr.shape[0]
    fig_size = max(8, n * 0.55)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr,
        mask=mask,
        annot=n <= 20,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.4,
        linecolor="white",
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.75, "label": f"{method.capitalize()} r"},
        annot_kws={"size": 7},
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    _save_or_show(fig, save_path)


def plot_learning_curves(
    model,
    X_train,
    y_train,
    cv: int = 5,
    scoring: str = "f1",
    train_sizes: np.ndarray | None = None,
    title: str = "Learning Curves",
    save_path: str | None = None,
) -> None:
    if train_sizes is None:
        train_sizes = np.linspace(0.10, 1.0, 10)

    sizes, train_scores, val_scores = learning_curve(
        model,
        X_train,
        y_train,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        shuffle=True,
        random_state=42,
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(sizes, train_mean, "o-", color=_EDA["pos"], label="Train", linewidth=2, markersize=5)
    ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color=_EDA["pos"])

    ax.plot(sizes, val_mean, "s--", color=_EDA["neg"], label=f"CV ({cv}-fold)", linewidth=2, markersize=5)
    ax.fill_between(sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color=_EDA["neg"])

    ax.legend(fontsize=10)
    _style_ax(ax, title, xlabel="Training samples", ylabel=scoring.replace("_", " ").title())
    ax.set_xlim(sizes[0] * 0.95, sizes[-1] * 1.02)
    _save_or_show(fig, save_path)


def plot_all(
    results: dict,
    y_test,
    y_train=None,
    X_train=None,
    X_train_lc=None,
    X_test=None,
    df_raw=None,
    pipelines: dict | None = None,
    feature_names: list | None = None,
    lc_model=None,
    lc_scoring: str = "f1",
    lc_cv: int = 5,
    experiment_name: str | None = None,
) -> None:

    print("\n[plot_all] ── Evaluation plots ─────────────────────────────")

    plot_roc_curves(results, y_test, experiment_name=experiment_name)
    plot_precision_recall_curves(results, y_test, experiment_name=experiment_name)
    plot_confusion_matrices(results, y_test, experiment_name=experiment_name)
    plot_metric_comparison_bar(results, experiment_name=experiment_name)


    if X_train is not None:
        if not isinstance(X_train, pd.DataFrame):
            cols = feature_names or [f"f{i}" for i in range(X_train.shape[1])]
            X_train_df = pd.DataFrame(X_train, columns=cols)
        else:
            X_train_df = X_train
    else:
        X_train_df = None

    df_eda = df_raw if df_raw is not None else X_train_df

    if y_train is not None and experiment_name == "combined":
        plot_target_distribution(
            y_train,
            labels={0: "Not Employed", 1: "Employed"},
            save_path=os.path.join(
                RESULTS_DIR,
                _prefixed_filename("target_distribution.png", experiment_name),
            ),
        )

    if df_eda is not None and experiment_name == "combined":
        plot_correlation_heatmap(
            df_eda,
            save_path=os.path.join(
                RESULTS_DIR,
                _prefixed_filename("correlation_heatmap.png", experiment_name),
            ),
        )

    X_lc = X_train_lc if X_train_lc is not None else X_train_df

    if experiment_name in {"experience",
                           "combined"} and lc_model is not None and X_lc is not None and y_train is not None:
        plot_learning_curves(
            clone(lc_model),
            X_lc,
            y_train,
            cv=lc_cv,
            scoring=lc_scoring,
            save_path=os.path.join(
                RESULTS_DIR,
                _prefixed_filename("learning_curves.png", experiment_name),
            ),
        )
    else:
        print("[plot_all] 9 — Learning curves skipped.")



    print("\n[plot_all] Done. All plots saved to results/\n")

