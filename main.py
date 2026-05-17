import os
import sys
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from baseline import main as run_baseline
from education import run_education_model
from experience import run_experience_experiments
from skills import run_skills_experiment
from combined import run_combined_experiments

from evaluation import (
    build_comparison_table,
    build_summary_table,
    plot_all,
)


def evaluate_results(results: dict) -> dict:
    y_test = results["splits"]["y_test"]
    evaluated = {}

    for model_key, preds in results["predictions"].items():
        y_pred = preds["test_pred"]
        y_proba = preds["test_prob"]

        evaluated[model_key] = {
            "model_key": model_key,
            "model_name": model_key,
            "accuracy": (y_test == y_pred).mean(),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "y_pred": y_pred,
            "y_proba": y_proba,
        }

    return evaluated


def show_all(results: dict, title: str):
    evaluated = evaluate_results(results)
    if not evaluated:
        return None

    print(f"\n[{title}] Evaluation")
    build_comparison_table(evaluated, experiment_name=title)
    build_summary_table(results, title)

    splits = results["splits"]
    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_test = splits["X_test"]
    y_test = splits["y_test"]

    feature_names = list(X_train.columns) if isinstance(X_train, pd.DataFrame) else None
    pipelines = results.get("models", {})

    if title == "combined":
        lc_model = pipelines["combined_lr"]
        X_train_lc = results["X_train_p"]
    else:
        lc_model = next((m for m in pipelines.values() if hasattr(m, "named_steps")), None)
        X_train_lc = X_train

    plot_all(
        results=evaluated,
        y_test=y_test,
        y_train=y_train,
        X_train=X_train,
        X_train_lc=X_train_lc,
        X_test=X_test,
        pipelines=pipelines,
        feature_names=feature_names,
        lc_model=lc_model,
        experiment_name=title,
    )

    return evaluated


def main():
    for title, fn in [
        ("baseline", run_baseline),
        ("education", run_education_model),
        ("experience", run_experience_experiments),
        ("skills", run_skills_experiment),
        ("combined", run_combined_experiments),
    ]:
        results = fn()
        show_all(results, title)


if __name__ == "__main__":
    main()