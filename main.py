"""
main.py

This script runs the full ML project end-to-end.

It orchestrates the following project modules:
- baseline.py
- education.py
- experience.py
- skills.py
- combined.py

Supporting modules used by the pipeline:
- config.py
- data_utils.py
- feature_engineering.py
- process_skills.py
- evaluation.py

Experiments are run in this order:
1. Baseline
2. Education
3. Experience
4. Skills
5. Combined

The script combines outputs from all modules and saves the final comparison
table to:

    results/final_model_comparison.csv

This is the final reproducible pipeline for the project.
"""

from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from config import RESULTS_DIR

import baseline
from education import run_education_model
from experience import run_experience_experiments
from skills import run_skills_experiment
from combined import run_combined_experiments


def ensure_raw_data_exists():
    """
    Ensures that the raw CSV file exists where config.py expects it.

    It searches for stackoverflow_full.csv anywhere inside the project folder,
    then copies it to the location expected by config.py:

    """

    project_dir = Path(__file__).resolve().parent

    expected_path = project_dir.parent / "data" / "raw" / "stackoverflow_full.csv"

    if expected_path.exists():
        print(f"Raw data already exists at: {expected_path}")
        return

    possible_paths = [
        project_dir / "data" / "raw" / "stackoverflow_full.csv",
        project_dir / "data" / "processed" / "stackoverflow_full.csv",
        project_dir / "stackoverflow_full.csv",
    ]

    source_path = None

    for path in possible_paths:
        if path.exists():
            source_path = path
            break

    if source_path is None:
        matches = list(project_dir.rglob("stackoverflow_full.csv"))

        if matches:
            source_path = matches[0]
        else:
            raise FileNotFoundError(
                "Could not find stackoverflow_full.csv anywhere inside the project folder."
            )

    expected_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, expected_path)

    print(f"Copied raw data from: {source_path}")
    print(f"Copied raw data to:   {expected_path}")


def evaluate_predictions(y_true, y_pred, y_prob=None):
    """
    Evaluate predictions using common classification metrics.
    """

    scores = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None:
        try:
            scores["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            scores["roc_auc"] = np.nan
    else:
        scores["roc_auc"] = np.nan

    return scores


def add_prediction_results(final_rows, experiment_name, module_results):
    """
    Add results from modules that return predictions.

    This works for:
    - experience.py
    - skills.py
    - combined.py
    """

    y_val = module_results["splits"]["y_val"]
    y_test = module_results["splits"]["y_test"]

    for model_name, preds in module_results["predictions"].items():
        val_scores = evaluate_predictions(
            y_true=y_val,
            y_pred=preds["val_pred"],
            y_prob=preds.get("val_prob"),
        )

        test_scores = evaluate_predictions(
            y_true=y_test,
            y_pred=preds["test_pred"],
            y_prob=preds.get("test_prob"),
        )

        final_rows.append({
            "experiment": experiment_name,
            "model": model_name,

            "val_accuracy": val_scores["accuracy"],
            "val_precision": val_scores["precision"],
            "val_recall": val_scores["recall"],
            "val_f1": val_scores["f1"],
            "val_roc_auc": val_scores["roc_auc"],

            "test_accuracy": test_scores["accuracy"],
            "test_precision": test_scores["precision"],
            "test_recall": test_scores["recall"],
            "test_f1": test_scores["f1"],
            "test_roc_auc": test_scores["roc_auc"],
        })


def add_education_results(final_rows, education_scores):
    """
    Add results from education.py.

    education.py returns only validation_accuracy and test_accuracy,
    so other metrics are left empty.
    """

    for model_name, scores in education_scores.items():
        final_rows.append({
            "experiment": "education",
            "model": model_name,

            "val_accuracy": scores["validation_accuracy"],
            "val_precision": np.nan,
            "val_recall": np.nan,
            "val_f1": np.nan,
            "val_roc_auc": np.nan,

            "test_accuracy": scores["test_accuracy"],
            "test_precision": np.nan,
            "test_recall": np.nan,
            "test_f1": np.nan,
            "test_roc_auc": np.nan,
        })


def save_final_results(final_rows):
    """
    Save all experiment results into one final CSV file.
    """

    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    final_df = pd.DataFrame(final_rows)

    final_df = final_df.sort_values(
        by=["test_f1", "test_accuracy"],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    output_path = results_dir / "final_model_comparison.csv"
    final_df.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print("FINAL MODEL COMPARISON")
    print("=" * 80)
    print(final_df)
    print("=" * 80)

    print(f"\nFinal results saved to: {output_path}")

    return final_df


def main():
    """
    Run the full ML project pipeline.
    """

    ensure_raw_data_exists()

    final_rows = []

    print("\n" + "=" * 80)
    print("1. RUNNING BASELINE EXPERIMENT")
    print("=" * 80)

    baseline.main()

    print("\n" + "=" * 80)
    print("2. RUNNING EDUCATION EXPERIMENT")
    print("=" * 80)

    education_scores = run_education_model()
    add_education_results(final_rows, education_scores)

    print("\n" + "=" * 80)
    print("3. RUNNING EXPERIENCE EXPERIMENT")
    print("=" * 80)

    experience_results = run_experience_experiments()
    add_prediction_results(final_rows, "experience", experience_results)

    print("\n" + "=" * 80)
    print("4. RUNNING SKILLS EXPERIMENT")
    print("=" * 80)

    skills_results = run_skills_experiment()
    add_prediction_results(final_rows, "skills", skills_results)

    print("\n" + "=" * 80)
    print("5. RUNNING COMBINED EXPERIMENT")
    print("=" * 80)

    combined_results = run_combined_experiments()
    add_prediction_results(final_rows, "combined", combined_results)

    final_df = save_final_results(final_rows)

    return final_df


if __name__ == "__main__":
    main()