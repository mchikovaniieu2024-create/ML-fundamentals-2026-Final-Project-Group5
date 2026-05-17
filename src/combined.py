from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from config import RANDOM_STATE
from data_utils import load_data, clean_data, get_split, build_pipeline


def run_combined_experiments():
    """
    Combined-model experiment:
    - uses from preprocessing load + clean + split
    - uses the shared 'combined' preprocessing pipeline
    - trains Gradient Boosting as the main combined model
    - also trains Logistic Regression
    """
    df = clean_data(load_data())
    X_train, X_val, X_test, y_train, y_val, y_test = get_split(df)

    lr = build_pipeline(
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "combined",
        skills_kind="embeddings",
    )

    gb = build_pipeline(
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        "combined",
        skills_kind="embeddings",
    )

    lr.fit(X_train, y_train)
    gb.fit(X_train, y_train)

    results = {
        "splits": {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        },
        "models": {
            "combined_lr": lr,
            "combined_gb": gb,
        },
        "predictions": {
            "combined_lr": {
                "val_pred": lr.predict(X_val),
                "val_prob": lr.predict_proba(X_val)[:, 1],
                "test_pred": lr.predict(X_test),
                "test_prob": lr.predict_proba(X_test)[:, 1],
            },
            "combined_gb": {
                "val_pred": gb.predict(X_val),
                "val_prob": gb.predict_proba(X_val)[:, 1],
                "test_pred": gb.predict(X_test),
                "test_prob": gb.predict_proba(X_test)[:, 1],
            },
        },
    }

    print("Combined models trained successfully.")
    print("Ready for evaluation.")

    return results

if __name__ == "__main__":
    results = run_combined_experiments()
    for model_name, preds in results["predictions"].items():
        print(model_name, list(preds.keys()))