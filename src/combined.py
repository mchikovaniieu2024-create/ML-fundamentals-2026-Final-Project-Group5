from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from config import RANDOM_STATE
from data_utils import load_data, clean_data, get_split, build_pipeline


def _safe_predict_and_score(model, X):
    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X)
    else:
        y_prob = y_pred.astype(float)

    return y_pred, y_prob


def run_combined_experiments():
    """
    Combined-model experiment using embeddings, but with the combined
    feature matrix precomputed once so embeddings are not recalculated
    for every predict/predict_proba call.
    """
    df = clean_data(load_data())
    X_train, X_val, X_test, y_train, y_val, y_test = get_split(df)

    # Build a combined preprocessor from your existing pipeline helper
    template_pipe = build_pipeline(
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "combined",
        skills_kind="embeddings",
    )
    preprocessor = template_pipe.named_steps["preprocessor"]

    print("Fitting combined preprocessor...")
    X_train_p = preprocessor.fit_transform(X_train, y_train)
    X_val_p = preprocessor.transform(X_val)
    X_test_p = preprocessor.transform(X_test)

    print("Training combined LR...")
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train_p, y_train)

    print("Training combined GB...")
    gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
    gb.fit(X_train_p, y_train)

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
        "predictions": {},
        "preprocessor": preprocessor,
        "X_train_p": X_train_p,
        "X_val_p": X_val_p,
        "X_test_p": X_test_p,
    }

    for model_key, model in results["models"].items():
        val_pred, val_prob = _safe_predict_and_score(model, X_val_p)
        test_pred, test_prob = _safe_predict_and_score(model, X_test_p)

        results["predictions"][model_key] = {
            "val_pred": val_pred,
            "val_prob": val_prob,
            "test_pred": test_pred,
            "test_prob": test_prob,
        }

    print("Combined models trained successfully.")
    print("Ready for evaluation.")

    return results


if __name__ == "__main__":
    run_combined_experiments()