from sklearn.dummy import DummyClassifier
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


def run_baseline_experiment():
    df = clean_data(load_data())
    X_train, X_val, X_test, y_train, y_val, y_test = get_split(df)

    results = {
        "splits": {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        },
        "models": {},
        "predictions": {},
    }

    zero_rule = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    zero_rule.fit(X_train, y_train)

    lr = build_pipeline(
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "baseline",
    )
    lr.fit(X_train, y_train)

    for model_key, model in {
        "zero_rule": zero_rule,
        "baseline_lr": lr,
    }.items():
        val_pred, val_prob = _safe_predict_and_score(model, X_val)
        test_pred, test_prob = _safe_predict_and_score(model, X_test)

        results["models"][model_key] = model
        results["predictions"][model_key] = {
            "val_pred": val_pred,
            "val_prob": val_prob,
            "test_pred": test_pred,
            "test_prob": test_prob,
        }

    print("Baseline models trained successfully.")
    return results


def main():
    return run_baseline_experiment()


if __name__ == "_main_":
    main()