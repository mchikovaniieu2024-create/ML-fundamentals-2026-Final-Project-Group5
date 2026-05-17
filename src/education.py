from sklearn.ensemble import RandomForestClassifier
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


def run_education_model():
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

    edu_lr = build_pipeline(
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "education",
    )

    edu_rf = build_pipeline(
        RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "education",
    )

    edu_lr.fit(X_train, y_train)
    edu_rf.fit(X_train, y_train)

    for model_key, model in {
        "education_lr": edu_lr,
        "education_rf": edu_rf,
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

    print("Education models trained successfully.")
    return results


def main():
    return run_education_model()


if __name__ == "_main_":
    main()