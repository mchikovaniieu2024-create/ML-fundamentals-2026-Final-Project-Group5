from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from config import RANDOM_STATE
from data_utils import load_data, clean_data, get_split, build_pipeline


def run_experience_experiments():
    """
    Experience-only experiment:
    - uses Massimo's raw load + cleaning + split
    - trains Logistic Regression and Random Forest
    - keeps preprocessing inside the shared pipeline
    """
    df = clean_data(load_data())

    X_train, X_val, X_test, y_train, y_val, y_test = get_split(df)

    lr = build_pipeline(
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "experience",
    )

    rf = build_pipeline(
        RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "experience",
    )

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    lr_val_pred = lr.predict(X_val)
    lr_val_prob = lr.predict_proba(X_val)[:, 1]

    rf_val_pred = rf.predict(X_val)
    rf_val_prob = rf.predict_proba(X_val)[:, 1]

    lr_test_pred = lr.predict(X_test)
    lr_test_prob = lr.predict_proba(X_test)[:, 1]

    rf_test_pred = rf.predict(X_test)
    rf_test_prob = rf.predict_proba(X_test)[:, 1]

    print("Experience-only models trained successfully.")
    print("Logistic Regression and Random Forest are ready for evaluation.")

    return {
        "splits": {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        },
        "models": {
            "logistic_regression": lr,
            "random_forest": rf,
        },
        "predictions": {
            "logistic_regression": {
                "val_pred": lr_val_pred,
                "val_prob": lr_val_prob,
                "test_pred": lr_test_pred,
                "test_prob": lr_test_prob,
            },
            "random_forest": {
                "val_pred": rf_val_pred,
                "val_prob": rf_val_prob,
                "test_pred": rf_test_pred,
                "test_prob": rf_test_prob,
            },
        },
    }


if __name__ == "__main__":
    run_experience_experiments()