from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from config import RANDOM_STATE
from data_utils import load_data, clean_data, get_split, build_pipeline


def run_skills_experiment():
    df = clean_data(load_data())
    X_train, X_val, X_test, y_train, y_val, y_test = get_split(df)

    skill_kinds = ["binary", "counts", "grouped", "embeddings"]

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

    for kind in skill_kinds:
        lr = build_pipeline(
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            "skills",
            skills_kind=kind,
        )

        rf = build_pipeline(
            RandomForestClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "skills",
            skills_kind=kind,
        )

        lr.fit(X_train, y_train)
        rf.fit(X_train, y_train)

        results["models"][f"{kind}_lr"] = lr
        results["models"][f"{kind}_rf"] = rf

        results["predictions"][f"{kind}_lr"] = {
            "val_pred": lr.predict(X_val),
            "val_prob": lr.predict_proba(X_val)[:, 1],
            "test_pred": lr.predict(X_test),
            "test_prob": lr.predict_proba(X_test)[:, 1],
        }

        results["predictions"][f"{kind}_rf"] = {
            "val_pred": rf.predict(X_val),
            "val_prob": rf.predict_proba(X_val)[:, 1],
            "test_pred": rf.predict(X_test),
            "test_prob": rf.predict_proba(X_test)[:, 1],
        }

    print("Skills models trained successfully.")
    print("Logistic Regression and Random Forest are ready for evaluation.")

    return results


if __name__ == "__main__":
    results = run_skills_experiment()