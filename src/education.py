from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from data_utils import load_data, clean_data, get_split, build_pipeline

def get_metrics(model, X, y):
    prediction_y = model.predict(X)
    probabilities_y = model.predict_proba(X)[:, 1]

    return {
        "acc": accuracy_score(y, prediction_y),
        "prec": precision_score(y, prediction_y, zero_division=0),
        "rec": recall_score(y, prediction_y, zero_division=0),
        "f1": f1_score(y, prediction_y, zero_division=0),
        "auc": roc_auc_score(y, probabilities_y),
    }
def run_education_model():
    df = load_data()
    df = clean_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = get_split(df)

    scores = {}

# Logistic Regression
    logistic_reg_model = LogisticRegression(max_iter=300, random_state=42)
    logistic_reg_pipe = build_pipeline(logistic_reg_model, "education")
    logistic_reg_pipe.fit(X_train, y_train)
    logistic_reg_val_metrics = get_metrics(logistic_reg_pipe, X_val, y_val)
    logistic_reg_test_metrics = get_metrics(logistic_reg_pipe, X_test, y_test)

    scores["logistic_regression_education"] = {
        "validation_accuracy": logistic_reg_val_metrics["acc"],
        "validation_precision": logistic_reg_val_metrics["prec"],
        "validation_recall": logistic_reg_val_metrics["rec"],
        "validation_f1": logistic_reg_val_metrics["f1"],
        "validation_auc": logistic_reg_val_metrics["auc"],
        "test_accuracy": logistic_reg_test_metrics["acc"],
        "test_precision": logistic_reg_test_metrics["prec"],
        "test_recall": logistic_reg_test_metrics["rec"],
        "test_f1": logistic_reg_test_metrics["f1"],
        "test_auc": logistic_reg_test_metrics["auc"],
    }

    print("validation_accuracy:", logistic_reg_val_metrics["acc"])
    print("test_accuracy:", logistic_reg_test_metrics["acc"])

# Random Forest
    random_forest_education = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    random_forest_pipe = build_pipeline(random_forest_education, "education")
    random_forest_pipe.fit(X_train, y_train)
    random_forest_validation_metrics = get_metrics(random_forest_pipe, X_val, y_val)
    random_forest_test_metrics = get_metrics(random_forest_pipe, X_test, y_test)

    scores["random_forest_education"] = {
        "validation_accuracy": random_forest_validation_metrics["acc"],
        "validation_precision": random_forest_validation_metrics["prec"],
        "validation_recall": random_forest_validation_metrics["rec"],
        "validation_f1": random_forest_validation_metrics["f1"],
        "validation_auc": random_forest_validation_metrics["auc"],
        "test_accuracy": random_forest_test_metrics["acc"],
        "test_precision": random_forest_test_metrics["prec"],
        "test_recall": random_forest_test_metrics["rec"],
        "test_f1": random_forest_test_metrics["f1"],
        "test_auc": random_forest_test_metrics["auc"],
    }

    print("validation_accuracy:", random_forest_validation_metrics["acc"])
    print("test_accuracy:", random_forest_test_metrics["acc"])

    return scores

if __name__ == "__main__":
    scores = run_education_model()
    print("Final scores:")
    print(scores)