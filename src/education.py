from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from data_utils import load_data, clean_data, get_split, build_pipeline

def run_education_model():
    df = load_data()
    df = clean_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = get_split(df)

    scores = {}

# Logistic Regression
    logistic_reg_model = LogisticRegression(max_iter=300, random_state=42)
    logistic_reg_pipe = build_pipeline(logistic_reg_model, "education")
    logistic_reg_pipe.fit(X_train, y_train)
    logistic_reg_val_acc = logistic_reg_pipe.score(X_val, y_val)
    logistic_reg_test_acc = logistic_reg_pipe.score(X_test, y_test)

    scores["logistic_regression_education"] = {
        "validation_accuracy": logistic_reg_val_acc,
        "test_accuracy": logistic_reg_test_acc,
    }

    print("validation_accuracy:", logistic_reg_val_acc)
    print("test_accuracy:", logistic_reg_test_acc)

# Random Forest
    random_forest_education = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    random_forest_pipe = build_pipeline(random_forest_education, "education")
    random_forest_pipe.fit(X_train, y_train)
    random_forest_validation_acc = random_forest_pipe.score(X_val, y_val)
    random_forest_test_acc = random_forest_pipe.score(X_test, y_test)

    scores["random_forest_education"] = {
        "validation_accuracy": random_forest_validation_acc,
        "test_accuracy": random_forest_test_acc,
    }

    print("validation_accuracy:", random_forest_validation_acc)
    print("test_accuracy:", random_forest_test_acc)

    return scores

if __name__ == "__main__":
    scores = run_education_model()
    print("Final scores:")
    print(scores)