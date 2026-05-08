import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from data_utils import load_data, clean_data, get_split, build_pipeline

def model_evaluation(y_real, y_prediction):
    return{
        "accuracy": accuracy_score(y_real, y_prediction),
        "precision": precision_score(y_real, y_prediction, zero_division=0),
        "recall": recall_score(y_real, y_prediction, zero_division=0),
        "f1": f1_score(y_real, y_prediction, zero_division=0),
    }

def main():
    df= clean_data(load_data())
    X_train, X_val, X_test, y_train, y_val, y_test = get_split(df)

    scores=[]

# Zero rule baseline

    baseline_model = DummyClassifier(strategy="most_frequent")
    baseline_model.fit(np.zeros((len(y_train), 1)), y_train)

    y_prediction_dummy = baseline_model.predict(np.zeros((len(y_val), 1)))
    score_dummy = model_evaluation(y_val, y_prediction_dummy)
    score_dummy["model"] = "zero_rule"
    scores.append(score_dummy)

# Logistic regression baseline
    model_logistic_regression = LogisticRegression(max_iter=1000)
    pipe_lr = build_pipeline(model_logistic_regression, "baseline")
    pipe_lr.fit(X_train, y_train)
    y_prediction_lr = pipe_lr.predict(X_val)
    scores_lr = model_evaluation(y_val, y_prediction_lr)
    scores_lr["model"] = "baseline_logistic_regression"
    scores.append(scores_lr)

    df_scores= pd.DataFrame(scores)

    print("\nbaseline:")
    print(df_scores)

if __name__ == "__main__":
    main()
