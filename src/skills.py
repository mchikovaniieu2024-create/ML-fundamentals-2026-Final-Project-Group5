from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


from config import RANDOM_STATE
from data_utils import load_data, clean_data, get_split, build_pipeline




def run_experience_experiments():
   """
   Experience-only experiment:
   - uses from preprocessing load + clean + split
   - uses the shared 'experience' preprocessing pipeline
   - trains Logistic Regression and Random Forest
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
           "experience_lr": lr,
           "experience_rf": rf,
       },
       "predictions": {
           "experience_lr": {
               "val_pred": lr.predict(X_val),
               "val_prob": lr.predict_proba(X_val)[:, 1],
               "test_pred": lr.predict(X_test),
               "test_prob": lr.predict_proba(X_test)[:, 1],
           },
           "experience_rf": {
               "val_pred": rf.predict(X_val),
               "val_prob": rf.predict_proba(X_val)[:, 1],
               "test_pred": rf.predict(X_test),
               "test_prob": rf.predict_proba(X_test)[:, 1],
           },
       },
   }


   print("Experience models trained successfully.")
   print("Ready for evaluation.")


   return results




if __name__ == "__main__":
   results = run_experience_experiments()
   for model_name, preds in results["predictions"].items():
       print(model_name, list(preds.keys()))

