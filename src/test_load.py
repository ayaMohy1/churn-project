import joblib
import pandas as pd

artifact = joblib.load("models/churn_model.joblib")
model = artifact["pipeline"]
threshold = artifact["threshold"]

# take 3 real customers just to test
df = pd.read_csv("data/TelcoCustomerChurn.csv")
sample = df[artifact["features"]].head(3)

proba = model.predict_proba(sample)[:, 1]
print("Churn probabilities:", proba.round(3))
print("Flagged as churn:", proba >= threshold)