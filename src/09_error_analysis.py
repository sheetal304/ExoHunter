import os
import joblib
import pandas as pd

features = [
    "koi_period", "koi_duration", "koi_depth", "koi_prad",
    "koi_teq", "koi_insol", "koi_model_snr", "koi_impact",
    "koi_steff", "koi_slogg", "koi_srad",
]

test = pd.read_csv("data/processed/test.csv")
model = joblib.load("models/random_forest.joblib")

test["pred"] = model.predict(test[features])
test["prob_planet"] = model.predict_proba(test[features])[:, 1]

test["result"] = "correct"
test.loc[(test["label"] == 0) & (test["pred"] == 1), "result"] = "false_alarm"
test.loc[(test["label"] == 1) & (test["pred"] == 0), "result"] = "missed_planet"

print(test["result"].value_counts())
print()
print("Median of each feature:")
print(test.groupby("result")[features].median().round(2).T)

os.makedirs("reports", exist_ok=True)
errors = test[test["result"] != "correct"]
errors.to_csv("reports/errors_random_forest.csv", index=False)
print()
print("Errors saved:", errors.shape)