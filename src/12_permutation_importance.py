import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

features = [
    "koi_period", "koi_duration", "koi_depth", "koi_prad",
    "koi_teq", "koi_insol", "koi_model_snr", "koi_impact",
    "koi_steff", "koi_slogg", "koi_srad",
]

test = pd.read_csv("data/processed/test.csv")
X_test = test[features]
y_test = test["label"]

model = joblib.load("models/random_forest.joblib")

result = permutation_importance(
    model, X_test, y_test,
    scoring="f1", n_repeats=10, random_state=42,
)

table = pd.DataFrame({
    "feature": features,
    "f1_drop_mean": result.importances_mean,
    "f1_drop_std": result.importances_std,
}).sort_values("f1_drop_mean", ascending=False)

print(table.round(3).to_string(index=False))

plot = table.sort_values("f1_drop_mean")
plt.figure(figsize=(8, 5))
plt.barh(plot["feature"], plot["f1_drop_mean"], xerr=plot["f1_drop_std"])
plt.title("Permutation importance (drop in F1 on the test set)")
plt.xlabel("Drop in F1-score")
plt.tight_layout()
plt.savefig("reports/figures/permutation_importance.png", dpi=150)
print("Chart saved.")