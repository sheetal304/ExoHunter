import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

features = [
    "koi_period", "koi_duration", "koi_depth", "koi_prad",
    "koi_teq", "koi_insol", "koi_model_snr", "koi_impact",
    "koi_steff", "koi_slogg", "koi_srad",
]

model = joblib.load("models/random_forest.joblib")
rf = model.named_steps["rf"]

importance = pd.Series(rf.feature_importances_, index=features)
importance = importance.sort_values(ascending=False)

print(importance.round(3))

os.makedirs("reports/figures", exist_ok=True)

plt.figure(figsize=(8, 5))
importance.sort_values().plot(kind="barh")
plt.title("Random Forest feature importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("reports/figures/rf_feature_importance.png", dpi=150)
print("Chart saved.")