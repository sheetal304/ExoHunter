import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

features = [
    "koi_period", "koi_duration", "koi_depth", "koi_prad",
    "koi_teq", "koi_insol", "koi_model_snr", "koi_impact",
    "koi_steff", "koi_slogg", "koi_srad",
]

test = pd.read_csv("data/processed/test.csv")
X_test = test[features]
y_test = test["label"]

models = [
    ("Random Forest", "models/random_forest.joblib"),
    ("Gradient Boosting", "models/gradient_boosting.joblib"),
]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, (name, path) in zip(axes, models):
    model = joblib.load(path)
    pred = model.predict(X_test)
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred,
        display_labels=["False positive", "Planet"],
        ax=ax, colorbar=False,
    )
    ax.set_title(name)

plt.tight_layout()
os.makedirs("reports/figures", exist_ok=True)
plt.savefig("reports/figures/confusion_matrices.png", dpi=150)
print("Chart saved.")