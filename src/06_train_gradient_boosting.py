import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

features = [
    "koi_period", "koi_duration", "koi_depth", "koi_prad",
    "koi_teq", "koi_insol", "koi_model_snr", "koi_impact",
    "koi_steff", "koi_slogg", "koi_srad",
]

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

X_train = train[features]
y_train = train["label"]
X_test = test[features]
y_test = test["label"]

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("gb", GradientBoostingClassifier(random_state=42)),
])

weights = compute_sample_weight("balanced", y_train)
model.fit(X_train, y_train, gb__sample_weight=weights)

pred = model.predict(X_test)

print("Accuracy :", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred))
print("Recall   :", recall_score(y_test, pred))
print("F1-score :", f1_score(y_test, pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, pred))

joblib.dump(model, "models/gradient_boosting.joblib")
print("Model saved.")