import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

features = [
    "koi_period", "koi_duration", "koi_depth", "koi_prad",
    "koi_teq", "koi_insol", "koi_model_snr", "koi_impact",
    "koi_steff", "koi_slogg", "koi_srad",
]

train = pd.read_csv("data/processed/train.csv")
X_train = train[features]
y_train = train["label"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "precision", "recall", "f1"]

settings = [
    ("Balanced class weight", "balanced"),
    ("No class weight", None),
]

for name, weight in settings:
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=200, class_weight=weight, random_state=42)),
    ])
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
    print(name)
    for m in scoring:
        vals = scores["test_" + m]
        print(f"  {m:9s}: {vals.mean():.3f} +/- {vals.std():.3f}")
