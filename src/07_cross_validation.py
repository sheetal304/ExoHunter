import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.utils.class_weight import compute_sample_weight

features = [
    "koi_period", "koi_duration", "koi_depth", "koi_prad",
    "koi_teq", "koi_insol", "koi_model_snr", "koi_impact",
    "koi_steff", "koi_slogg", "koi_srad",
]

train = pd.read_csv("data/processed/train.csv")
X_train = train[features]
y_train = train["label"]

rf = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("rf", RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42)),
])

gb = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("gb", GradientBoostingClassifier(random_state=42)),
])

weights = compute_sample_weight("balanced", y_train)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "precision", "recall", "f1"]

rf_scores = cross_validate(rf, X_train, y_train, cv=cv, scoring=scoring)
gb_scores = cross_validate(
    gb, X_train, y_train, cv=cv, scoring=scoring,
    params={"gb__sample_weight": weights},
)

for name, s in [("Random Forest", rf_scores), ("Gradient Boosting", gb_scores)]:
    print(name)
    for m in scoring:
        vals = s["test_" + m]
        print(f"  {m:9s}: {vals.mean():.3f} +/- {vals.std():.3f}")