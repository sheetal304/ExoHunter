import pandas as pd
import glob

file_path = glob.glob("data/raw/cumulative*.csv")[0]
df = pd.read_csv(file_path, comment="#")

features = [
    "koi_period", "koi_duration", "koi_depth", "koi_prad",
    "koi_teq", "koi_insol", "koi_model_snr", "koi_impact",
    "koi_steff", "koi_slogg", "koi_srad",
]

# Sirf CONFIRMED aur FALSE POSITIVE rakho
labeled = df[df["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])].copy()

# Label: 1 = planet, 0 = false positive
labeled["label"] = (labeled["koi_disposition"] == "CONFIRMED").astype(int)

clean = labeled[["kepoi_name"] + features + ["label"]]

print("Shape:", clean.shape)
print(clean["label"].value_counts())
print("Missing values:")
print(clean.isna().sum())

clean.to_csv("data/processed/koi_labeled.csv", index=False)

# CANDIDATE rows alag save, baad ke liye
cand = df[df["koi_disposition"] == "CANDIDATE"][["kepoi_name"] + features]
cand.to_csv("data/processed/koi_candidates.csv", index=False)
print("Candidates saved:", cand.shape)