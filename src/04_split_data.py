import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/processed/koi_labeled.csv")

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"],
)

train_df.to_csv("data/processed/train.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Train label counts:")
print(train_df["label"].value_counts())
print("Test label counts:")
print(test_df["label"].value_counts())