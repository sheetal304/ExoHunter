import pandas as pd
import glob

file_path = glob.glob("data/raw/cumulative*.csv")[0]
df = pd.read_csv(file_path, comment="#")

print("File:", file_path)
print("Rows, columns:", df.shape)
print(df["koi_disposition"].value_counts())