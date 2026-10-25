import pandas as pd
import glob

file_path = glob.glob("data/raw/cumulative*.csv")[0]
df = pd.read_csv(file_path, comment="#")

for i, name in enumerate(df.columns):
    print(i, name)