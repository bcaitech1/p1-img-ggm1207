import os
import pandas as pd

for files in os.listdir("."):
    if "csv" not in files:
        continue

    df = pd.read_csv(files, index_col=0)
    df.to_csv(files, index=False)
