import pandas as pd, glob
f = glob.glob(r"C:\Users\Administrator\OneDrive\Desktop\archive\*.csv")[0]
d = pd.read_csv(f, nrows=100000)
print("duplicated rows in first file:", d.duplicated().sum())
print("unique rows:", len(d.drop_duplicates()))