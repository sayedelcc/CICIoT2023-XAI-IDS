import pandas as pd

path = r"C:\Users\Administrator\OneDrive\Desktop\edge\ML-EdgeIIoT-dataset.csv"
df = pd.read_csv(path, low_memory=False)

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nDtypes:\n", df.dtypes.value_counts())

print("\nObject (text) columns:")
for c in df.select_dtypes(include="object").columns:
    print(" ", c, "-> unique values:", df[c].nunique())

for c in df.columns[-3:]:
    print("\n=== Last columns:", c, "===")
    print(df[c].value_counts().head(20))