# leakage_check.py -- standalone check, does NOT modify main.py or its results
# Replicates: file loading order, label normalisation, stratified 80:20 split (seed 42),
# and the 20,000-per-class cap used in main.py.

import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_FOLDER = r"C:\Users\Administrator\OneDrive\Desktop\archive"
SAMPLE_NROWS = 100000
SEED = 42
TEST_SIZE = 0.20
MAX_PER_CLASS = 20000


def find_label_column(df):
    for c in df.columns:
        if c.lower() in ['label', 'labels', 'class', 'attack', 'type', 'category']:
            return c
    return df.columns[-1]


def norm_label(x):
    return 'BENIGN' if x.strip().lower() in ['benign', 'normal', 'normal traffic'] else x.strip()


# ---------------------------------------------------------------
# 1) Load file by file; keep only row-hash, label, source-file id
# ---------------------------------------------------------------
files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
print("CSV files found:", len(files))

hash_parts, label_parts, src_parts = [], [], []
for i, f in enumerate(files):
    df = pd.read_csv(f, nrows=SAMPLE_NROWS)
    lab = find_label_column(df)
    df = df.dropna(subset=[lab])
    y = df[lab].astype(str).apply(norm_label).to_numpy()
    h = pd.util.hash_pandas_object(df.drop(columns=[lab]), index=False).to_numpy()
    hash_parts.append(h)
    label_parts.append(y)
    src_parts.append(np.full(len(df), i, dtype=np.int32))
    if (i + 1) % 20 == 0:
        print(f"  processed {i + 1}/{len(files)} files")

hashes = np.concatenate(hash_parts)
labels = np.concatenate(label_parts)
src = np.concatenate(src_parts)
del hash_parts, label_parts, src_parts

n = len(hashes)
print("Total rows:", n)

le = LabelEncoder()
y_enc = le.fit_transform(labels)
class_names = le.classes_

# ---------------------------------------------------------------
# 2) Same stratified split + same cap as main.py
# ---------------------------------------------------------------
idx = np.arange(n)
train_idx, test_idx = train_test_split(
    idx, test_size=TEST_SIZE, stratify=y_enc, random_state=SEED
)

rng = np.random.default_rng(SEED)
y_train_series = pd.Series(y_enc[train_idx], index=train_idx)
capped = []
for c in np.sort(y_train_series.unique()):
    ci = y_train_series[y_train_series == c].index.to_numpy()
    if len(ci) > MAX_PER_CLASS:
        ci = rng.choice(ci, size=MAX_PER_CLASS, replace=False)
    capped.extend(ci.tolist())
capped_idx = np.array(capped)

# ---------------------------------------------------------------
# 3) Results
# ---------------------------------------------------------------
print("\n================ RESULT 1: duplicates in the full data ================")
dup_pct = 100 * pd.Series(hashes).duplicated().mean()
print(f"Rows that are an exact repeat of an earlier row (features only): {dup_pct:.3f}%")

print("\n================ RESULT 2: test rows with an exact copy in train ================")
test_hashes = hashes[test_idx]
in_full = np.isin(test_hashes, hashes[train_idx])
in_capped = np.isin(test_hashes, hashes[capped_idx])
print(f"vs FULL 80% train partition : {100 * in_full.mean():.3f}%")
print(f"vs CAPPED train (used by models): {100 * in_capped.mean():.3f}%")

print("\nPer-class (capped train), 10 smallest classes:")
test_y = y_enc[test_idx]
counts = pd.Series(y_enc).value_counts().sort_values()
for c in counts.index[:10]:
    m = test_y == c
    print(f"  {class_names[c]:<25} test n={m.sum():>6}   with exact copy in train: {100 * in_capped[m].mean():6.2f}%")

print("\n================ RESULT 3: number of classes per source file ================")
per_file = pd.DataFrame({"src": src, "label": y_enc}).groupby("src")["label"].nunique()
print(per_file.describe())
print("Files with only 1 class:", int((per_file == 1).sum()), "of", len(per_file))

print("\nDone. Copy everything printed above and send it to me.")
