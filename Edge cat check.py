# edge_cat_check.py -- What did the preprocessing do to text columns, and which numeric columns are really codes?
#
# (1) Edge-IIoTset: for every text (object) column that survives the cleaning, report how many values are
#     NOT numeric. The pipeline treats a text column as numeric if more than 95% of its values convert to
#     numbers; the remaining text values (for example GET / POST) then become missing and are imputed.
# (2) Edge-IIoTset and CICIoT2023: list numeric columns with at most 10 distinct values. Such columns are
#     binary flags or numeric codes, and SMOTE interpolates them as if they were continuous.
#
# Put this file next to edge_strategies.py and run:   python edge_cat_check.py
# ----------------------------------------------------------------------------------------------------

import os
import glob

import numpy as np
import pandas as pd

from edge_strategies import DATA_PATH, LABEL_COL, BINARY_LABEL_COL, DROP_COLS

CIC_FOLDER = r"C:\Users\Administrator\OneDrive\Desktop\archive"
MAX_DISTINCT = 10
RESULTS_PATH = "results_edge_cat_check/"
os.makedirs(RESULTS_PATH, exist_ok=True)

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 80)


def low_cardinality(df, label):
    rows = []
    for c in df.columns:
        if c == label or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        nu = df[c].nunique(dropna=True)
        if nu <= MAX_DISTINCT:
            vals = sorted(df[c].dropna().unique().tolist())
            kind = "binary" if nu <= 2 else "few distinct values"
            rows.append({"Column": c, "Distinct values": int(nu), "Type": kind,
                         "Values": ", ".join(str(v) for v in vals[:MAX_DISTINCT])})
    return pd.DataFrame(rows)


def main():
    # ---------------- Edge-IIoTset (same cleaning as the experiments) ----------------
    print("Loading:", DATA_PATH)
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df.drop(columns=[c for c in DROP_COLS + [BINARY_LABEL_COL] if c in df.columns])
    df = df.dropna(subset=[LABEL_COL]).drop_duplicates().reset_index(drop=True)
    feats = [c for c in df.columns if c != LABEL_COL]
    df = df.drop(columns=[c for c in feats if df[c].nunique(dropna=False) <= 1])
    X = df.drop(columns=[LABEL_COL])
    print(f"Edge-IIoTset after cleaning: {len(df)} records, {X.shape[1]} features")

    rows = []
    for c in X.select_dtypes(include=["object", "category"]).columns:
        conv = pd.to_numeric(X[c], errors="coerce")
        bad = X[c][conv.isna() & X[c].notna()]
        top = bad.astype(str).value_counts().head(5)
        rows.append({"Column": c, "Distinct values": int(X[c].nunique()),
                     "Numeric-convertible (%)": round(100 * conv.notna().mean(), 3),
                     "Non-numeric values": int(len(bad)), "Non-numeric (%)": round(100 * len(bad) / len(X), 3),
                     "Treated as numeric by the pipeline (>95% convertible)": bool(conv.notna().mean() > 0.95),
                     "Most frequent non-numeric values": "; ".join(f"{k}: {v}" for k, v in top.items())})
    obj = pd.DataFrame(rows)
    obj.to_csv(os.path.join(RESULTS_PATH, "edge_text_columns.csv"), index=False)
    print(f"\n=== Edge-IIoTset: text (object) columns among the {X.shape[1]} features: {len(obj)} ===")
    print(obj.to_string(index=False) if len(obj) else "(none)")

    lc = low_cardinality(X, LABEL_COL)
    lc.to_csv(os.path.join(RESULTS_PATH, "edge_low_cardinality_columns.csv"), index=False)
    print(f"\n=== Edge-IIoTset: numeric columns with at most {MAX_DISTINCT} distinct values: {len(lc)} of {X.shape[1]} ===")
    print(lc.to_string(index=False) if len(lc) else "(none)")

    # ---------------- CICIoT2023 (first file only, 100,000 rows) ----------------
    files = sorted(glob.glob(os.path.join(CIC_FOLDER, "*.csv")))
    if files:
        c0 = pd.read_csv(files[0], nrows=100000)
        label = next((c for c in c0.columns if c.lower() in ["label", "labels", "class", "attack", "type", "category"]), c0.columns[-1])
        n_obj = c0.drop(columns=[label]).select_dtypes(include=["object", "category"]).shape[1]
        print(f"\n=== CICIoT2023 (first file): text columns among the features: {n_obj} ===")
        lc2 = low_cardinality(c0, label)
        lc2.to_csv(os.path.join(RESULTS_PATH, "cic_low_cardinality_columns.csv"), index=False)
        print(f"numeric columns with at most {MAX_DISTINCT} distinct values: {len(lc2)} of {c0.shape[1] - 1}")
        print(lc2.to_string(index=False) if len(lc2) else "(none)")
    else:
        print("\nCICIoT2023 folder not found; edit CIC_FOLDER at the top of this file.")
    print("\nFinished. Files saved in:", RESULTS_PATH)


if __name__ == "__main__":
    main()
