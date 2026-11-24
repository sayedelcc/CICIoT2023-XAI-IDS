# edge_cat_check2.py -- Are the text columns of Edge-IIoTset handled the same way in all experiments?
#
# The preprocessing treats a text column as NUMERIC (text entries -> missing -> median) when more than 95%
# of the training values convert to numbers, and label-encodes it otherwise. The share of text entries
# depends on the class mix of the training data. This script replicates
#   (a) the ten training folds of the strategy comparison (natural class proportions), and
#   (b) the capped training pool of the explanation study (at most 5,000 records per class, 80% / 20% split),
# and reports, per text column, the share of convertible values and which branch is taken.
#
# Put this file next to edge_strategies.py and run:   python edge_cat_check2.py
# ---------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from edge_strategies import DATA_PATH, LABEL_COL, BINARY_LABEL_COL, DROP_COLS, CV_SEED

N_FOLDS = 10
XAI_SEED = 42                 # same constants as xai_eval.py
XAI_CAP_PER_CLASS = 5000
THRESHOLD = 0.95

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)


def cap_per_class(idx, y, cap, rng):            # identical to xai_eval.py
    out = []
    for c in np.unique(y[idx]):
        ci = idx[y[idx] == c]
        if len(ci) > cap:
            ci = rng.choice(ci, size=cap, replace=False)
        out.append(ci)
    return np.concatenate(out)


def main():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df.drop(columns=[c for c in DROP_COLS + [BINARY_LABEL_COL] if c in df.columns])
    df = df.dropna(subset=[LABEL_COL]).drop_duplicates().reset_index(drop=True)
    feats = [c for c in df.columns if c != LABEL_COL]
    df = df.drop(columns=[c for c in feats if df[c].nunique(dropna=False) <= 1])
    y = LabelEncoder().fit_transform(df[LABEL_COL].astype(str))
    X = df.drop(columns=[LABEL_COL])
    text_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    conv = {c: pd.to_numeric(X[c], errors="coerce").notna().to_numpy() for c in text_cols}
    print(f"Records: {len(X)} | features: {X.shape[1]} | text columns: {len(text_cols)}")

    # ---- what does the numeric part of each text column look like?
    print("\n=== Numeric part of each text column (value counts after conversion; NaN = text entries) ===")
    for c in text_cols:
        vc = pd.to_numeric(X[c], errors="coerce").value_counts(dropna=False).head(4)
        print(f"{c:<24} " + "; ".join(f"{k}: {v}" for k, v in vc.items()))

    # ---- (a) the ten training folds of the strategy comparison
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    fold_share = {c: [] for c in text_cols}
    for tr, _ in skf.split(X, y):
        for c in text_cols:
            fold_share[c].append(conv[c][tr].mean())

    # ---- (b) the capped training pool of the explanation study
    idx_all = np.arange(len(y))
    tr_idx, _ = train_test_split(idx_all, test_size=0.20, stratify=y, random_state=XAI_SEED)
    pool_idx = cap_per_class(tr_idx, y, XAI_CAP_PER_CLASS, np.random.default_rng(XAI_SEED))

    rows = []
    for c in text_cols:
        fs = np.array(fold_share[c]); ps = conv[c][pool_idx].mean()
        rows.append({"Column": c,
                     "Convertible, full data (%)": round(100 * conv[c].mean(), 3),
                     "Strategy folds: min (%)": round(100 * fs.min(), 3),
                     "Strategy folds: max (%)": round(100 * fs.max(), 3),
                     "Strategy folds: branch": "numeric (text -> median)" if (fs > THRESHOLD).all() else
                                               ("label-encoded" if (fs <= THRESHOLD).all() else "MIXED across folds"),
                     "XAI pool (%)": round(100 * ps, 3),
                     "XAI pool: branch": "numeric (text -> median)" if ps > THRESHOLD else "label-encoded"})
    out = pd.DataFrame(rows)
    out.to_csv("edge_text_column_branches.csv", index=False)
    print("\n=== Branch taken by the preprocessing, per text column ===")
    print(out.to_string(index=False))
    same = (out["Strategy folds: branch"] == out["XAI pool: branch"]).all()
    print("\nSame handling in the strategy comparison and in the explanation study:", "YES" if same else "NO")
    print("\nFinished. File saved: edge_text_column_branches.csv")


if __name__ == "__main__":
    main()
