# edge_confusion.py -- Why does rebalancing lower the precision of rare classes on Edge-IIoTset?
#
# Re-runs the Edge-IIoTset protocol (same cleaning, same 10 folds, same seeds, same models and
# resampling code as edge_strategies.py, which is imported) and keeps the pooled confusion matrices.
# For the rare classes it reports precision, recall, how many records are predicted as the class
# versus how many truly belong to it, and WHICH true classes the false positives come from.
#
# Put this file in the same folder as edge_strategies.py (do not rename that file) and run:
#       python edge_confusion.py
# ---------------------------------------------------------------------------------------------

import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

from edge_strategies import (preprocess_features, resample, fit_model, DATA_PATH,
                             LABEL_COL, BINARY_LABEL_COL, DROP_COLS, CV_SEED)

# =========================================================
# CONFIG
# =========================================================
N_FOLDS = 10                                   # set to 2 for a quick sanity run
STRATEGIES = ["Baseline", "ClassWeight", "RandomOverSampler", "SMOTE"]
MODELS = ["RandomForest", "XGBoost"]
TARGET_CLASSES = ["Fingerprinting", "MITM"]
RESULTS_PATH = "results_edge_confusion/"
os.makedirs(RESULTS_PATH, exist_ok=True)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def main():
    # ---- identical data preparation to edge_strategies.py
    print("Loading:", DATA_PATH)
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df.drop(columns=[c for c in DROP_COLS + [BINARY_LABEL_COL] if c in df.columns])
    df = df.dropna(subset=[LABEL_COL]).reset_index(drop=True)
    df = df.drop_duplicates().reset_index(drop=True)
    feat_cols = [c for c in df.columns if c != LABEL_COL]
    const = [c for c in feat_cols if df[c].nunique(dropna=False) <= 1]
    df = df.drop(columns=const)

    le = LabelEncoder()
    y = le.fit_transform(df[LABEL_COL])
    class_names = list(le.classes_)
    n_classes = len(class_names)
    X = df.drop(columns=[LABEL_COL])
    print(f"Records: {len(df)} | features: {X.shape[1]} | classes: {n_classes}")
    targets = {c: class_names.index(c) for c in TARGET_CLASSES if c in class_names}

    pooled = {(s, m): np.zeros((n_classes, n_classes), dtype=np.int64) for s in STRATEGIES for m in MODELS}
    fold_f1 = {(s, m, c): [] for s in STRATEGIES for m in MODELS for c in targets}
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    t0 = time.time()

    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        seed = CV_SEED + fold
        X_tr, X_te, cat_cols = preprocess_features(X.iloc[tr], X.iloc[te])
        cols = list(X_tr.columns)
        cat_idx = [cols.index(c) for c in cat_cols]
        Xtr_np, Xte_np = X_tr.values.astype(float), X_te.values.astype(float)
        y_tr, y_te = y[tr], y[te]

        for strategy in STRATEGIES:
            cw = strategy == "ClassWeight"
            if strategy in ("Baseline", "ClassWeight"):
                Xfit, yfit = Xtr_np, y_tr
            else:
                try:
                    Xfit, yfit = resample(strategy, Xtr_np, y_tr, seed, cat_idx)
                except Exception as e:
                    print(f"  WARNING fold {fold} | {strategy} failed: {type(e).__name__}: {e}")
                    continue
            for name in MODELS:
                model = fit_model(name, Xfit, yfit, seed, cw)
                cm = confusion_matrix(y_te, model.predict(Xte_np), labels=np.arange(n_classes))
                pooled[(strategy, name)] += cm
                for c, ci in targets.items():
                    tp = cm[ci, ci]; fn = cm[ci].sum() - tp; fp = cm[:, ci].sum() - tp
                    fold_f1[(strategy, name, c)].append(prf(tp, fp, fn)[2])
        print(f"Fold {fold}/{N_FOLDS} done | elapsed {(time.time() - t0) / 60:.1f} min")

    # ---- save pooled confusion matrices
    for (s, m), cm in pooled.items():
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
            os.path.join(RESULTS_PATH, f"cm_{m}_{s}.csv"))

    # ---- summary for target classes
    rows, src_rows = [], []
    for c, ci in targets.items():
        for m in MODELS:
            for s in STRATEGIES:
                cm = pooled[(s, m)]
                tp = int(cm[ci, ci]); fn = int(cm[ci].sum() - tp); fp = int(cm[:, ci].sum() - tp)
                p, r, f1 = prf(tp, fp, fn)
                fps = pd.Series(cm[:, ci], index=class_names).drop(class_names[ci]).sort_values(ascending=False)
                fns = pd.Series(cm[ci, :], index=class_names).drop(class_names[ci]).sort_values(ascending=False)
                top_fp = "; ".join(f"{k}: {int(v)}" for k, v in fps.head(3).items() if v > 0)
                top_fn = "; ".join(f"{k}: {int(v)}" for k, v in fns.head(3).items() if v > 0)
                rows.append({"Class": c, "Model": m, "Strategy": s, "True records": tp + fn, "Predicted as class": tp + fp,
                             "Predicted / true": round((tp + fp) / (tp + fn), 3), "TP": tp, "FP": fp, "FN": fn,
                             "Precision (pooled)": round(p, 4), "Recall (pooled)": round(r, 4), "F1 (pooled)": round(f1, 4),
                             "F1 (mean over folds)": round(float(np.mean(fold_f1[(s, m, c)])), 4),
                             "Top false-positive sources": top_fp, "Top false-negative destinations": top_fn})
                for k, v in fps.items():
                    if v > 0:
                        src_rows.append({"Class": c, "Model": m, "Strategy": s, "True class of the false positives": k, "Count": int(v)})
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(RESULTS_PATH, "rare_class_summary.csv"), index=False)
    pd.DataFrame(src_rows).to_csv(os.path.join(RESULTS_PATH, "false_positive_sources.csv"), index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 70)
    for c in targets:
        sub = summ[summ["Class"] == c]
        print(f"\n================ {c}: precision / recall (pooled over all test folds) ================")
        print(sub[["Model", "Strategy", "True records", "Predicted as class", "Predicted / true", "Precision (pooled)",
                   "Recall (pooled)", "F1 (pooled)", "F1 (mean over folds)"]].to_string(index=False))
        print(f"\n---- {c}: which true classes are wrongly predicted AS {c} (top 3), and where its own records go ----")
        print(sub[["Model", "Strategy", "Top false-positive sources", "Top false-negative destinations"]].to_string(index=False))
    print("\nFinished. Files saved in:", RESULTS_PATH, "| total time:", f"{(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
