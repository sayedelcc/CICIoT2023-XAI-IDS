# edge_text_sensitivity.py -- Does the handling of the text columns of Edge-IIoTset change the conclusions?
#
# In the strategy comparison, the seven text columns (HTTP method / version / referer, MQTT protocol name /
# topic / CONNACK flags, DNS name) were turned into missing values and imputed, so six of them are constant
# and the information they carry was not used. This script compares two variants with identical folds,
# seeds and models:
#
#   "Text unused"    = the preprocessing used in the paper (edge_strategies.py)
#   "Text encoded"   = the seven text columns are label-encoded (training categories, unseen = -1)
#
# Strategies: Baseline, random oversampling, SMOTE (label-encoded codes interpolated and rounded) and, for the
# encoded variant, SMOTENC (the seven text columns declared categorical). Put this file next to
# edge_strategies.py (do not rename it) and run:    python edge_text_sensitivity.py
# ---------------------------------------------------------------------------------------------

import os
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score, f1_score,
                             confusion_matrix, precision_recall_fscore_support)
from imblearn.over_sampling import SMOTENC

from edge_strategies import (preprocess_features, resample, fit_model, DATA_PATH,
                             LABEL_COL, BINARY_LABEL_COL, DROP_COLS, CV_SEED)

# =========================================================
# CONFIG
# =========================================================
N_FOLDS = 10                                     # set to 2 for a quick sanity run
MODELS = ["RandomForest", "XGBoost"]
VARIANTS = {"Text unused (paper)": ["Baseline", "RandomOverSampler", "SMOTE"],
            "Text encoded": ["Baseline", "RandomOverSampler", "SMOTE", "SMOTENC"]}
FOCUS_CLASS = "Fingerprinting"
RESULTS_PATH = "results_edge_text_sensitivity/"
os.makedirs(RESULTS_PATH, exist_ok=True)


def prep_encoded(X_tr, X_te, text_cols):
    """Label-encode the text columns (not scaled), impute and standardize all other columns."""
    X_tr, X_te = X_tr.copy(), X_te.copy()
    for c in text_cols:
        tr = X_tr[c].fillna("MISSING").astype(str)
        te = X_te[c].fillna("MISSING").astype(str)
        mapping = {v: i for i, v in enumerate(sorted(tr.unique()))}
        X_tr[c] = tr.map(mapping).astype(int)
        X_te[c] = te.map(lambda v: mapping.get(v, -1)).astype(int)
    num = [c for c in X_tr.columns if c not in text_cols]
    for c in num:
        X_tr[c] = pd.to_numeric(X_tr[c], errors="coerce")
        X_te[c] = pd.to_numeric(X_te[c], errors="coerce")
        med = X_tr[c].median()
        X_tr[c] = X_tr[c].fillna(med)
        X_te[c] = X_te[c].fillna(med)
    sc = StandardScaler()
    X_tr[num] = sc.fit_transform(X_tr[num])
    X_te[num] = sc.transform(X_te[num])
    return X_tr, X_te, list(text_cols)


def validity(Xr, yr, n_orig, Xreal, yreal, cat_idx):
    """Share of synthetic records whose combination of text-category codes never occurs in the real data."""
    real_all = set(map(tuple, Xreal[:, cat_idx].astype(int).tolist()))
    by_class = {c: set(map(tuple, Xreal[yreal == c][:, cat_idx].astype(int).tolist())) for c in np.unique(yreal)}
    syn = Xr[n_orig:][:, cat_idx].astype(int).tolist()
    ys = yr[n_orig:]
    if not syn:
        return 0, np.nan, np.nan
    return (len(syn), float(np.mean([tuple(r) not in real_all for r in syn])),
            float(np.mean([tuple(r) not in by_class[c] for r, c in zip(syn, ys)])))


def holm_adjust(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def safe_w(a, b):
    try:
        return wilcoxon(a, b)[1]
    except ValueError:
        return 1.0


def main():
    print("Loading:", DATA_PATH)
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df.drop(columns=[c for c in DROP_COLS + [BINARY_LABEL_COL] if c in df.columns])
    df = df.dropna(subset=[LABEL_COL]).reset_index(drop=True)
    df = df.drop_duplicates().reset_index(drop=True)
    feats = [c for c in df.columns if c != LABEL_COL]
    df = df.drop(columns=[c for c in feats if df[c].nunique(dropna=False) <= 1])

    le = LabelEncoder()
    y = le.fit_transform(df[LABEL_COL])
    class_names = list(le.classes_)
    fp = class_names.index(FOCUS_CLASS)
    X = df.drop(columns=[LABEL_COL])
    text_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    print(f"Records: {len(X)} | features: {X.shape[1]} | text columns: {len(text_cols)} -> {text_cols}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    rows, pc_rows, val_rows = [], [], []
    t0 = time.time()

    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        seed = CV_SEED + fold
        y_tr, y_te = y[tr], y[te]
        for variant, strategies in VARIANTS.items():
            if variant.startswith("Text unused"):
                X_tr, X_te, cat_cols = preprocess_features(X.iloc[tr], X.iloc[te])     # as in the paper
            else:
                X_tr, X_te, cat_cols = prep_encoded(X.iloc[tr], X.iloc[te], text_cols)
            cols = list(X_tr.columns)
            cat_idx = [cols.index(c) for c in cat_cols]
            Xtr_np, Xte_np = X_tr.values.astype(float), X_te.values.astype(float)

            for strategy in strategies:
                try:
                    if strategy == "Baseline":
                        Xfit, yfit = Xtr_np, y_tr
                    elif strategy == "SMOTENC":
                        Xfit, yfit = SMOTENC(categorical_features=cat_idx, random_state=seed,
                                             k_neighbors=5).fit_resample(Xtr_np, y_tr)
                    else:
                        Xfit, yfit = resample(strategy, Xtr_np, y_tr, seed, cat_idx)
                except Exception as e:
                    print(f"  WARNING fold {fold} | {variant} | {strategy} failed: {type(e).__name__}: {e}")
                    continue
                if variant == "Text encoded" and strategy in ("SMOTE", "SMOTENC"):
                    n_syn, ua, uc = validity(Xfit, yfit, len(Xtr_np), Xtr_np, y_tr, cat_idx)
                    val_rows.append({"Fold": fold, "Strategy": strategy, "Synthetic records": n_syn,
                                     "Unseen text-category combination (any class)": ua,
                                     "Unseen text-category combination (own class)": uc})
                for name in MODELS:
                    model = fit_model(name, Xfit, yfit, seed, False)
                    pred = model.predict(Xte_np)
                    cm = confusion_matrix(y_te, pred, labels=np.arange(len(class_names)))
                    tp = cm[fp, fp]; fpn = cm[:, fp].sum() - tp; fn = cm[fp].sum() - tp
                    p = tp / (tp + fpn) if (tp + fpn) else 0.0
                    r = tp / (tp + fn) if (tp + fn) else 0.0
                    rows.append({"Fold": fold, "Variant": variant, "Strategy": strategy, "Model": name,
                                 "Accuracy": accuracy_score(y_te, pred),
                                 "Balanced accuracy": balanced_accuracy_score(y_te, pred),
                                 "Macro precision": precision_score(y_te, pred, average="macro", zero_division=0),
                                 "Macro F1": f1_score(y_te, pred, average="macro", zero_division=0),
                                 f"{FOCUS_CLASS} precision": p, f"{FOCUS_CLASS} recall": r,
                                 f"{FOCUS_CLASS} F1": 2 * p * r / (p + r) if (p + r) else 0.0})
                    _, _, f1c, _ = precision_recall_fscore_support(y_te, pred, labels=np.arange(len(class_names)), zero_division=0)
                    for ci, cn in enumerate(class_names):
                        pc_rows.append({"Fold": fold, "Variant": variant, "Strategy": strategy, "Model": name,
                                        "Class": cn, "F1": f1c[ci]})
        print(f"Fold {fold}/{N_FOLDS} done | elapsed {(time.time() - t0) / 60:.1f} min")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RESULTS_PATH, "all_results.csv"), index=False)
    pc = pd.DataFrame(pc_rows).groupby(["Variant", "Strategy", "Model", "Class"])["F1"].mean().reset_index()
    pc.to_csv(os.path.join(RESULTS_PATH, "per_class_f1_mean.csv"), index=False)
    val = pd.DataFrame(val_rows)
    val.to_csv(os.path.join(RESULTS_PATH, "synthetic_validity.csv"), index=False)

    metrics = ["Accuracy", "Balanced accuracy", "Macro precision", "Macro F1",
               f"{FOCUS_CLASS} precision", f"{FOCUS_CLASS} recall", f"{FOCUS_CLASS} F1"]
    agg = res.groupby(["Variant", "Strategy", "Model"])[metrics].agg(["mean", "std"])
    agg.columns = [f"{m} {s}" for m, s in agg.columns]
    agg.reset_index().to_csv(os.path.join(RESULTS_PATH, "summary_mean_std.csv"), index=False)

    # ---- paired tests
    trows = []
    def get(variant, strategy, model, m):
        s = res[(res["Variant"] == variant) & (res["Strategy"] == strategy) & (res["Model"] == model)].set_index("Fold")[m]
        return s
    for model in MODELS:
        for m in ["Macro F1", "Balanced accuracy"]:
            a, b = get("Text encoded", "Baseline", model, m), get("Text unused (paper)", "Baseline", model, m)
            ix = a.index.intersection(b.index)
            trows.append({"Model": model, "Comparison": "Baseline: text encoded vs text unused", "Metric": m,
                          "Mean diff": float((a[ix] - b[ix]).mean()), "p-value": safe_w(a[ix].values, b[ix].values)})
            for st in ["RandomOverSampler", "SMOTE", "SMOTENC"]:
                a, b = get("Text encoded", st, model, m), get("Text encoded", "Baseline", model, m)
                ix = a.index.intersection(b.index)
                trows.append({"Model": model, "Comparison": f"Text encoded: {st} vs Baseline", "Metric": m,
                              "Mean diff": float((a[ix] - b[ix]).mean()), "p-value": safe_w(a[ix].values, b[ix].values)})
            a, b = get("Text encoded", "SMOTENC", model, m), get("Text encoded", "SMOTE", model, m)
            ix = a.index.intersection(b.index)
            trows.append({"Model": model, "Comparison": "Text encoded: SMOTENC vs SMOTE", "Metric": m,
                          "Mean diff": float((a[ix] - b[ix]).mean()), "p-value": safe_w(a[ix].values, b[ix].values)})
    tests = pd.DataFrame(trows)
    tests["p-value (Holm)"] = holm_adjust(tests["p-value"].values)
    tests.to_csv(os.path.join(RESULTS_PATH, "tests.csv"), index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    for m in ["Macro F1", "Balanced accuracy", "Macro precision", f"{FOCUS_CLASS} precision", f"{FOCUS_CLASS} recall", f"{FOCUS_CLASS} F1"]:
        print(f"\n================ {m} (mean over folds) ================")
        print(res.pivot_table(index=["Variant", "Strategy"], columns="Model", values=m, aggfunc="mean").round(4).to_string())

    print("\n================ Per-class F1, Baseline: text unused vs text encoded ================")
    b = pc[pc["Strategy"] == "Baseline"].pivot_table(index="Class", columns=["Model", "Variant"], values="F1")
    print(b.round(3).to_string())

    print("\n================ Synthetic records with an unseen text-category combination (mean over folds) ================")
    if len(val):
        print(val.groupby("Strategy")[["Synthetic records", "Unseen text-category combination (any class)",
                                       "Unseen text-category combination (own class)"]].mean().round(4).to_string())
    print("\n================ Paired Wilcoxon tests (Holm over all rows) ================")
    print(tests.round(5).to_string(index=False))
    print("\nFinished. Files saved in:", RESULTS_PATH, "| total time:", f"{(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
