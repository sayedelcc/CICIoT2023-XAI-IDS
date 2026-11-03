# edge_pipeline.py -- Independent validation on Edge-IIoTset (ML version)
# Protocol: same as CICIoT2023 (stratified 10-fold, RF/XGBoost/MLP, Before vs After oversampling)
# ---------------------------------------------------------------------------------------------

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, precision_recall_fscore_support
)
from xgboost import XGBClassifier
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# CONFIG
# =========================================================
DATA_PATH = r"C:\Users\Administrator\OneDrive\Desktop\edge\ML-EdgeIIoT-dataset.csv"
if len(sys.argv) > 1:  # optional: python edge_pipeline.py other_file.csv
    DATA_PATH = sys.argv[1]

RESULTS_PATH = "results_edge/"
os.makedirs(RESULTS_PATH, exist_ok=True)

N_FOLDS = 10
CV_SEED = 42

LABEL_COL = "Attack_type"  # 15-class target
BINARY_LABEL_COL = "Attack_label"  # binary copy of the target -> must be removed (leakage)

# Identifier / session / payload columns: they identify the capture session or carry raw
# payload text, not attack behaviour. Removed to avoid label leakage and high-cardinality noise.
DROP_COLS = [
    "frame.time", "ip.src_host", "ip.dst_host",
    "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
    "icmp.transmit_timestamp",
    "http.file_data", "http.request.uri.query", "http.request.full_uri",
    "tcp.options", "tcp.payload", "tcp.srcport", "tcp.dstport", "udp.port",
    "mqtt.msg",
]


# =========================================================
# PREPROCESSING (same logic as the CICIoT2023 pipeline)
# =========================================================
def preprocess_features(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    for c in object_cols[:]:
        train_conv = pd.to_numeric(X_train[c], errors="coerce")
        test_conv = pd.to_numeric(X_test[c], errors="coerce")
        if train_conv.notna().mean() > 0.95:
            X_train[c] = train_conv
            X_test[c] = test_conv
            numeric_cols.append(c)
            object_cols.remove(c)

    for c in object_cols:
        le = LabelEncoder()
        tr = X_train[c].fillna("MISSING").astype(str)
        te = X_test[c].fillna("MISSING").astype(str)
        le.fit(tr)
        mapping = {cls: i for i, cls in enumerate(le.classes_)}
        X_train[c] = tr.map(mapping).astype(int)
        X_test[c] = te.map(lambda x: mapping.get(x, -1)).astype(int)

    for c in numeric_cols:
        med = X_train[c].median()
        X_train[c] = X_train[c].fillna(med)
        X_test[c] = X_test[c].fillna(med)

    scaler = StandardScaler()
    if numeric_cols:
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_test


# =========================================================
# BALANCING: random oversampling up to the largest class of the training fold
# =========================================================
def oversample_to_majority(X, y, random_state=42):
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    target = counts.max()

    X_parts, y_parts = [], []
    for c, n in zip(classes, counts):
        idx = np.flatnonzero(y == c)
        if n < target:
            extra = rng.choice(idx, size=target - n, replace=True)
            idx = np.concatenate([idx, extra])
        X_parts.append(X[idx])
        y_parts.append(y[idx])

    Xb = np.concatenate(X_parts, axis=0)
    yb = np.concatenate(y_parts, axis=0)
    perm = rng.permutation(len(yb))
    return Xb[perm], yb[perm]


# =========================================================
# MODELS
# =========================================================
def train_models(X, y, seed):
    models = {}
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=100, random_state=seed, n_jobs=-1).fit(X, y)
    models["XGBoost"] = XGBClassifier(
        n_estimators=100, eval_metric="mlogloss", random_state=seed, n_jobs=-1).fit(X, y)
    models["MLP"] = MLPClassifier(
        hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True,
        n_iter_no_change=10, random_state=seed).fit(X, y)
    return models


def evaluate(model, X_test, y_test, n_classes):
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    dt = time.perf_counter() - t0
    row = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
        "Macro Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Macro Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "Macro F1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "Weighted F1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "Inference Time (ms/sample)": dt / len(X_test) * 1000,
    }
    _, _, f1_per_class, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=np.arange(n_classes), zero_division=0)
    return row, f1_per_class


# =========================================================
# STATISTICS
# =========================================================
def safe_wilcoxon(a, b):
    try:
        stat, p = wilcoxon(a, b)
        return stat, p
    except ValueError:  # all differences are exactly zero
        return np.nan, 1.0


def run_statistical_tests(results_df):
    rows = []
    metrics = ["Accuracy", "Balanced Accuracy", "Macro F1", "Weighted F1"]

    # (a) model vs model, inside each condition
    pairs = [("RandomForest", "XGBoost"), ("RandomForest", "MLP"), ("XGBoost", "MLP")]
    for cond in results_df["Condition"].unique():
        sub = results_df[results_df["Condition"] == cond]
        for a, b in pairs:
            sa = sub[sub["Model"] == a].sort_values("Fold")
            sb = sub[sub["Model"] == b].sort_values("Fold")
            for m in metrics:
                stat, p = safe_wilcoxon(sa[m].values, sb[m].values)
                rows.append({"Test": "model_vs_model", "Condition": cond,
                             "Comparison": f"{a} vs {b}", "Metric": m,
                             "Statistic": stat, "p-value": p, "Significant (p < 0.05)": p < 0.05})

    # (b) Before vs After oversampling, per model  (the key question of the paper)
    for model in results_df["Model"].unique():
        sub = results_df[results_df["Model"] == model]
        before = sub[sub["Condition"] == "Before_Balancing"].sort_values("Fold")
        after = sub[sub["Condition"] == "After_Balancing"].sort_values("Fold")
        for m in metrics:
            stat, p = safe_wilcoxon(after[m].values, before[m].values)
            diff = float(np.mean(after[m].values - before[m].values))
            rows.append({"Test": "after_vs_before", "Condition": "After - Before",
                         "Comparison": model, "Metric": m,
                         "Statistic": stat, "p-value": p, "Significant (p < 0.05)": p < 0.05,
                         "Mean difference (After-Before)": diff})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS_PATH, "wilcoxon_tests.csv"), index=False)
    return out


def summarize(results_df):
    metrics = ["Accuracy", "Balanced Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1"]
    rows = []
    for (cond, model), g in results_df.groupby(["Condition", "Model"]):
        r = {"Condition": cond, "Model": model, "Folds": len(g)}
        for m in metrics:
            v = g[m].to_numpy(dtype=float)
            r[f"{m} Mean"] = v.mean()
            r[f"{m} Std"] = v.std(ddof=1) if len(v) > 1 else np.nan
        rows.append(r)
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(RESULTS_PATH, "cv10_summary_mean_std.csv"), index=False)
    return s


# =========================================================
# MAIN
# =========================================================
def main():
    print("Loading:", DATA_PATH)
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print("Raw shape:", df.shape)

    # --- remove leakage / identifier columns
    to_drop = [c for c in DROP_COLS + [BINARY_LABEL_COL] if c in df.columns]
    df = df.drop(columns=to_drop)
    print(f"Dropped {len(to_drop)} columns:", to_drop)

    df = df.dropna(subset=[LABEL_COL]).reset_index(drop=True)

    # --- duplicate check (features + label), then removal
    n_before = len(df)
    n_dup = int(df.duplicated().sum())
    print(f"\nExact duplicate rows (features+label): {n_dup} ({100 * n_dup / n_before:.2f}%)")
    df = df.drop_duplicates().reset_index(drop=True)
    feat_cols = [c for c in df.columns if c != LABEL_COL]
    n_conflict = int(df.duplicated(subset=feat_cols, keep=False).sum())
    print(f"Rows after removing duplicates: {len(df)}")
    print(f"Rows with identical features but DIFFERENT labels (ambiguous): {n_conflict}")

    # --- drop constant columns
    const = [c for c in feat_cols if df[c].nunique(dropna=False) <= 1]
    if const:
        df = df.drop(columns=const)
        print("Dropped constant columns:", const)

    print("\nClass distribution after cleaning:")
    print(df[LABEL_COL].value_counts().to_string())
    df[LABEL_COL].value_counts().to_csv(os.path.join(RESULTS_PATH, "class_distribution.csv"))

    le = LabelEncoder()
    y = le.fit_transform(df[LABEL_COL])
    class_names = list(le.classes_)
    X = df.drop(columns=[LABEL_COL])
    print("Final feature count:", X.shape[1])

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    results, per_class_rows = [], []

    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        seed = CV_SEED + fold
        print(f"\n===== FOLD {fold}/{N_FOLDS} =====")
        X_tr, X_te = preprocess_features(X.iloc[tr], X.iloc[te])
        Xtr_np, Xte_np = X_tr.values, X_te.values
        y_tr, y_te = y[tr], y[te]

        for condition in ["Before_Balancing", "After_Balancing"]:
            if condition == "After_Balancing":
                Xfit, yfit = oversample_to_majority(Xtr_np, y_tr, random_state=seed)
            else:
                Xfit, yfit = Xtr_np, y_tr
            models = train_models(Xfit, yfit, seed)
            for name, model in models.items():
                row, f1_pc = evaluate(model, Xte_np, y_te, len(class_names))
                results.append({"Fold": fold, "Model": name, "Condition": condition, **row})
                for ci, cn in enumerate(class_names):
                    per_class_rows.append({"Fold": fold, "Model": name, "Condition": condition,
                                           "Class": cn, "F1": f1_pc[ci]})
            print(f"  {condition}: done")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_PATH, "cv10_all_results.csv"), index=False)

    pc = pd.DataFrame(per_class_rows).groupby(["Model", "Condition", "Class"])["F1"].mean().reset_index()
    pc.to_csv(os.path.join(RESULTS_PATH, "per_class_f1_mean.csv"), index=False)

    summary = summarize(results_df)
    tests = run_statistical_tests(results_df)

    pd.set_option("display.width", 200)
    print("\n================ SUMMARY (mean over folds) ================")
    print(summary[["Condition", "Model", "Accuracy Mean", "Balanced Accuracy Mean",
                   "Macro F1 Mean", "Macro F1 Std"]].round(4).to_string(index=False))
    print("\n================ AFTER vs BEFORE oversampling (Wilcoxon) ================")
    t = tests[tests["Test"] == "after_vs_before"]
    print(t[["Comparison", "Metric", "Mean difference (After-Before)", "p-value"]].round(5).to_string(index=False))
    print("\nFinished. Files saved in:", RESULTS_PATH)


if __name__ == "__main__":
    main()
