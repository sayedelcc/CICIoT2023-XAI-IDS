# edge_strategies.py -- Imbalance-handling strategy comparison on Edge-IIoTset (ML version)
# Strategies: None | ClassWeight | RandomOverSampler | SMOTE | BorderlineSMOTE | ADASYN
# Protocol : stratified 10-fold CV; resampling applied to TRAINING folds only.
# ---------------------------------------------------------------------------------------

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
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, precision_recall_fscore_support
)
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================
# CONFIG
# =========================================================
DATA_PATH = r"C:\Users\Administrator\OneDrive\Desktop\edge\ML-EdgeIIoT-dataset.csv"
if len(sys.argv) > 1:
    DATA_PATH = sys.argv[1]

RESULTS_PATH = "results_edge_strategies/"
os.makedirs(RESULTS_PATH, exist_ok=True)

# Set N_FOLDS = 2 for a quick sanity run, then back to 10 for the real run.
N_FOLDS = 10
CV_SEED = 42

LABEL_COL = "Attack_type"
BINARY_LABEL_COL = "Attack_label"

DROP_COLS = [
    "frame.time", "ip.src_host", "ip.dst_host",
    "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
    "icmp.transmit_timestamp",
    "http.file_data", "http.request.uri.query", "http.request.full_uri",
    "tcp.options", "tcp.payload", "tcp.srcport", "tcp.dstport", "udp.port",
    "mqtt.msg",
]

STRATEGIES = ["None", "ClassWeight", "RandomOverSampler", "SMOTE", "BorderlineSMOTE", "ADASYN"]
MODELS = ["RandomForest", "XGBoost", "MLP"]  # MLP (sklearn) has no class_weight -> skipped for ClassWeight
METRICS_TESTED = ["Macro F1", "Balanced Accuracy"]


# =========================================================
# PREPROCESSING
# =========================================================
def preprocess_features(X_train, X_test):
    """Impute, label-encode text columns, scale numeric columns.
    Returns the transformed frames and the names of the label-encoded (categorical) columns."""
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

    cat_cols = []
    for c in object_cols:
        le = LabelEncoder()
        tr = X_train[c].fillna("MISSING").astype(str)
        te = X_test[c].fillna("MISSING").astype(str)
        le.fit(tr)
        mapping = {cls: i for i, cls in enumerate(le.classes_)}
        X_train[c] = tr.map(mapping).astype(int)
        X_test[c] = te.map(lambda x: mapping.get(x, -1)).astype(int)
        cat_cols.append(c)

    for c in numeric_cols:
        med = X_train[c].median()
        X_train[c] = X_train[c].fillna(med)
        X_test[c] = X_test[c].fillna(med)

    scaler = StandardScaler()
    if numeric_cols:
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_test, cat_cols


# =========================================================
# RESAMPLING
# =========================================================
def make_sampler(strategy, seed):
    if strategy == "RandomOverSampler":
        return RandomOverSampler(random_state=seed)
    if strategy == "SMOTE":
        return SMOTE(random_state=seed, k_neighbors=5)
    if strategy == "BorderlineSMOTE":
        return BorderlineSMOTE(random_state=seed, k_neighbors=5, m_neighbors=10, kind="borderline-1")
    if strategy == "ADASYN":
        return ADASYN(random_state=seed, n_neighbors=5)
    raise ValueError(strategy)


def resample(strategy, X, y, seed, cat_idx):
    """Oversample every minority class up to the majority-class size (training fold only).
    Label-encoded categorical columns are snapped back to valid integer codes after interpolation."""
    sampler = make_sampler(strategy, seed)
    Xr, yr = sampler.fit_resample(X, y)
    for i in cat_idx:
        lo, hi = X[:, i].min(), X[:, i].max()
        Xr[:, i] = np.clip(np.rint(Xr[:, i]), lo, hi)
    return Xr, yr


# =========================================================
# MODELS
# =========================================================
def fit_model(name, X, y, seed, class_weighting=False):
    if name == "RandomForest":
        m = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1,
                                   class_weight="balanced" if class_weighting else None)
        return m.fit(X, y)
    if name == "XGBoost":
        m = XGBClassifier(n_estimators=100, eval_metric="mlogloss", random_state=seed, n_jobs=-1)
        if class_weighting:
            return m.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
        return m.fit(X, y)
    if name == "MLP":
        m = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True,
                          n_iter_no_change=10, random_state=seed)
        return m.fit(X, y)
    raise ValueError(name)


def evaluate(model, X_test, y_test, n_classes):
    y_pred = model.predict(X_test)
    row = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
        "Macro Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Macro Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "Macro F1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "Weighted F1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }
    _, _, f1_pc, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=np.arange(n_classes), zero_division=0)
    return row, f1_pc


# =========================================================
# STATISTICS (each strategy vs "None", paired over folds, Holm-corrected)
# =========================================================
def holm_adjust(p):
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def strategy_tests(results_df):
    rows = []
    base = results_df[results_df["Strategy"] == "None"]
    for model in MODELS:
        for strat in STRATEGIES:
            if strat == "None":
                continue
            cur = results_df[(results_df["Strategy"] == strat) & (results_df["Model"] == model)]
            ref = base[base["Model"] == model]
            merged = cur.merge(ref, on="Fold", suffixes=("_s", "_b"))
            if len(merged) < 3:
                continue
            for metric in METRICS_TESTED:
                a, b = merged[f"{metric}_s"].values, merged[f"{metric}_b"].values
                try:
                    _, p = wilcoxon(a, b)
                except ValueError:
                    p = 1.0
                rows.append({"Model": model, "Strategy vs None": strat, "Metric": metric,
                             "Folds": len(merged), "Mean diff": float(np.mean(a - b)), "p-value": p})
    out = pd.DataFrame(rows)
    if len(out):
        out["p-value (Holm)"] = np.nan
        for metric in METRICS_TESTED:  # one correction family per metric
            idx = out.index[out["Metric"] == metric]
            out.loc[idx, "p-value (Holm)"] = holm_adjust(out.loc[idx, "p-value"].values)
        out["Significant (Holm < 0.05)"] = out["p-value (Holm)"] < 0.05
    out.to_csv(os.path.join(RESULTS_PATH, "wilcoxon_strategy_vs_none.csv"), index=False)
    return out


# =========================================================
# MAIN
# =========================================================
def main():
    print("Loading:", DATA_PATH)
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print("Raw shape:", df.shape)

    to_drop = [c for c in DROP_COLS + [BINARY_LABEL_COL] if c in df.columns]
    df = df.drop(columns=to_drop)
    df = df.dropna(subset=[LABEL_COL]).reset_index(drop=True)

    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed {n0 - len(df)} exact duplicate rows -> {len(df)} rows")

    feat_cols = [c for c in df.columns if c != LABEL_COL]
    const = [c for c in feat_cols if df[c].nunique(dropna=False) <= 1]
    df = df.drop(columns=const)

    le = LabelEncoder()
    y = le.fit_transform(df[LABEL_COL])
    class_names = list(le.classes_)
    X = df.drop(columns=[LABEL_COL])
    print("Features:", X.shape[1], "| Classes:", len(class_names))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    results, per_class, failures = [], [], []
    t_start = time.time()

    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        seed = CV_SEED + fold
        X_tr, X_te, cat_cols = preprocess_features(X.iloc[tr], X.iloc[te])
        cols = list(X_tr.columns)
        cat_idx = [cols.index(c) for c in cat_cols]
        Xtr_np, Xte_np = X_tr.values.astype(float), X_te.values.astype(float)
        y_tr, y_te = y[tr], y[te]

        for strategy in STRATEGIES:
            class_weighting = strategy == "ClassWeight"
            if strategy in ("None", "ClassWeight"):
                Xfit, yfit = Xtr_np, y_tr
            else:
                try:
                    Xfit, yfit = resample(strategy, Xtr_np, y_tr, seed, cat_idx)
                except Exception as e:  # e.g. ADASYN can fail on some folds
                    msg = f"fold {fold} | {strategy} failed: {type(e).__name__}: {e}"
                    print("  WARNING:", msg)
                    failures.append(msg)
                    continue

            for name in MODELS:
                if class_weighting and name == "MLP":
                    continue
                model = fit_model(name, Xfit, yfit, seed, class_weighting)
                row, f1_pc = evaluate(model, Xte_np, y_te, len(class_names))
                results.append({"Fold": fold, "Strategy": strategy, "Model": name, **row})
                for ci, cn in enumerate(class_names):
                    per_class.append({"Fold": fold, "Strategy": strategy, "Model": name,
                                      "Class": cn, "F1": f1_pc[ci]})

        print(f"Fold {fold}/{N_FOLDS} done | elapsed {(time.time() - t_start) / 60:.1f} min")

    res = pd.DataFrame(results)
    res.to_csv(os.path.join(RESULTS_PATH, "all_results.csv"), index=False)
    pcdf = pd.DataFrame(per_class)
    pcdf.groupby(["Strategy", "Model", "Class"])["F1"].mean().reset_index().to_csv(
        os.path.join(RESULTS_PATH, "per_class_f1_mean.csv"), index=False)

    metrics = ["Accuracy", "Balanced Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1"]
    agg = res.groupby(["Strategy", "Model"])[metrics].agg(["mean", "std"])
    agg.columns = [f"{m} {s}" for m, s in agg.columns]
    agg = agg.reset_index()
    agg["Folds"] = res.groupby(["Strategy", "Model"]).size().values
    agg.to_csv(os.path.join(RESULTS_PATH, "summary_mean_std.csv"), index=False)

    tests = strategy_tests(res)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    order = [s for s in STRATEGIES if s in res["Strategy"].unique()]
    for metric in ["Macro F1", "Balanced Accuracy", "Macro Precision", "Macro Recall"]:
        print(f"\n================ {metric} (mean over folds) ================")
        pv = res.pivot_table(index="Strategy", columns="Model", values=metric, aggfunc="mean").reindex(order)
        print(pv.round(4).to_string())

    print("\n================ F1 of the rarest classes (mean over folds) ================")
    counts = pd.Series(y).value_counts().sort_values()
    rare = [class_names[i] for i in counts.index[:2]]
    pr = pcdf[pcdf["Class"].isin(rare)].pivot_table(
        index="Strategy", columns=["Model", "Class"], values="F1", aggfunc="mean").reindex(order)
    print(pr.round(3).to_string())

    print("\n================ Each strategy vs None (Wilcoxon, Holm-corrected) ================")
    if len(tests):
        print(tests.round(5).to_string(index=False))

    if failures:
        print("\nFailures:")
        for f in failures:
            print(" ", f)
    print("\nFinished. Files saved in:", RESULTS_PATH, "| total time:", f"{(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
