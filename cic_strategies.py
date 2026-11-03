# cic_strategies.py -- Imbalance-strategy comparison on CICIoT2023
# Strategies: Baseline (no rebalancing) | ClassWeight | RandomOverSampler | SMOTE   (training set capped at 20,000/class first)
# Protocol : stratified 10-fold CV; capping/resampling applied to TRAINING folds only.
# Each finished fold is saved to disk; if the run stops, just run the script again and it resumes.
# ---------------------------------------------------------------------------------------------

import os
import sys
import glob
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
from imblearn.over_sampling import RandomOverSampler, SMOTE
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================
# CONFIG
# =========================================================
DATA_FOLDER = r"C:\Users\Administrator\OneDrive\Desktop\archive"
if len(sys.argv) > 1:
    DATA_FOLDER = sys.argv[1]

SAMPLE_NROWS = 100000  # rows read from each CSV file
MAX_TRAIN_SAMPLES_PER_CLASS = 20000  # cap per class in the training set (same as before)

N_FOLDS = 10
CV_SEED = 42

RESULTS_PATH = "results_cic_strategies/"
os.makedirs(RESULTS_PATH, exist_ok=True)

STRATEGIES = ["Baseline", "ClassWeight", "RandomOverSampler", "SMOTE"]
MODELS = ["RandomForest", "XGBoost", "MLP"]  # MLP has no class_weight -> skipped for ClassWeight
METRICS_TESTED = ["Macro F1", "Balanced Accuracy"]


# =========================================================
# DATA
# =========================================================
def find_label_column(df):
    for c in df.columns:
        if c.lower() in ['label', 'labels', 'class', 'attack', 'type', 'category']:
            return c
    return df.columns[-1]


def load_data():
    files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {DATA_FOLDER}")
    print("CSV files found:", len(files))
    dfs = [pd.read_csv(f, nrows=SAMPLE_NROWS) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    del dfs
    label_col = find_label_column(df)
    df = df.dropna(subset=[label_col]).reset_index(drop=True)
    y_raw = df[label_col].astype(str).apply(
        lambda x: 'BENIGN' if x.strip().lower() in ['benign', 'normal', 'normal traffic'] else x.strip())
    X = df.drop(columns=[label_col])
    del df
    print("Data shape:", X.shape)
    return X, y_raw


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


def cap_indices(train_idx, y, max_per_class, seed):
    """Pick at most max_per_class training rows per class (indices only -> low memory)."""
    rng = np.random.default_rng(seed)
    y_tr = y[train_idx]
    chosen = []
    for c in np.sort(np.unique(y_tr)):
        ci = train_idx[y_tr == c]
        if len(ci) > max_per_class:
            ci = rng.choice(ci, size=max_per_class, replace=False)
        chosen.append(ci)
    chosen = np.concatenate(chosen)
    rng.shuffle(chosen)
    return chosen


# =========================================================
# RESAMPLING / MODELS / EVALUATION
# =========================================================
def resample(strategy, X, y, seed, cat_idx):
    if strategy == "RandomOverSampler":
        sampler = RandomOverSampler(random_state=seed)
    elif strategy == "SMOTE":
        sampler = SMOTE(random_state=seed, k_neighbors=5)
    else:
        raise ValueError(strategy)
    Xr, yr = sampler.fit_resample(X, y)
    for i in cat_idx:
        lo, hi = X[:, i].min(), X[:, i].max()
        Xr[:, i] = np.clip(np.rint(Xr[:, i]), lo, hi)
    return Xr, yr


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
# STATISTICS
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
    base = results_df[results_df["Strategy"] == "Baseline"]
    for model in MODELS:
        for strat in STRATEGIES:
            if strat == "Baseline":
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
        for metric in METRICS_TESTED:
            idx = out.index[out["Metric"] == metric]
            out.loc[idx, "p-value (Holm)"] = holm_adjust(out.loc[idx, "p-value"].values)
        out["Significant (Holm < 0.05)"] = out["p-value (Holm)"] < 0.05
    out.to_csv(os.path.join(RESULTS_PATH, "wilcoxon_strategy_vs_none.csv"), index=False)
    return out


# =========================================================
# AGGREGATION (reads the per-fold files)
# =========================================================
def aggregate():
    res_files = sorted(glob.glob(os.path.join(RESULTS_PATH, "fold_*_results.csv")))
    pc_files = sorted(glob.glob(os.path.join(RESULTS_PATH, "fold_*_perclass.csv")))
    if not res_files:
        print("No finished folds yet.")
        return
    res = pd.concat([pd.read_csv(f) for f in res_files], ignore_index=True)
    pcdf = pd.concat([pd.read_csv(f) for f in pc_files], ignore_index=True)
    n_folds_done = res["Fold"].nunique()

    res.to_csv(os.path.join(RESULTS_PATH, "all_results.csv"), index=False)
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
    print(f"\n######## RESULTS over {n_folds_done} finished fold(s) ########")
    for metric in ["Macro F1", "Balanced Accuracy", "Macro Precision", "Macro Recall"]:
        print(f"\n================ {metric} (mean over folds) ================")
        print(res.pivot_table(index="Strategy", columns="Model", values=metric,
                              aggfunc="mean").reindex(order).round(4).to_string())

    class_counts = pcdf.groupby("Class")["F1"].size()
    rare_candidates = ["Uploading_Attack", "Recon-PingSweep"]
    rare = [c for c in rare_candidates if c in set(pcdf["Class"])]
    if rare:
        print("\n================ F1 of the rarest classes (mean over folds) ================")
        pr = pcdf[pcdf["Class"].isin(rare)].pivot_table(
            index="Strategy", columns=["Model", "Class"], values="F1", aggfunc="mean").reindex(order)
        print(pr.round(3).to_string())

    print("\n================ Each strategy vs None (Wilcoxon, Holm-corrected) ================")
    if len(tests):
        print(tests.round(5).to_string(index=False))
    else:
        print("Not enough folds yet for the tests (need >= 3).")


# =========================================================
# MAIN
# =========================================================
def main():
    X, y_raw = load_data()
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = list(le.classes_)
    n_classes = len(class_names)
    print("Classes:", n_classes)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    t_start = time.time()

    for fold, (tr, te) in enumerate(skf.split(np.zeros((len(y), 1)), y), start=1):
        res_file = os.path.join(RESULTS_PATH, f"fold_{fold:02d}_results.csv")
        pc_file = os.path.join(RESULTS_PATH, f"fold_{fold:02d}_perclass.csv")
        if os.path.exists(res_file) and os.path.exists(pc_file):
            print(f"Fold {fold}/{N_FOLDS}: already finished, skipping")
            continue

        seed = CV_SEED + fold
        cap_idx = cap_indices(tr, y, MAX_TRAIN_SAMPLES_PER_CLASS, seed)
        X_tr, X_te, cat_cols = preprocess_features(X.iloc[cap_idx], X.iloc[te])
        cols = list(X_tr.columns)
        cat_idx = [cols.index(c) for c in cat_cols]
        Xtr_np, Xte_np = X_tr.values.astype(float), X_te.values.astype(float)
        y_tr, y_te = y[cap_idx], y[te]
        del X_tr, X_te

        results, per_class = [], []
        for strategy in STRATEGIES:
            class_weighting = strategy == "ClassWeight"
            if strategy in ("Baseline", "ClassWeight"):
                Xfit, yfit = Xtr_np, y_tr
            else:
                try:
                    Xfit, yfit = resample(strategy, Xtr_np, y_tr, seed, cat_idx)
                except Exception as e:
                    msg = f"fold {fold} | {strategy} failed: {type(e).__name__}: {e}"
                    print("  WARNING:", msg)
                    with open(os.path.join(RESULTS_PATH, "failures.txt"), "a") as fh:
                        fh.write(msg + "\n")
                    continue

            for name in MODELS:
                if class_weighting and name == "MLP":
                    continue
                t0 = time.time()
                model = fit_model(name, Xfit, yfit, seed, class_weighting)
                row, f1_pc = evaluate(model, Xte_np, y_te, n_classes)
                results.append({"Fold": fold, "Strategy": strategy, "Model": name, **row})
                for ci, cn in enumerate(class_names):
                    per_class.append({"Fold": fold, "Strategy": strategy, "Model": name,
                                      "Class": cn, "F1": f1_pc[ci]})
                print(f"  fold {fold} | {strategy:<18} | {name:<12} | Macro F1 {row['Macro F1']:.4f} "
                      f"| {(time.time() - t0) / 60:.1f} min")
                del model

        pd.DataFrame(per_class).to_csv(pc_file, index=False)
        pd.DataFrame(results).to_csv(res_file, index=False)  # written last = "fold finished" marker
        print(f"Fold {fold}/{N_FOLDS} done | elapsed {(time.time() - t_start) / 60:.1f} min")

    aggregate()
    print("\nFinished. Files saved in:", RESULTS_PATH)


if __name__ == "__main__":
    main()
