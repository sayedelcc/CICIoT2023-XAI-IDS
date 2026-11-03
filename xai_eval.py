# xai_eval.py -- Quantitative evaluation of SHAP vs LIME explanations for a Random Forest IDS
#
# Usage:   python xai_eval.py edge            (Edge-IIoTset)      <- run this first
#          python xai_eval.py cic             (CICIoT2023)
# Optional second argument = path override (CSV file for edge, folder for cic).
#
# What is measured (all numbers, no plots-only claims):
#   1. FIDELITY  (deletion test): remove the k features a method ranks as most supportive of the
#                predicted class (replace by the training median) and measure the drop in the predicted-
#                class probability and the fraction of flipped predictions. Compared with removing k RANDOM
#                features. A faithful explanation causes a larger drop than random removal.
#   2. STABILITY: N_RUNS random forests are trained on different 80% subsamples of the training pool.
#                For the same test instances we compare the top-k features (Jaccard) and the full
#                importance ranking (Spearman) between runs.
#   3. AGREEMENT: overlap between SHAP and LIME inside the same run.
#   4. COST: seconds per explanation.
# ------------------------------------------------------------------------------------------------

import os
import sys
import glob
import time
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon, rankdata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

import shap
from lime.lime_tabular import LimeTabularExplainer

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================
# CONFIG
# =========================================================
DATASET = sys.argv[1].lower() if len(sys.argv) > 1 else "edge"
PATH_OVERRIDE = sys.argv[2] if len(sys.argv) > 2 else None

EDGE_PATH = r"C:\Users\Administrator\OneDrive\Desktop\edge\ML-EdgeIIoT-dataset.csv"
CIC_FOLDER = r"C:\Users\Administrator\OneDrive\Desktop\archive"
CIC_NROWS_PER_FILE = 20000        # rows read from each CICIoT2023 file for this study

# Set True for a ~5-minute sanity run (tiny settings), then back to False for the real run.
QUICK_TEST = False

SEED = 42
N_RUNS = 5                        # independently trained models (stability)
MAX_TRAIN_PER_CLASS = 5000        # cap per class in the training pool
SUBSAMPLE_FRAC = 0.8              # each run trains on this fraction of the pool
PER_CLASS_CANDIDATES = 40         # candidate test instances per class
PER_CLASS_EXPLAIN = 10            # explained instances per class (after filtering)
MAX_EXPLAIN = 200                 # overall cap on explained instances
K_LIST = [1, 3, 5, 10]            # deletion sizes
STAB_K = [3, 5, 10]               # top-k sizes for Jaccard stability
N_RANDOM = 10                     # random-removal repetitions
LIME_SAMPLES = 5000
LIME_BG = 10000                   # background rows given to LIME

if QUICK_TEST:
    N_RUNS, MAX_TRAIN_PER_CLASS = 2, 500
    PER_CLASS_CANDIDATES, PER_CLASS_EXPLAIN, MAX_EXPLAIN = 6, 2, 30
    LIME_SAMPLES, LIME_BG, N_RANDOM = 500, 2000, 3

RESULTS_PATH = f"results_xai_{DATASET}/"
os.makedirs(RESULTS_PATH, exist_ok=True)

EDGE_LABEL, EDGE_BINARY = "Attack_type", "Attack_label"
EDGE_DROP = [
    "frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
    "icmp.transmit_timestamp", "http.file_data", "http.request.uri.query", "http.request.full_uri",
    "tcp.options", "tcp.payload", "tcp.srcport", "tcp.dstport", "udp.port", "mqtt.msg",
]


# =========================================================
# DATA LOADING
# =========================================================
def load_edge(path):
    df = pd.read_csv(path, low_memory=False)
    df = df.drop(columns=[c for c in EDGE_DROP + [EDGE_BINARY] if c in df.columns])
    df = df.dropna(subset=[EDGE_LABEL]).drop_duplicates().reset_index(drop=True)
    feats = [c for c in df.columns if c != EDGE_LABEL]
    const = [c for c in feats if df[c].nunique(dropna=False) <= 1]
    df = df.drop(columns=const)
    return df.drop(columns=[EDGE_LABEL]), df[EDGE_LABEL].astype(str)


def load_cic(folder):
    files = glob.glob(os.path.join(folder, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {folder}")
    df = pd.concat([pd.read_csv(f, nrows=CIC_NROWS_PER_FILE) for f in files], ignore_index=True)
    label = next((c for c in df.columns if c.lower() in
                  ['label', 'labels', 'class', 'attack', 'type', 'category']), df.columns[-1])
    df = df.dropna(subset=[label]).reset_index(drop=True)
    y = df[label].astype(str).apply(
        lambda x: 'BENIGN' if x.strip().lower() in ['benign', 'normal', 'normal traffic'] else x.strip())
    return df.drop(columns=[label]), y


def preprocess_features(X_train, X_test):
    X_train, X_test = X_train.copy(), X_test.copy()
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    for c in object_cols[:]:
        tr = pd.to_numeric(X_train[c], errors="coerce")
        te = pd.to_numeric(X_test[c], errors="coerce")
        if tr.notna().mean() > 0.95:
            X_train[c], X_test[c] = tr, te
            numeric_cols.append(c)
            object_cols.remove(c)

    for c in object_cols:
        le = LabelEncoder()
        tr = X_train[c].fillna("MISSING").astype(str)
        te = X_test[c].fillna("MISSING").astype(str)
        le.fit(tr)
        mp = {cls: i for i, cls in enumerate(le.classes_)}
        X_train[c] = tr.map(mp).astype(int)
        X_test[c] = te.map(lambda x: mp.get(x, -1)).astype(int)

    for c in numeric_cols:
        med = X_train[c].median()
        X_train[c] = X_train[c].fillna(med)
        X_test[c] = X_test[c].fillna(med)

    if numeric_cols:
        sc = StandardScaler()
        X_train[numeric_cols] = sc.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = sc.transform(X_test[numeric_cols])
    return X_train, X_test


def cap_per_class(idx, y, cap, rng):
    out = []
    for c in np.unique(y[idx]):
        ci = idx[y[idx] == c]
        if len(ci) > cap:
            ci = rng.choice(ci, size=cap, replace=False)
        out.append(ci)
    return np.concatenate(out)


def sample_per_class(idx, y, per_class, rng):
    out = []
    for c in np.unique(y[idx]):
        ci = idx[y[idx] == c]
        if len(ci) > per_class:
            ci = rng.choice(ci, size=per_class, replace=False)
        out.append(ci)
    return np.concatenate(out)


# =========================================================
# EXPLAINERS
# =========================================================
def shap_attributions(rf, Xe, cidx):
    """SHAP values of the TRUE class (= predicted class, instances are filtered to be correct)."""
    explainer = shap.TreeExplainer(rf)
    sv = explainer.shap_values(Xe, check_additivity=False)
    if isinstance(sv, list):                       # older shap: list of K arrays (n, f)
        sv = np.stack(sv, axis=-1)
    sv = np.asarray(sv)
    n = Xe.shape[0]
    if sv.ndim != 3 or sv.shape[0] != n:
        raise RuntimeError(f"Unexpected SHAP output shape {sv.shape}; expected (n, features, classes)")
    return sv[np.arange(n), :, cidx]


def lime_attributions(rf, explainer, Xe, cidx, n_feat):
    A = np.zeros((Xe.shape[0], n_feat))
    for i in range(Xe.shape[0]):
        lab = int(cidx[i])
        exp = explainer.explain_instance(Xe[i], rf.predict_proba, labels=(lab,),
                                         num_features=n_feat, num_samples=LIME_SAMPLES)
        for j, w in exp.as_map()[lab]:
            A[i, j] = w
    return A


# =========================================================
# METRICS
# =========================================================
def deletion_scores(rf, Xe, cidx, A, med, k_list):
    """Remove the top-k most SUPPORTIVE features (largest signed attribution for the class)."""
    n = Xe.shape[0]
    rows = np.arange(n)[:, None]
    p0 = rf.predict_proba(Xe)[np.arange(n), cidx]
    order = np.argsort(-A, axis=1)
    out = {}
    for k in k_list:
        Xm = Xe.copy()
        cols = order[:, :k]
        Xm[rows, cols] = med[cols]
        proba = rf.predict_proba(Xm)
        out[k] = (p0 - proba[np.arange(n), cidx], (proba.argmax(1) != cidx).astype(float))
    return out


def random_deletion_scores(rf, Xe, cidx, med, k_list, rng):
    n, f = Xe.shape
    rows = np.arange(n)[:, None]
    p0 = rf.predict_proba(Xe)[np.arange(n), cidx]
    acc = {k: [np.zeros(n), np.zeros(n)] for k in k_list}
    for _ in range(N_RANDOM):
        order = rng.random((n, f)).argsort(axis=1)
        for k in k_list:
            Xm = Xe.copy()
            cols = order[:, :k]
            Xm[rows, cols] = med[cols]
            proba = rf.predict_proba(Xm)
            acc[k][0] += (p0 - proba[np.arange(n), cidx]) / N_RANDOM
            acc[k][1] += (proba.argmax(1) != cidx) / N_RANDOM
    return {k: (v[0], v[1]) for k, v in acc.items()}


def topk_mask(A, k):
    idx = np.argpartition(-np.abs(A), k - 1, axis=1)[:, :k]
    m = np.zeros(A.shape, dtype=bool)
    m[np.arange(A.shape[0])[:, None], idx] = True
    return m


def jaccard_topk(A, B, k):
    ma, mb = topk_mask(A, k), topk_mask(B, k)
    inter = (ma & mb).sum(1)
    union = (ma | mb).sum(1)
    return inter / union


def spearman_rows(A, B):
    ra = rankdata(np.abs(A), axis=1)
    rb = rankdata(np.abs(B), axis=1)
    ra = ra - ra.mean(1, keepdims=True)
    rb = rb - rb.mean(1, keepdims=True)
    den = np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (ra * rb).sum(1) / den


def safe_wilcoxon(a, b):
    try:
        return wilcoxon(a, b)[1]
    except ValueError:
        return 1.0


# =========================================================
# MAIN
# =========================================================
def main():
    t_all = time.time()
    rng = np.random.default_rng(SEED)

    print(f"Dataset: {DATASET} | QUICK_TEST={QUICK_TEST}")
    if DATASET == "edge":
        X_df, y_raw = load_edge(PATH_OVERRIDE or EDGE_PATH)
    elif DATASET == "cic":
        X_df, y_raw = load_cic(PATH_OVERRIDE or CIC_FOLDER)
    else:
        raise SystemExit("First argument must be 'edge' or 'cic'")

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = list(le.classes_)
    feature_names = list(X_df.columns)
    n_feat = len(feature_names)
    print(f"Data: {X_df.shape} | classes: {len(class_names)} | features: {n_feat}")

    # --- outer split + capped training pool
    idx_all = np.arange(len(y))
    tr_idx, te_idx = train_test_split(idx_all, test_size=0.20, stratify=y, random_state=SEED)
    pool_idx = cap_per_class(tr_idx, y, MAX_TRAIN_PER_CLASS, rng)
    cand_idx = sample_per_class(te_idx, y, PER_CLASS_CANDIDATES, rng)

    # preprocessing fitted on the pool; candidates transformed with it
    X_pool_df, X_cand_df = preprocess_features(X_df.iloc[pool_idx], X_df.iloc[cand_idx])
    X_pool, y_pool = X_pool_df.values.astype(float), y[pool_idx]
    X_cand, y_cand = X_cand_df.values.astype(float), y[cand_idx]
    med = np.median(X_pool, axis=0)
    del X_df

    # --- Phase 1: train N_RUNS forests on different subsamples
    print(f"\nTraining {N_RUNS} Random Forests ...")
    models = []
    for r in range(N_RUNS):
        rs = np.random.default_rng(SEED + 100 + r)
        sub = rs.choice(len(y_pool), size=int(SUBSAMPLE_FRAC * len(y_pool)), replace=False)
        rf = RandomForestClassifier(n_estimators=100, random_state=SEED + r, n_jobs=-1)
        rf.fit(X_pool[sub], y_pool[sub])
        models.append(rf)
        print(f"  run {r + 1}/{N_RUNS}: trained on {len(sub)} rows")
    classes_ref = models[0].classes_
    assert all(np.array_equal(m.classes_, classes_ref) for m in models), \
        "A class is missing in one of the subsamples; lower SUBSAMPLE_FRAC issues or raise the cap."

    # --- Phase 2: keep instances that ALL runs classify correctly
    ok = np.ones(len(y_cand), dtype=bool)
    for m in models:
        ok &= (m.predict(X_cand) == y_cand)
    keep_pos = np.flatnonzero(ok)
    keep_pos = sample_per_class(keep_pos, y_cand, PER_CLASS_EXPLAIN, rng)  # indexes into candidates
    if len(keep_pos) > MAX_EXPLAIN:
        keep_pos = rng.choice(keep_pos, size=MAX_EXPLAIN, replace=False)
    keep_pos = np.sort(keep_pos)
    Xe, ye = X_cand[keep_pos], y_cand[keep_pos]
    cidx = np.searchsorted(classes_ref, ye)
    n = len(ye)
    cnt = pd.Series([class_names[c] for c in ye]).value_counts()
    cnt.to_csv(os.path.join(RESULTS_PATH, "explained_instances_per_class.csv"), header=["n_instances"])
    print(f"\nExplained instances: {n} (classified correctly by all {N_RUNS} runs) "
          f"| classes covered: {cnt.shape[0]}/{len(class_names)}")

    # --- Phase 3: explanations, fidelity per run
    lime_bg = X_pool[rng.choice(len(X_pool), size=min(LIME_BG, len(X_pool)), replace=False)]
    attrs = {"SHAP": [], "LIME": []}
    drops = {(m, k): [] for m in ["SHAP", "LIME", "Random"] for k in K_LIST}
    flips = {(m, k): [] for m in ["SHAP", "LIME", "Random"] for k in K_LIST}
    t_shap, t_lime = [], []

    for r, rf in enumerate(models):
        print(f"\n--- run {r + 1}/{N_RUNS} ---")
        t0 = time.time()
        A_shap = shap_attributions(rf, Xe, cidx)
        t_shap.append((time.time() - t0) / n)
        print(f"  SHAP done ({time.time() - t0:.1f}s)")

        lime_exp = LimeTabularExplainer(
            lime_bg, feature_names=feature_names,
            class_names=[class_names[c] for c in rf.classes_],
            discretize_continuous=True, random_state=SEED + r, mode="classification")
        t0 = time.time()
        A_lime = lime_attributions(rf, lime_exp, Xe, cidx, n_feat)
        t_lime.append((time.time() - t0) / n)
        print(f"  LIME done ({time.time() - t0:.1f}s)")

        attrs["SHAP"].append(A_shap)
        attrs["LIME"].append(A_lime)

        for name, A in [("SHAP", A_shap), ("LIME", A_lime)]:
            res = deletion_scores(rf, Xe, cidx, A, med, K_LIST)
            for k in K_LIST:
                drops[(name, k)].append(res[k][0])
                flips[(name, k)].append(res[k][1])
        res = random_deletion_scores(rf, Xe, cidx, med, K_LIST, np.random.default_rng(SEED + 200 + r))
        for k in K_LIST:
            drops[("Random", k)].append(res[k][0])
            flips[("Random", k)].append(res[k][1])

    # =====================================================
    # 1) FIDELITY table
    # =====================================================
    rows, ptab = [], []
    for k in K_LIST:
        per_inst = {}
        for m in ["SHAP", "LIME", "Random"]:
            D = np.array(drops[(m, k)])               # (runs, n)
            F = np.array(flips[(m, k)])
            run_means = D.mean(1)
            per_inst[m] = D.mean(0)
            rows.append({"k removed": k, "Method": m,
                         "Mean prob. drop": D.mean(), "Std over runs": run_means.std(ddof=1) if N_RUNS > 1 else np.nan,
                         "Prediction flip rate": F.mean()})
        ptab.append({"k removed": k,
                     "p SHAP vs Random": safe_wilcoxon(per_inst["SHAP"], per_inst["Random"]),
                     "p LIME vs Random": safe_wilcoxon(per_inst["LIME"], per_inst["Random"]),
                     "p SHAP vs LIME": safe_wilcoxon(per_inst["SHAP"], per_inst["LIME"])})
    fid = pd.DataFrame(rows)
    fid.to_csv(os.path.join(RESULTS_PATH, "fidelity_deletion.csv"), index=False)
    fid_p = pd.DataFrame(ptab)
    fid_p.to_csv(os.path.join(RESULTS_PATH, "fidelity_wilcoxon_per_instance.csv"), index=False)

    plt.figure(figsize=(6, 4))
    for m, mk in [("SHAP", "o"), ("LIME", "s"), ("Random", "^")]:
        sub = fid[fid["Method"] == m]
        plt.errorbar(sub["k removed"], sub["Mean prob. drop"], yerr=sub["Std over runs"],
                     marker=mk, capsize=3, label=m)
    plt.xlabel("Number of top features removed (k)")
    plt.ylabel("Drop in predicted-class probability")
    plt.title(f"Deletion fidelity ({DATASET})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, "fidelity_curve.png"), dpi=300)
    plt.close()

    # =====================================================
    # 2) STABILITY (between runs) and 3) AGREEMENT (SHAP vs LIME, same run)
    # =====================================================
    stab_rows = []
    for m in ["SHAP", "LIME"]:
        pairs = list(combinations(range(N_RUNS), 2))
        for k in STAB_K:
            v = [np.nanmean(jaccard_topk(attrs[m][a], attrs[m][b], k)) for a, b in pairs]
            stab_rows.append({"Method": m, "Metric": f"Jaccard top-{k}",
                              "Mean": np.mean(v), "Std over run-pairs": np.std(v, ddof=1) if len(v) > 1 else np.nan})
        v = [np.nanmean(spearman_rows(attrs[m][a], attrs[m][b])) for a, b in pairs]
        stab_rows.append({"Method": m, "Metric": "Spearman (|attribution| ranks)",
                          "Mean": np.mean(v), "Std over run-pairs": np.std(v, ddof=1) if len(v) > 1 else np.nan})
    stab = pd.DataFrame(stab_rows)
    stab.to_csv(os.path.join(RESULTS_PATH, "stability_between_runs.csv"), index=False)

    agr_rows = []
    for k in STAB_K:
        v = [np.nanmean(jaccard_topk(attrs["SHAP"][r], attrs["LIME"][r], k)) for r in range(N_RUNS)]
        agr_rows.append({"Metric": f"Jaccard top-{k}", "Mean": np.mean(v),
                         "Std over runs": np.std(v, ddof=1) if N_RUNS > 1 else np.nan})
    v = [np.nanmean(spearman_rows(attrs["SHAP"][r], attrs["LIME"][r])) for r in range(N_RUNS)]
    agr_rows.append({"Metric": "Spearman (|attribution| ranks)", "Mean": np.mean(v),
                     "Std over runs": np.std(v, ddof=1) if N_RUNS > 1 else np.nan})
    agr = pd.DataFrame(agr_rows)
    agr.to_csv(os.path.join(RESULTS_PATH, "agreement_shap_vs_lime.csv"), index=False)

    # =====================================================
    # 4) timing and global SHAP importance
    # =====================================================
    timing = pd.DataFrame([{"Method": "SHAP (TreeExplainer)", "Seconds per explanation": np.mean(t_shap)},
                           {"Method": f"LIME ({LIME_SAMPLES} samples)", "Seconds per explanation": np.mean(t_lime)}])
    timing.to_csv(os.path.join(RESULTS_PATH, "explanation_time.csv"), index=False)

    glob_imp = np.mean([np.abs(A).mean(0) for A in attrs["SHAP"]], axis=0)
    gi = pd.DataFrame({"Feature": feature_names, "Mean |SHAP|": glob_imp}).sort_values(
        "Mean |SHAP|", ascending=False)
    gi.to_csv(os.path.join(RESULTS_PATH, "global_importance_shap.csv"), index=False)

    # =====================================================
    # PRINT
    # =====================================================
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n================ FIDELITY: deletion test ================")
    print(fid.round(4).to_string(index=False))
    print("\nPer-instance Wilcoxon p-values (descriptive; instances from the same models are not independent):")
    print(fid_p.round(5).to_string(index=False))
    print("\n================ STABILITY between runs ================")
    print(stab.round(4).to_string(index=False))
    print("\n================ AGREEMENT SHAP vs LIME (same run) ================")
    print(agr.round(4).to_string(index=False))
    print("\n================ COST ================")
    print(timing.round(4).to_string(index=False))
    print("\n================ TOP-10 FEATURES BY MEAN |SHAP| ================")
    print(gi.head(10).round(4).to_string(index=False))
    print(f"\nFinished. Files saved in: {RESULTS_PATH} | total time: {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
