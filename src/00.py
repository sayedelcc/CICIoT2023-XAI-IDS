# main.py -- CICIoT2023 + Controlled Class Imbalance + SHAP + LIME + Wilcoxon Test
# ---------------------------------------------------------------------------------

import os
import glob
import time

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score
)

import joblib
from scipy.stats import t, wilcoxon
import shap
from lime.lime_tabular import LimeTabularExplainer

import matplotlib.pyplot as plt


# =========================================================
# CONFIG
# =========================================================

DATA_FOLDER = r"C:\Users\Administrator\OneDrive\Desktop\archive"

# Number of rows read from EACH CSV file (Use None to read all rows)
SAMPLE_NROWS = 100000

# Ten repeated random stratified train-test splits.
N_RUNS = 10
RANDOM_SEEDS = list(range(42, 52))

TEST_SIZE = 0.20

# Maximum number of training samples retained/target per class.
MAX_TRAIN_SAMPLES_PER_CLASS = 20000

SAVE_MODEL_PATH = "models/"
RESULTS_PATH = "results/"

os.makedirs(SAVE_MODEL_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def find_label_column(df):
    for c in df.columns:
        if c.lower() in ['label', 'labels', 'class', 'attack', 'type', 'category']:
            return c
    return df.columns[-1]


# =========================================================
# LOAD DATA
# =========================================================

def load_csvs_concat(folder, nrows=None):
    files = glob.glob(os.path.join(folder, "*.csv"))
    if len(files) == 0:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    print("\nNumber of CSV files found:", len(files))
    dfs = []
    for f in files:
        print("Reading file:", os.path.basename(f))
        df = pd.read_csv(f, nrows=nrows)
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    print("\nConcatenated dataset shape:", df_all.shape)
    return df_all


# =========================================================
# DATA INSPECTION
# =========================================================

def inspect_df(df, n=5):
    print("\n================ DATASET INFORMATION ================")
    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nMissing values:\n", df.isna().sum().sort_values(ascending=False).head(10))


# =========================================================
# CLASS DISTRIBUTION
# =========================================================

def analyze_class_distribution(df, label_col):
    print("\n================ CLASS DISTRIBUTION ================")
    counts = df[label_col].value_counts().sort_values(ascending=False)
    percentages = counts / len(df) * 100

    distribution = pd.DataFrame({
        "Attack Class": counts.index,
        "Number of Samples": counts.values,
        "Percentage": percentages.values
    })

    print(distribution.to_string(index=False))
    distribution.to_csv(os.path.join(RESULTS_PATH, "class_distribution.csv"), index=False)

    plt.figure(figsize=(14, 7))
    plt.bar(distribution["Attack Class"], distribution["Number of Samples"])
    plt.yscale("log")
    plt.xlabel("Attack Class")
    plt.ylabel("Number of Samples (log scale)")
    plt.title("Class Distribution of CICIoT2023 Dataset")
    plt.xticks(rotation=90, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, "class_distribution.png"), dpi=300)
    plt.close()

    return distribution


# =========================================================
# PREPROCESSING
# =========================================================

def prepare_raw_data(df, label_col=None):
    df = df.copy()
    if label_col is None:
        label_col = find_label_column(df)

    print("\nDetected label column:", label_col)
    df = df.dropna(subset=[label_col]).reset_index(drop=True)

    df[label_col] = df[label_col].astype(str).apply(
        lambda x: 'BENIGN' if x.strip().lower() in ['benign', 'normal', 'normal traffic'] else x.strip()
    )

    y = df[label_col].copy()
    X = df.drop(columns=[label_col]).copy()
    return X, y, label_col


def preprocess_features(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    for c in object_cols[:]:
        train_converted = pd.to_numeric(X_train[c], errors="coerce")
        test_converted = pd.to_numeric(X_test[c], errors="coerce")
        if train_converted.notna().mean() > 0.95:
            X_train[c] = train_converted
            X_test[c] = test_converted
            numeric_cols.append(c)
            object_cols.remove(c)

    encoders = {}
    for c in object_cols:
        le = LabelEncoder()
        train_values = X_train[c].fillna("MISSING").astype(str)
        test_values = X_test[c].fillna("MISSING").astype(str)
        le.fit(train_values)
        known_classes = set(le.classes_)

        X_train[c] = train_values.map(lambda x: le.transform([x])[0] if x in known_classes else -1)
        X_test[c] = test_values.map(lambda x: le.transform([x])[0] if x in known_classes else -1)
        encoders[c] = le

    medians = {}
    for c in numeric_cols:
        median_value = X_train[c].median()
        medians[c] = median_value
        X_train[c] = X_train[c].fillna(median_value)
        X_test[c] = X_test[c].fillna(median_value)

    scaler = StandardScaler()
    if numeric_cols:
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    preprocessing_objects = {
        "scaler": scaler,
        "encoders": encoders,
        "medians": medians,
        "numeric_cols": numeric_cols
    }

    return X_train, X_test, preprocessing_objects


# =========================================================
# CONTROLLED SAMPLING / BALANCING
# =========================================================

def cap_training_set_per_class(X_train_raw, y_train, max_samples_per_class=MAX_TRAIN_SAMPLES_PER_CLASS, random_state=42):
    rng = np.random.default_rng(random_state)
    y_series = pd.Series(y_train, index=X_train_raw.index)

    selected_indices = []
    for class_value in np.sort(y_series.unique()):
        class_indices = y_series[y_series == class_value].index.to_numpy()
        if len(class_indices) > max_samples_per_class:
            selected = rng.choice(class_indices, size=max_samples_per_class, replace=False)
        else:
            selected = class_indices
        selected_indices.extend(selected.tolist())

    rng.shuffle(selected_indices)
    X_sampled = X_train_raw.loc[selected_indices].reset_index(drop=True)
    y_sampled = y_series.loc[selected_indices].to_numpy()
    return X_sampled, y_sampled


def oversample_minority_classes(X_train, y_train, target_samples_per_class=MAX_TRAIN_SAMPLES_PER_CLASS, random_state=42):
    rng = np.random.default_rng(random_state)
    y_array = np.array(y_train)
    class_values, class_counts = np.unique(y_array, return_counts=True)

    X_parts, y_parts = [], []
    for class_value, class_count in zip(class_values, class_counts):
        class_indices = np.flatnonzero(y_array == class_value)
        if class_count < target_samples_per_class:
            additional_count = target_samples_per_class - class_count
            additional_indices = rng.choice(class_indices, size=additional_count, replace=True)
            selected_indices = np.concatenate([class_indices, additional_indices])
        else:
            selected_indices = class_indices

        X_parts.append(X_train[selected_indices])
        y_parts.append(y_array[selected_indices])

    X_balanced = np.concatenate(X_parts, axis=0)
    y_balanced = np.concatenate(y_parts, axis=0)

    shuffle_indices = rng.permutation(len(y_balanced))
    return X_balanced[shuffle_indices], y_balanced[shuffle_indices]


# =========================================================
# TRAIN MODELS
# =========================================================

def train_models(X_train, y_train, balanced=False, random_state=42):
    models = {}
    print(f"\nTraining models | Oversampled: {balanced}")

    rf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    rf.fit(X_train, y_train)
    models["RandomForest"] = rf

    xgb = XGBClassifier(n_estimators=100, eval_metric="mlogloss", random_state=random_state, n_jobs=-1)
    xgb.fit(X_train, y_train)
    models["XGBoost"] = xgb

    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100, random_state=random_state)
    mlp.fit(X_train, y_train)
    models["MLP"] = mlp

    return models


# =========================================================
# EVALUATION
# =========================================================

def evaluate_model(model, X_test, y_test, model_name, condition, class_names=None, run_id=None, save_detailed=True):
    start_time = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_time = time.perf_counter() - start_time
    avg_time_per_sample = inference_time / len(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    macro_precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    if save_detailed:
        run_suffix = f"_Run{run_id}" if run_id is not None else ""
        report = classification_report(
            y_test, y_pred, labels=np.arange(len(class_names)) if class_names else None,
            target_names=class_names, zero_division=0, output_dict=True
        )
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(os.path.join(RESULTS_PATH, f"{model_name}_{condition}{run_suffix}_classification_report.csv"))

        cm = confusion_matrix(y_test, y_pred)
        pd.DataFrame(cm).to_csv(os.path.join(RESULTS_PATH, f"{model_name}_{condition}{run_suffix}_confusion_matrix.csv"), index=False)

        plt.figure(figsize=(12, 10))
        plt.imshow(cm, aspect="auto")
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.title(f"Confusion Matrix - {model_name} {condition}{run_suffix}")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_PATH, f"{model_name}_{condition}{run_suffix}_confusion_matrix.png"), dpi=300)
        plt.close()

    return {
        "Run": run_id,
        "Model": model_name,
        "Condition": condition,
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_accuracy,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
        "Inference Time (sec)": inference_time,
        "Inference Time (ms/sample)": avg_time_per_sample * 1000
    }


# =========================================================
# STATISTICAL SIGNIFICANCE TESTS (WILCOXON SIGNED-RANK TEST)
# =========================================================

def run_statistical_tests(results_df):
    print("\n================ STATISTICAL SIGNIFICANCE TESTS (WILCOXON) ================")
    rows = []
    metric_columns = ["Accuracy", "Balanced Accuracy", "Macro F1", "Weighted F1"]

    for condition in results_df["Condition"].unique():
        sub_df = results_df[results_df["Condition"] == condition]
        pairs = [("RandomForest", "XGBoost"), ("RandomForest", "MLP"), ("XGBoost", "MLP")]

        for model_a, model_b in pairs:
            scores_a = sub_df[sub_df["Model"] == model_a].sort_values("Run")
            scores_b = sub_df[sub_df["Model"] == model_b].sort_values("Run")

            if len(scores_a) < 2 or len(scores_b) < 2:
                continue

            for metric in metric_columns:
                val_a = scores_a[metric].values
                val_b = scores_b[metric].values
                stat, p_val = wilcoxon(val_a, val_b)

                rows.append({
                    "Condition": condition,
                    "Comparison": f"{model_a} vs {model_b}",
                    "Metric": metric,
                    "Wilcoxon Statistic": stat,
                    "p-value": p_val,
                    "Significant (p < 0.05)": p_val < 0.05
                })
                print(f"[{condition}] {model_a} vs {model_b} | {metric}: p-value = {p_val:.5f}")

    stat_df = pd.DataFrame(rows)
    stat_file = os.path.join(RESULTS_PATH, "wilcoxon_statistical_tests.csv")
    stat_df.to_csv(stat_file, index=False)
    print(f"\nStatistical significance results saved to: {stat_file}")
    return stat_df


def summarize_repeated_results(results_df):
    metric_columns = ["Accuracy", "Balanced Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1"]
    rows = []
    for (condition, model_name), group in results_df.groupby(["Condition", "Model"]):
        row = {"Condition": condition, "Model": model_name, "Runs": len(group)}
        for metric in metric_columns:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric} Mean"] = np.mean(values)
            row[f"{metric} Std"] = np.std(values, ddof=1) if len(values) > 1 else np.nan
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(os.path.join(RESULTS_PATH, "repeated_10_runs_summary_mean_std.csv"), index=False)
    return summary_df


# =========================================================
# EXPLAINABILITY (SHAP & LIME)
# =========================================================

def explain_shap(model, X_sample, feature_names, save_name="shap_rf.png"):
    print("\nComputing SHAP values for Random Forest...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, save_name), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary plot saved to: {os.path.join(RESULTS_PATH, save_name)}")


def explain_lime(model, X_train_np, X_test_np, y_test, feature_names, idx=0):
    print("\nComputing LIME explanation...")
    explainer = LimeTabularExplainer(
        X_train_np,
        feature_names=feature_names,
        class_names=[str(i) for i in np.unique(y_test)],
        discretize_continuous=True,
        random_state=42
    )

    inst = X_test_np[idx]
    explanation = explainer.explain_instance(inst, model.predict_proba, num_features=10)
    explanation.save_to_file(os.path.join(RESULTS_PATH, "lime_example.html"))
    print(f"LIME explanation saved to: {os.path.join(RESULTS_PATH, 'lime_example.html')}")


# =========================================================
# MAIN PIPELINE
# =========================================================

def main():
    df = load_csvs_concat(DATA_FOLDER, nrows=SAMPLE_NROWS)
    inspect_df(df)

    X_raw, y_raw, label_col = prepare_raw_data(df)
    analyze_class_distribution(df, label_col)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = [str(cls) for cls in label_encoder.classes_]

    results_list = []
    last_rf_model = None
    last_X_train_bal = None
    last_X_test_prep = None

    for run_id, seed in enumerate(RANDOM_SEEDS, start=1):
        print(f"\n================ RUN {run_id}/{N_RUNS} (Seed: {seed}) ================")

        X_train_raw, X_test_raw, y_train_raw, y_test = train_test_split(
            X_raw, y, test_size=TEST_SIZE, stratify=y, random_state=seed
        )

        X_train_capped, y_train_capped = cap_training_set_per_class(
            X_train_raw, y_train_raw, max_samples_per_class=MAX_TRAIN_SAMPLES_PER_CLASS, random_state=seed
        )

        X_train_prep, X_test_prep, _ = preprocess_features(X_train_capped, X_test_raw)

        # Baseline Models (Before Oversampling)
        models_before = train_models(X_train_prep.values, y_train_capped, balanced=False, random_state=seed)
        for name, model in models_before.items():
            res = evaluate_model(model, X_test_prep.values, y_test, name, "Before_Balancing", class_names, run_id)
            results_list.append(res)

        # Oversampled Models (After Oversampling)
        X_train_bal, y_train_bal = oversample_minority_classes(
            X_train_prep.values, y_train_capped, target_samples_per_class=MAX_TRAIN_SAMPLES_PER_CLASS, random_state=seed
        )
        models_after = train_models(X_train_bal, y_train_bal, balanced=True, random_state=seed)
        for name, model in models_after.items():
            res = evaluate_model(model, X_test_prep.values, y_test, name, "After_Balancing", class_names, run_id)
            results_list.append(res)

        # Save reference for XAI from last run
        if run_id == N_RUNS:
            last_rf_model = models_after["RandomForest"]
            last_X_train_bal = X_train_bal
            last_X_test_prep = X_test_prep
            feature_names = X_train_prep.columns.tolist()

    # Save summary results
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(os.path.join(RESULTS_PATH, "repeated_10_runs_all_results.csv"), index=False)

    summarize_repeated_results(results_df)
    run_statistical_tests(results_df)

    # Compute XAI explanations (SHAP & LIME)
    if last_rf_model is not None:
        explain_shap(last_rf_model, last_X_test_prep.values[:200], feature_names)
        explain_lime(last_rf_model, last_X_train_bal, last_X_test_prep.values, y_test, feature_names)

    print("\nPipeline execution finished successfully!")

if __name__ == "__main__":
    main()
