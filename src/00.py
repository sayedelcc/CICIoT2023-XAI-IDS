# main.py -- CICIoT2023 + Controlled Class Imbalance + SHAP + LIME
# -----------------------------------------------------

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
from scipy.stats import t
import shap
from lime.lime_tabular import LimeTabularExplainer

import matplotlib.pyplot as plt


# =========================================================
# CONFIG
# =========================================================

DATA_FOLDER = r"C:\Users\Administrator\OneDrive\Desktop\archive"

# Number of rows read from EACH CSV file
# Use None to read all rows
SAMPLE_NROWS = 100000

# Ten repeated random stratified train-test splits.
# Each run uses a different seed while keeping an 80/20 split.
N_RUNS = 10
RANDOM_SEEDS = list(range(42, 52))

TEST_SIZE = 0.20

# Maximum number of training samples retained from each class.
# This controls memory usage while preserving all minority classes.
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

        if c.lower() in [
            'label',
            'labels',
            'class',
            'attack',
            'type',
            'category'
        ]:

            return c

    return df.columns[-1]


# =========================================================
# LOAD DATA
# =========================================================

def load_csvs_concat(folder, nrows=None):

    files = glob.glob(os.path.join(folder, "*.csv"))

    if len(files) == 0:
        raise FileNotFoundError(
            f"No CSV files found in {folder}"
        )

    print("\nNumber of CSV files:", len(files))

    dfs = []

    for f in files:

        print("Reading:", os.path.basename(f))

        df = pd.read_csv(
            f,
            nrows=nrows
        )

        dfs.append(df)

    df_all = pd.concat(
        dfs,
        ignore_index=True
    )

    print(
        "\nConcatenated dataset shape:",
        df_all.shape
    )

    return df_all


# =========================================================
# DATA INSPECTION
# =========================================================

def inspect_df(df, n=5):

    print("\n================ DATASET INFORMATION ================")

    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes.head(20))

    print("\nMissing values:")
    print(
        df.isna()
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    print("\nExample rows:")
    print(df.head(n))


# =========================================================
# CLASS DISTRIBUTION
# =========================================================

def analyze_class_distribution(df, label_col):

    print("\n================ CLASS DISTRIBUTION ================")

    counts = (
        df[label_col]
        .value_counts()
        .sort_values(ascending=False)
    )

    percentages = (
        counts / len(df) * 100
    )

    distribution = pd.DataFrame({
        "Attack Class": counts.index,
        "Number of Samples": counts.values,
        "Percentage": percentages.values
    })

    print("\nClass distribution:")
    print(distribution.to_string(index=False))

    # Save table
    distribution.to_csv(
        os.path.join(
            RESULTS_PATH,
            "class_distribution.csv"
        ),
        index=False
    )

    # Plot
    plt.figure(figsize=(14, 7))

    plt.bar(
        distribution["Attack Class"],
        distribution["Number of Samples"]
    )

    plt.yscale("log")

    plt.xlabel("Attack Class")
    plt.ylabel("Number of Samples (log scale)")
    plt.title(
        "Class Distribution of CICIoT2023 Dataset"
    )

    plt.xticks(
        rotation=90,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_PATH,
            "class_distribution.png"
        ),
        dpi=300
    )

    plt.close()

    print(
        "\nClass distribution table saved to:",
        os.path.join(
            RESULTS_PATH,
            "class_distribution.csv"
        )
    )

    print(
        "Class distribution figure saved to:",
        os.path.join(
            RESULTS_PATH,
            "class_distribution.png"
        )
    )

    return distribution


# =========================================================
# PREPROCESSING
# =========================================================

def prepare_raw_data(df, label_col=None):

    df = df.copy()

    if label_col is None:
        label_col = find_label_column(df)

    print(
        "\nDetected label column:",
        label_col
    )

    # Remove rows with missing labels
    df = df.dropna(
        subset=[label_col]
    ).reset_index(drop=True)

    # Normalize benign labels
    df[label_col] = (
        df[label_col]
        .astype(str)
        .apply(
            lambda x:
            'BENIGN'
            if x.strip().lower()
            in [
                'benign',
                'normal',
                'normal traffic'
            ]
            else x.strip()
        )
    )

    y = df[label_col].copy()

    X = df.drop(
        columns=[label_col]
    ).copy()

    return X, y, label_col


# =========================================================
# PREPROCESS FEATURES
# =========================================================

def preprocess_features(
    X_train,
    X_test
):

    X_train = X_train.copy()
    X_test = X_test.copy()

    # -----------------------------------------
    # Identify numeric and categorical columns
    # -----------------------------------------

    numeric_cols = (
        X_train
        .select_dtypes(
            include=[np.number]
        )
        .columns
        .tolist()
    )

    object_cols = (
        X_train
        .select_dtypes(
            include=[
                'object',
                'category'
            ]
        )
        .columns
        .tolist()
    )

    # -----------------------------------------
    # Convert object columns to numeric
    # when possible
    # -----------------------------------------

    for c in object_cols[:]:

        train_converted = pd.to_numeric(
            X_train[c],
            errors="coerce"
        )

        test_converted = pd.to_numeric(
            X_test[c],
            errors="coerce"
        )

        # If most training values can be converted
        if train_converted.notna().mean() > 0.95:

            X_train[c] = train_converted
            X_test[c] = test_converted

            numeric_cols.append(c)
            object_cols.remove(c)

    # -----------------------------------------
    # Encode remaining categorical columns
    # -----------------------------------------

    encoders = {}

    for c in object_cols:

        le = LabelEncoder()

        # Handle missing values
        train_values = (
            X_train[c]
            .fillna("MISSING")
            .astype(str)
        )

        test_values = (
            X_test[c]
            .fillna("MISSING")
            .astype(str)
        )

        # Fit encoder ONLY on training data
        le.fit(train_values)

        # Unknown categories in test
        # are mapped to -1
        known_classes = set(
            le.classes_
        )

        X_train[c] = train_values.map(
            lambda x:
            le.transform([x])[0]
            if x in known_classes
            else -1
        )

        X_test[c] = test_values.map(
            lambda x:
            le.transform([x])[0]
            if x in known_classes
            else -1
        )

        encoders[c] = le

    # -----------------------------------------
    # Fill numeric missing values
    # using TRAINING medians only
    # -----------------------------------------

    medians = {}

    for c in numeric_cols:

        median_value = X_train[c].median()

        medians[c] = median_value

        X_train[c] = (
            X_train[c]
            .fillna(median_value)
        )

        X_test[c] = (
            X_test[c]
            .fillna(median_value)
        )

    # -----------------------------------------
    # Standardization
    # -----------------------------------------

    scaler = StandardScaler()

    if numeric_cols:

        X_train[numeric_cols] = (
            scaler.fit_transform(
                X_train[numeric_cols]
            )
        )

        X_test[numeric_cols] = (
            scaler.transform(
                X_test[numeric_cols]
            )
        )

    preprocessing_objects = {
        "scaler": scaler,
        "encoders": encoders,
        "medians": medians,
        "numeric_cols": numeric_cols
    }

    return (
        X_train,
        X_test,
        preprocessing_objects
    )


# =========================================================
# CONTROLLED TRAINING SAMPLING / BALANCING
# =========================================================

def cap_training_set_per_class(
    X_train_raw,
    y_train,
    max_samples_per_class=MAX_TRAIN_SAMPLES_PER_CLASS,
    random_state=42
):
    """
    Randomly retain at most max_samples_per_class examples from
    each class in the TRAINING SET only.

    The test set is NOT downsampled, so final evaluation keeps the
    original class distribution.
    """
    rng = np.random.default_rng(random_state)

    y_series = pd.Series(
        y_train,
        index=X_train_raw.index
    )

    selected_indices = []

    for class_value in np.sort(y_series.unique()):
        class_indices = y_series[
            y_series == class_value
        ].index.to_numpy()

        if len(class_indices) > max_samples_per_class:
            selected = rng.choice(
                class_indices,
                size=max_samples_per_class,
                replace=False
            )
        else:
            selected = class_indices

        selected_indices.extend(selected.tolist())

    rng.shuffle(selected_indices)

    X_sampled = X_train_raw.loc[
        selected_indices
    ].reset_index(drop=True)

    y_sampled = y_series.loc[
        selected_indices
    ].to_numpy()

    return X_sampled, y_sampled


def calculate_class_weights(y_train):
    return compute_sample_weight(
        class_weight="balanced",
        y=y_train
    )


def oversample_minority_classes(
    X_train,
    y_train,
    target_samples_per_class=MAX_TRAIN_SAMPLES_PER_CLASS,
    random_state=42
):
    """
    Randomly oversample minority classes in the TRAINING SET only.

    Each class with fewer than target_samples_per_class instances is
    increased by sampling existing training instances with replacement.
    Classes already at the target are left unchanged.

    The held-out test set is never passed to this function.
    """
    rng = np.random.default_rng(random_state)

    y_array = np.asarray(y_train)
    class_values, class_counts = np.unique(
        y_array, return_counts=True
    )

    X_parts = []
    y_parts = []

    print("\n================================================")
    print("MINORITY CLASS OVERSAMPLING")
    print("================================================")
    print("Target training samples per class:", target_samples_per_class)

    for class_value, class_count in zip(class_values, class_counts):
        class_indices = np.flatnonzero(y_array == class_value)

        if class_count < target_samples_per_class:
            additional_count = target_samples_per_class - class_count
            additional_indices = rng.choice(
                class_indices,
                size=additional_count,
                replace=True
            )
            selected_indices = np.concatenate([
                class_indices, additional_indices
            ])
            print(
                f"Class {class_value}: {class_count:,} -> "
                f"{len(selected_indices):,} samples"
            )
        else:
            selected_indices = class_indices
            print(
                f"Class {class_value}: {class_count:,} -> "
                f"{class_count:,} samples (unchanged)"
            )

        X_parts.append(X_train[selected_indices])
        y_parts.append(y_array[selected_indices])

    X_balanced = np.concatenate(X_parts, axis=0)
    y_balanced = np.concatenate(y_parts, axis=0)

    shuffle_indices = rng.permutation(len(y_balanced))
    X_balanced = X_balanced[shuffle_indices]
    y_balanced = y_balanced[shuffle_indices]

    print(
        "Total training samples after oversampling:",
        len(y_balanced)
    )

    return X_balanced, y_balanced


def save_training_distribution(y, filename):
    counts = pd.Series(y).value_counts().sort_index()

    distribution = pd.DataFrame({
        "Encoded Class": counts.index,
        "Number of Training Samples": counts.values,
        "Percentage": (
            counts.values / len(y) * 100
        )
    })

    distribution.to_csv(
        os.path.join(
            RESULTS_PATH,
            filename
        ),
        index=False
    )

    return distribution


# =========================================================
# TRAIN MODELS
# =========================================================

def train_models(
    X_train,
    y_train,
    balanced=False,
    random_state=42
):

    models = {}

    print("\n================================================")
    print("Training models | Oversampled:", balanced)
    print("================================================")

    # For the after-balancing condition, the training data have already
    # been explicitly oversampled so that minority classes are increased
    # to the controlled target size. No additional class weighting is
    # applied, allowing the experiment to isolate the effect of
    # increasing minority-class representation.

    # =================================================
    # RANDOM FOREST
    # =================================================

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1
    )

    print("\nTraining Random Forest...")
    start_time = time.perf_counter()
    rf.fit(X_train, y_train)
    rf_training_time = time.perf_counter() - start_time

    print(
        "Random Forest training time:",
        round(rf_training_time, 4),
        "seconds"
    )
    models["RandomForest"] = rf

    # =================================================
    # XGBOOST
    # =================================================

    xgb = XGBClassifier(
        n_estimators=100,
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1
    )

    print("\nTraining XGBoost...")
    start_time = time.perf_counter()
    xgb.fit(X_train, y_train)
    xgb_training_time = time.perf_counter() - start_time

    print(
        "XGBoost training time:",
        round(xgb_training_time, 4),
        "seconds"
    )
    models["XGBoost"] = xgb

    # =================================================
    # MLP
    # =================================================

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=100,
        random_state=random_state
    )

    print("\nTraining MLP...")
    start_time = time.perf_counter()
    mlp.fit(X_train, y_train)
    mlp_training_time = time.perf_counter() - start_time

    print(
        "MLP training time:",
        round(mlp_training_time, 4),
        "seconds"
    )
    models["MLP"] = mlp

    return models


# =========================================================
# REPEATED RANDOM TRAIN / TEST EVALUATION
# =========================================================

def summarize_repeated_results(results_df):
    """Calculate mean, standard deviation, and 95% CI over repeated runs."""
    metric_columns = [
        "Accuracy",
        "Balanced Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
        "Weighted F1"
    ]

    rows = []
    for (condition, model_name), group in results_df.groupby(["Condition", "Model"]):
        row = {
            "Condition": condition,
            "Model": model_name,
            "Runs": len(group)
        }
        for metric in metric_columns:
            values = group[metric].to_numpy(dtype=float)
            mean_value = np.mean(values)
            std_value = np.std(values, ddof=1) if len(values) > 1 else np.nan
            if len(values) > 1:
                se = std_value / np.sqrt(len(values))
                t_critical = t.ppf(0.975, df=len(values) - 1)
                ci_half_width = t_critical * se
            else:
                ci_half_width = np.nan
            row[f"{metric} Mean"] = mean_value
            row[f"{metric} Std"] = std_value
            row[f"{metric} 95% CI Lower"] = mean_value - ci_half_width if np.isfinite(ci_half_width) else np.nan
            row[f"{metric} 95% CI Upper"] = mean_value + ci_half_width if np.isfinite(ci_half_width) else np.nan
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(
        os.path.join(RESULTS_PATH, "repeated_10_runs_summary_mean_std_ci.csv"),
        index=False
    )
    return summary_df


def paired_run_comparison(results_df, model_a="RandomForest", model_b="XGBoost"):
    """Exploratory paired comparison across the same 10 random splits."""
    metric_columns = [
        "Accuracy", "Balanced Accuracy", "Macro Precision",
        "Macro Recall", "Macro F1", "Weighted F1"
    ]
    rows = []
    for condition in results_df["Condition"].unique():
        subset = results_df[results_df["Condition"] == condition]
        a = subset[subset["Model"] == model_a].sort_values("Run")
        b = subset[subset["Model"] == model_b].sort_values("Run")
        if len(a) != len(b) or len(a) < 2:
            continue
        from scipy.stats import ttest_rel
        for metric in metric_columns:
            stat, p_value = ttest_rel(a[metric].to_numpy(), b[metric].to_numpy())
            rows.append({
                "Condition": condition,
                "Model A": model_a,
                "Model B": model_b,
                "Metric": metric,
                "Paired t-statistic": stat,
                "p-value": p_value
            })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS_PATH, "rf_vs_xgb_paired_ttest_10_runs.csv"), index=False)
    return out


# =========================================================
# EVALUATION
# =========================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    model_name,
    condition,
    class_names=None,
    run_id=None,
    save_detailed=True
):

    print("\n==============================================")
    print(
        f"{model_name} | {condition}"
    )
    print("==============================================")

    # Warm-up prediction
    model.predict(
        X_test[:min(10, len(X_test))]
    )

    # -----------------------------------------
    # Inference time
    # -----------------------------------------

    start_time = time.perf_counter()

    y_pred = model.predict(
        X_test
    )

    inference_time = (
        time.perf_counter()
        - start_time
    )

    avg_time_per_sample = (
        inference_time
        / len(X_test)
    )

    # -----------------------------------------
    # Metrics
    # -----------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        y_pred
    )

    macro_precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print(
        f"Accuracy:            {accuracy:.6f}"
    )

    print(
        f"Balanced Accuracy:   {balanced_accuracy:.6f}"
    )

    print(
        f"Macro Precision:     {macro_precision:.6f}"
    )

    print(
        f"Macro Recall:        {macro_recall:.6f}"
    )

    print(
        f"Macro F1:            {macro_f1:.6f}"
    )

    print(
        f"Weighted F1:         {weighted_f1:.6f}"
    )

    print(
        f"Inference Time:      {inference_time:.6f} sec"
    )

    print(
        f"Time per Sample:     "
        f"{avg_time_per_sample * 1000:.6f} ms"
    )

    # -----------------------------------------
    # Classification report
    # -----------------------------------------

    if class_names is None:
        report = classification_report(
            y_test,
            y_pred,
            zero_division=0,
            output_dict=True
        )
    else:
        report = classification_report(
            y_test,
            y_pred,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            zero_division=0,
            output_dict=True
        )

    report_df = pd.DataFrame(
        report
    ).transpose()

    if save_detailed:
        run_suffix = f"_Run{run_id}" if run_id is not None else ""
        report_file = os.path.join(
            RESULTS_PATH,
            f"{model_name}_{condition}{run_suffix}_classification_report.csv"
        )
        report_df.to_csv(report_file)

        # -----------------------------------------
        # Confusion Matrix
        # -----------------------------------------
        cm = confusion_matrix(y_test, y_pred)
        cm_file = os.path.join(
            RESULTS_PATH,
            f"{model_name}_{condition}{run_suffix}_confusion_matrix.csv"
        )
        pd.DataFrame(cm).to_csv(cm_file, index=False)

        plt.figure(figsize=(12, 10))
        plt.imshow(cm, aspect="auto")
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.title(f"Confusion Matrix - {model_name} - {condition}{run_suffix}")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(
            os.path.join(RESULTS_PATH, f"{model_name}_{condition}{run_suffix}_confusion_matrix.png"),
            dpi=300
        )
        plt.close()

    return {
        "Model": model_name,
        "Condition": condition,
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_accuracy,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
        "Inference Time (sec)": inference_time,
        "Inference Time (ms/sample)": (
            avg_time_per_sample * 1000
        )
    }


# =========================================================
# PER-CLASS BEFORE / AFTER COMPARISON
# =========================================================

def create_per_class_before_after_comparison(class_names):
    """
    Create a readable per-class comparison for Random Forest and
    XGBoost before and after imbalance mitigation.

    The comparison is based on the classification-report CSV files
    generated during evaluation. Only actual attack/traffic classes
    are included; aggregate rows such as accuracy and macro average
    are excluded.
    """

    rows = []

    selected_models = [
        "RandomForest",
        "XGBoost"
    ]

    for model_name in selected_models:
        before_file = os.path.join(
            RESULTS_PATH,
            f"{model_name}_Before_Balancing_Run10_classification_report.csv"
        )
        after_file = os.path.join(
            RESULTS_PATH,
            f"{model_name}_After_Balancing_Run10_classification_report.csv"
        )

        if not (os.path.exists(before_file) and os.path.exists(after_file)):
            print(f"Per-class comparison skipped for {model_name}: report file missing.")
            continue

        before = pd.read_csv(before_file, index_col=0)
        after = pd.read_csv(after_file, index_col=0)

        for class_name in class_names:
            if class_name not in before.index or class_name not in after.index:
                continue

            rows.append({
                "Model": model_name,
                "Attack Class": class_name,
                "Before Precision": before.loc[class_name, "precision"],
                "After Precision": after.loc[class_name, "precision"],
                "Precision Change": after.loc[class_name, "precision"] - before.loc[class_name, "precision"],
                "Before Recall": before.loc[class_name, "recall"],
                "After Recall": after.loc[class_name, "recall"],
                "Recall Change": after.loc[class_name, "recall"] - before.loc[class_name, "recall"],
                "Before F1": before.loc[class_name, "f1-score"],
                "After F1": after.loc[class_name, "f1-score"],
                "F1 Change": after.loc[class_name, "f1-score"] - before.loc[class_name, "f1-score"],
                "Test Support": int(after.loc[class_name, "support"])
            })

    if not rows:
        print("No per-class before/after results were generated.")
        return None

    comparison_df = pd.DataFrame(rows)

    comparison_file = os.path.join(
        RESULTS_PATH,
        "per_class_before_after_comparison_RF_XGB.csv"
    )
    comparison_df.to_csv(comparison_file, index=False)

    # Separate files for easier thesis table preparation.
    for model_name in selected_models:
        model_df = comparison_df[comparison_df["Model"] == model_name].copy()
        if not model_df.empty:
            model_df.to_csv(
                os.path.join(
                    RESULTS_PATH,
                    f"{model_name}_per_class_before_after.csv"
                ),
                index=False
            )

    # Highlight the classes with the largest recall and F1 improvements.
    ranking = comparison_df.copy()
    ranking["Absolute Recall Change"] = ranking["Recall Change"].abs()
    ranking["Absolute F1 Change"] = ranking["F1 Change"].abs()
    ranking = ranking.sort_values(
        ["Model", "Absolute Recall Change"],
        ascending=[True, False]
    )
    ranking.to_csv(
        os.path.join(
            RESULTS_PATH,
            "per_class_change_ranking_RF_XGB.csv"
        ),
        index=False
    )

    print("\n================================================")
    print("PER-CLASS BEFORE / AFTER COMPARISON")
    print("================================================")
    display_cols = [
        "Model", "Attack Class",
        "Before Recall", "After Recall", "Recall Change",
        "Before F1", "After F1", "F1 Change"
    ]
    print(comparison_df[display_cols].to_string(index=False))

    print("\nPer-class comparison saved to:")
    print(comparison_file)

    return comparison_df


# =========================================================
# SHAP
# =========================================================

def explain_shap(
    model,
    X_sample,
    feature_names,
    save_name="shap_summary.png"
):

    print("\nComputing SHAP...")

    start_time = time.perf_counter()

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = (
        explainer.shap_values(
            X_sample
        )
    )

    shap_time = (
        time.perf_counter()
        - start_time
    )

    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_PATH,
            save_name
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "SHAP time:",
        round(shap_time, 4),
        "seconds"
    )

    print(
        "Saved SHAP plot:",
        save_name
    )

    return shap_time


# =========================================================
# LIME
# =========================================================

def explain_lime(
    model,
    X_train_np,
    X_test_np,
    y_test,
    feature_names,
    idx=0
):

    print("\nComputing LIME...")

    start_time = time.perf_counter()

    explainer = LimeTabularExplainer(
        X_train_np,
        feature_names=feature_names,
        class_names=[
            str(i)
            for i in np.unique(y_test)
        ],
        discretize_continuous=True,
        random_state=42
    )

    inst = X_test_np[idx]

    explanation = explainer.explain_instance(
        inst,
        model.predict_proba,
        num_features=10
    )

    lime_time = (
        time.perf_counter()
        - start_time
    )

    print(
        "\nLIME explanation:"
    )

    print(
        explanation.as_list()
    )

    explanation.save_to_file(
        os.path.join(
            RESULTS_PATH,
            "lime_example.html"
        )
    )

    print(
        "LIME time:",
        round(lime_time, 4),
        "seconds"
    )

    print(
        "Saved lime_example.html"
    )

    return lime_time


# =========================================================
# MAIN
# =========================================================

def main():

    # =====================================================
    # 1. LOAD DATA
    # =====================================================
    df = load_csvs_concat(DATA_FOLDER, nrows=SAMPLE_NROWS)
    inspect_df(df)

    # =====================================================
    # 2. PREPARE DATA
    # =====================================================
    X_raw, y_raw, label_col = prepare_raw_data(df)
    analyze_class_distribution(df, label_col)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    print("\n================================================")
    print("10-REPEAT STRATIFIED RANDOM TRAIN / TEST EVALUATION")
    print("================================================")
    print(f"Number of runs: {N_RUNS}")
    print(f"Seeds: {RANDOM_SEEDS}")
    print(f"Train ratio: {1 - TEST_SIZE:.0%}")
    print(f"Test ratio: {TEST_SIZE:.0%}")
    print("Oversampling: training data only")
    print("Test data: untouched in every run")

    all_run_results = []
    last_run_objects = None

    for run_id, seed in enumerate(RANDOM_SEEDS, start=1):

        print("\n\n============================================================")
        print(f"RUN {run_id}/{N_RUNS} | RANDOM SEED = {seed}")
        print("============================================================")

        # -----------------------------------------------------
        # 3. RANDOM STRATIFIED 80/20 SPLIT
        # -----------------------------------------------------
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw,
            y,
            test_size=TEST_SIZE,
            train_size=1 - TEST_SIZE,
            stratify=y,
            random_state=seed
        )

        print("Training samples before controlled cap:", len(X_train_raw))
        print("Testing samples:", len(X_test_raw))

        # -----------------------------------------------------
        # 4. CONTROLLED TRAINING SAMPLING
        # -----------------------------------------------------
        X_train_raw, y_train = cap_training_set_per_class(
            X_train_raw,
            y_train,
            MAX_TRAIN_SAMPLES_PER_CLASS,
            random_state=seed
        )

        print("Controlled training samples:", len(X_train_raw))

        # -----------------------------------------------------
        # 5. PREPROCESS - FIT TRAIN ONLY
        # -----------------------------------------------------
        X_train, X_test, preproc = preprocess_features(
            X_train_raw,
            X_test_raw
        )

        print("Training shape after preprocessing:", X_train.shape)
        print("Testing shape after preprocessing:", X_test.shape)

        # -----------------------------------------------------
        # 6. BASELINE: NO OVERSAMPLING
        # -----------------------------------------------------
        baseline_models = train_models(
            X_train.values,
            y_train,
            balanced=False,
            random_state=seed
        )

        for name, model in baseline_models.items():
            result = evaluate_model(
                model,
                X_test.values,
                y_test,
                name,
                "Before_Balancing",
                class_names=label_encoder.classes_,
                run_id=run_id,
                save_detailed=(run_id == N_RUNS)
            )
            result["Run"] = run_id
            result["Random Seed"] = seed
            result["Train Samples"] = len(y_train)
            result["Test Samples"] = len(y_test)
            all_run_results.append(result)

        # -----------------------------------------------------
        # 7. RANDOM OVERSAMPLING - TRAINING ONLY
        # -----------------------------------------------------
        X_train_oversampled, y_train_oversampled = oversample_minority_classes(
            X_train.values,
            y_train,
            MAX_TRAIN_SAMPLES_PER_CLASS,
            random_state=seed
        )

        # -----------------------------------------------------
        # 8. AFTER OVERSAMPLING
        # -----------------------------------------------------
        balanced_models = train_models(
            X_train_oversampled,
            y_train_oversampled,
            balanced=True,
            random_state=seed
        )

        for name, model in balanced_models.items():
            result = evaluate_model(
                model,
                X_test.values,
                y_test,
                name,
                "After_Balancing",
                class_names=label_encoder.classes_,
                run_id=run_id,
                save_detailed=(run_id == N_RUNS)
            )
            result["Run"] = run_id
            result["Random Seed"] = seed
            result["Train Samples"] = len(y_train_oversampled)
            result["Test Samples"] = len(y_test)
            all_run_results.append(result)

        # Keep only the last run's models for detailed XAI and model saving.
        if run_id == N_RUNS:
            last_run_objects = {
                "baseline_models": baseline_models,
                "balanced_models": balanced_models,
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "preproc": preproc
            }

        # Release run-specific large objects.
        if run_id != N_RUNS:
            del baseline_models
            del balanced_models
        del X_train_raw, X_test_raw
        del X_train, X_test
        del X_train_oversampled, y_train_oversampled

    # =====================================================
    # 9. SAVE ALL 10-RUN RESULTS
    # =====================================================
    results_df = pd.DataFrame(all_run_results)
    results_df.to_csv(
        os.path.join(RESULTS_PATH, "10_runs_all_results.csv"),
        index=False
    )

    summary_df = summarize_repeated_results(results_df)

    paired_df = paired_run_comparison(
        results_df,
        model_a="RandomForest",
        model_b="XGBoost"
    )

    print("\n================================================")
    print("10-RUN SUMMARY: MEAN +/- STANDARD DEVIATION")
    print("================================================")
    display_columns = [
        "Condition", "Model", "Runs",
        "Accuracy Mean", "Accuracy Std",
        "Balanced Accuracy Mean", "Balanced Accuracy Std",
        "Macro Precision Mean", "Macro Precision Std",
        "Macro Recall Mean", "Macro Recall Std",
        "Macro F1 Mean", "Macro F1 Std",
        "Weighted F1 Mean", "Weighted F1 Std"
    ]
    print(summary_df[display_columns].to_string(index=False))

    print("\n10-run summary saved to:")
    print(os.path.join(RESULTS_PATH, "repeated_10_runs_summary_mean_std_ci.csv"))
    print("\nPaired RF vs XGBoost results saved to:")
    print(os.path.join(RESULTS_PATH, "rf_vs_xgb_paired_ttest_10_runs.csv"))

    # =====================================================
    # 10. DETAILED PER-CLASS ANALYSIS FOR FINAL RUN ONLY
    # =====================================================
    # Detailed per-class reports and confusion matrices are generated
    # only for the tenth run to avoid producing very large numbers of
    # duplicate files. The aggregate performance is based on all 10 runs.
    per_class_results = create_per_class_before_after_comparison(
        label_encoder.classes_
    )

    # =====================================================
    # 11. SHAP - FINAL RUN BALANCED RANDOM FOREST
    # =====================================================
    print("\nRunning SHAP on balanced Random Forest from Run 10...")

    X_shap = last_run_objects["X_test"].iloc[:min(200, len(last_run_objects["X_test"]))]

    shap_time = explain_shap(
        last_run_objects["balanced_models"]["RandomForest"],
        X_shap,
        list(last_run_objects["X_train"].columns),
        "shap_rf_balanced_run10.png"
    )

    # =====================================================
    # 12. LIME - FINAL RUN BALANCED RANDOM FOREST
    # =====================================================
    lime_time = explain_lime(
        last_run_objects["balanced_models"]["RandomForest"],
        last_run_objects["X_train"].values,
        last_run_objects["X_test"].values,
        last_run_objects["y_test"],
        list(last_run_objects["X_train"].columns),
        idx=5
    )

    # =====================================================
    # 13. SAVE FINAL RUN MODELS
    # =====================================================
    joblib.dump(
        last_run_objects["balanced_models"]["RandomForest"],
        os.path.join(SAVE_MODEL_PATH, "rf_balanced_model_run10.joblib")
    )

    label_mapping = pd.DataFrame({
        "Encoded Class": np.arange(len(label_encoder.classes_)),
        "Original Label": label_encoder.classes_
    })
    label_mapping.to_csv(
        os.path.join(RESULTS_PATH, "label_mapping.csv"),
        index=False
    )

    joblib.dump(
        last_run_objects["preproc"],
        os.path.join(SAVE_MODEL_PATH, "preprocessing_run10.joblib")
    )

    joblib.dump(
        label_encoder,
        os.path.join(SAVE_MODEL_PATH, "label_encoder.joblib")
    )

    pd.DataFrame({
        "Metric": ["SHAP Time (sec)", "LIME Time (sec)"],
        "Value": [shap_time, lime_time]
    }).to_csv(
        os.path.join(RESULTS_PATH, "xai_runtime_run10.csv"),
        index=False
    )

    print("\nModels and preprocessing objects saved.")
    print("All 10-run results saved in:", RESULTS_PATH)


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()