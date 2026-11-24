# Class-imbalance handling and explainability for multi-class IoT intrusion detection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23977260.svg)](https://doi.org/10.5281/zenodo.23977260)

Code accompanying the study of imbalance-handling strategies (class weighting, random oversampling, SMOTE,
Borderline-SMOTE, ADASYN) and of SHAP vs LIME explanations for tree-based and neural intrusion-detection
models on two independent datasets: **CICIoT2023** (34 classes) and **Edge-IIoTset** (15 classes).

## Repository layout

| File | Purpose |
|---|---|
| `leakage_check.py` | Checks exact-duplicate rows in CICIoT2023 and train/test overlap for the stratified split |
| `cic_strategies.py` | CICIoT2023: stratified 10-fold CV, Baseline / ClassWeight / RandomOverSampler / SMOTE with RF, XGBoost, MLP; Wilcoxon + Holm |
| `edge_strategies.py` | Edge-IIoTset: same protocol, six strategies (adds Borderline-SMOTE and ADASYN) |
| `edge_confusion.py` | Edge-IIoTset: pooled confusion matrices of RF and XGBoost for the rare classes (Fingerprinting, MITM); shows where false positives and false negatives come from (imports code from `edge_strategies.py`) |
| `edge_text_sensitivity.py` | Edge-IIoTset sensitivity analysis: text-valued fields unused (as in the main experiments) vs label-encoded; SMOTE vs SMOTENC; share of synthetic records with unseen category combinations |
| `edge_cat_check.py`, `edge_cat_check2.py` | Diagnostics: how the text-valued columns of Edge-IIoTset are handled by the preprocessing, and which numeric columns are binary flags or small-integer codes |
| `xai_eval.py` | Quantitative evaluation of SHAP vs LIME (deletion fidelity, stability, agreement, cost); text-valued columns are handled as in the strategy comparison (`FORCE_NUMERIC_TEXT = True`) |
| `requirements.txt` | Pinned library versions |
| `results_cic_strategies/`, `results_edge_strategies/`, `results_edge_confusion/`, `results_edge_text_sensitivity/`, `results_edge_cat_check/`, `results_xai_edge/`, `results_xai_cic/` | Result tables (CSV) and figures produced by the scripts above, one folder per script run |

## Environment

```
pip install -r requirements.txt
```

## Data (not included)

Download the datasets from their official sources and edit the path constants at the top of each script
(the defaults are Windows paths and must be changed):

* **CICIoT2023**: the 169 CSV files of the processed dataset in one folder (`DATA_FOLDER` / `CIC_FOLDER`).
* **Edge-IIoTset**: the file `ML-EdgeIIoT-dataset.csv` (`DATA_PATH` / `EDGE_PATH`).

## How to reproduce

```
python leakage_check.py                 # duplicate / overlap check (CICIoT2023)
python cic_strategies.py                # ~1 day on a single workstation; resumes automatically if interrupted
python edge_strategies.py               # ~70 minutes
python edge_confusion.py                # ~20 minutes; needs edge_strategies.py in the same folder
python edge_text_sensitivity.py         # about 1 hour; needs edge_strategies.py in the same folder
python edge_cat_check.py                # seconds; diagnostics for the text-valued columns
python edge_cat_check2.py               # seconds; diagnostics for the text-valued columns
python xai_eval.py edge                 # a few minutes
python xai_eval.py cic                  # ~20 minutes
```

Each script writes its tables to its own `results_*` folder.
`cic_strategies.py` stores every finished fold separately; re-running it continues from the first unfinished fold.
Set `QUICK_TEST = True` in `xai_eval.py` or `N_FOLDS = 2` in the strategy scripts for a short sanity run.

## Protocol summary

* **Splitting:** stratified K-fold (K = 10), fixed seed (42). Resampling, scaling, imputation and encoding are fitted on
  the training part of each fold only.
* **CICIoT2023 subset:** the first 100,000 rows of each of the 169 files (16.9 M rows). The training part of each fold is
  capped at 20,000 rows per class before any strategy is applied, so the *Baseline* is a capped, not a raw, training set.
* **Edge-IIoTset cleaning:** identifier / session / payload columns (IP addresses, timestamps, ports, payload fields) and the
  binary label are removed; exact duplicate rows are dropped; constant columns are dropped.
* **Text-valued columns (Edge-IIoTset):** a text column is converted to numbers when more than 95 % of its training values are
  numeric; the remaining text entries become missing values and are median-imputed. All seven text-valued fields passed this
  rule in every fold of the strategy comparison, so six of them are constant and one is nearly constant: their text content
  is not used (see `edge_cat_check.py`, `edge_cat_check2.py` and `edge_text_sensitivity.py`). The explanation study uses the same handling.
* **Statistics:** paired Wilcoxon signed-rank test over folds for each strategy vs Baseline, Holm-corrected per metric.
  The folds share training data, so the p-values are approximate.
* **XAI study:** 5 Random Forests trained on different 80 % subsamples of a capped training pool; explanations are computed for
  instances that all five models classify correctly. SHAP = `TreeExplainer`; LIME = tabular explainer with 5,000 samples.
  Fidelity = drop in predicted-class probability after replacing the top-k supporting features by their training median
  (compared with random removal). Stability = Jaccard / Spearman agreement of feature rankings between the five models.

## Known limitations

* Evaluation uses random stratified partitions of each dataset; no temporal or cross-dataset transfer is evaluated.
* The deletion test favours exact attribution methods such as `TreeExplainer` on tree ensembles.
* ADASYN fails on some folds of Edge-IIoTset; failures are recorded and the test uses the folds that completed.
* In the main Edge-IIoTset experiments the text-valued fields are unused; using them raises the Baseline macro-F1 by about
  0.05 to 0.06 without changing the conclusions on rebalancing (`edge_text_sensitivity.py`). Binary indicator and small-integer
  code columns are interpolated as if they were continuous by SMOTE-type methods in both datasets.
* MLP training times recorded in the logs of the long CICIoT2023 run are not reliable (the machine was not
  dedicated to the run).

## How to cite

Archived releases: https://doi.org/10.5281/zenodo.23977260 (see also `CITATION.cff`).

## License

Add the license of your choice (for example MIT) as a `LICENSE` file.
