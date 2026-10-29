Explainable AI (XAI) System for IoT Intrusion Detection



This repository contains the source code and experimental pipeline for an Explainable Artificial Intelligence (XAI)-based Multiclass Intrusion Detection System (IDS) tailored for Internet of Things (IoT) environments.



📌 Project Overview



With the rapid proliferation of IoT devices, securing network traffic against complex cyberattacks has become vital. This project presents a framework designed to evaluate machine learning classifiers under severe class imbalance and provide transparent model predictions using post-hoc XAI techniques.



Dataset: CICIoT2023 Dataset (Fine-grained 34-class evaluation comprising benign and 33 attack categories).



Models Evaluated:



Random Forest (RF)



Extreme Gradient Boosting (XGBoost)



Multi-Layer Perceptron (MLP)



Explainability (XAI):



SHAP (SHapley Additive exPlanations): Global feature attribution analysis.



LIME (Local Interpretable Model-agnostic Explanations): Instance-level local explanation.



Imbalance Mitigation: Random Oversampling applied strictly to training partitions across 10-fold repeated stratified splits.



🛠️ Repository Structure



.

├── src/

│   └── main.py                 # Core experimental pipeline \& model evaluation

├── models/                     # Trained models (saved checkpoints)

├── results/                    # Generated charts, figures, and execution logs

├── requirements.txt            # Required Python packages

├── README.md                   # Project documentation

├── shap\_rf.png                 # Global SHAP feature importance plot

└── lime\_example.html           # Local LIME explanation sample





🚀 Getting Started



Prerequisites



Make sure you have Python 3.8+ installed on your system.



Installation



Clone this repository:



git clone https://github.com/YOUR\_USERNAME/YOUR\_REPOSITORY\_NAME.git

cd YOUR\_REPOSITORY\_NAME





Install required dependencies:



pip install -r requirements.txt





🔬 Experimental Pipeline



Preprocessing \& Scaling: Features are standardized using StandardScaler fitted exclusively on the training sets to prevent data leakage.



Repeated Stratified Splits: Evaluated across 10 independent stratified random 80:20 train-test splits.



Imbalance Mitigation: Random oversampling is executed on training partitions to address minority-class detection challenges.



Statistical Significance Testing: Includes non-parametric tests (e.g., Wilcoxon signed-rank test) to compare classifier performance.



Post-Hoc Interpretability: Generates feature attribution scores via SHAP and LIME for the Random Forest model.



📜 Citation \& License



If you find this code or research useful, please consider citing our work. This project is licensed under the MIT License.

