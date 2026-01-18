# tabpfn-evaluation
Empirical evaluation of TabPFN on tabular datasets

# Evaluating TabPFN on Tabular Classification Tasks

This project evaluates TabPFN on tabular classification problems and compares its performance against classical machine learning baselines.
The focus is on performance, robustness, calibration, stability, and computational behavior, using clean and reproducible experimental scripts.
# Objectives

The project aims to answer the following questions:

How does TabPFN perform compared to classical models on tabular data?

Is the performance fair across classes (Balanced Accuracy)?

Are the predicted probabilities well calibrated (Brier Score)?

How robust is the model to noise, duplicates, and reduced data?

Are results stable across different random seeds?

How does fit time vs inference time compare across models?

# Project Structure
project-root/
│
├── datasets/
│   └── breast_cancer/
│       └── breast_cancer.csv
│
├── scripts/
│   ├── baseline_eval.py
│   ├── tabpfn_breast_cancer_eval.py
│   ├── robustness_eval.py
│   ├── stability_eval.py
│   ├── seed_sensitivity_eval.py
│   └── tabpfn_vs_baselines_summary.py
│
├── results/
│   ├── tabpfn_vs_baselines_summary.csv
│   └── ...
│
├── notebooks/          # optional (exploration)
├── src/                # optional (future extensions)
└── README.md

# Models Evaluated

TabPFN

Logistic Regression

Random Forest

All models are evaluated on the same train–test splits for fair comparison.

# Evaluation Metrics

The following metrics are used throughout the experiments:

Accuracy

Balanced Accuracy
→ accounts for class imbalance and fairness

F1-score

Brier Score
→ evaluates probabilistic calibration (lower is better)

Fit Time
→ model setup / training time

Predict Time
→ inference latency

# Experiments
1. Standard Evaluation

Evaluates all models on clean data using a fixed train–test split.

Script:

scripts/tabpfn_breast_cancer_eval.py

2. Robustness Evaluation

Tests model behavior under dataset perturbations:

Gaussian noise

Duplicated samples

Reduced dataset size

Metrics include accuracy, balanced accuracy, F1, Brier score, and timing.

Script:

scripts/robustness_eval.py

3. Stability Evaluation

Runs multiple experiments with different random seeds and reports mean ± standard deviation.

Script:

scripts/stability_eval.py

4. Seed Sensitivity Analysis

Explicitly evaluates performance across seeds {0, 42, 99} to verify consistency.

Script:

scripts/seed_sensitivity_eval.py

5. Final Summary Comparison

Generates a single consolidated comparison table across all models and metrics.

Script:

scripts/tabpfn_vs_baselines_summary.py


Output:

results/tabpfn_vs_baselines_summary.csv

# How to Run

Install dependencies:

pip install -r requirements.txt


Run any evaluation script from the scripts/ directory:

python tabpfn_vs_baselines_summary.py

# Key Takeaways

Balanced Accuracy ensures fair evaluation across classes

Brier Score provides insight into probabilistic calibration

TabPFN shows competitive performance without iterative training

Results remain stable across random seeds

Fit vs predict time analysis clarifies TabPFN’s computational behavior

# Notes

Current experiments focus on binary classification

Multi-class extensions (e.g., Wine Quality without binarization) can be added later

The project is designed to be modular and easily extensible

# License

This project is for academic and educational use.
