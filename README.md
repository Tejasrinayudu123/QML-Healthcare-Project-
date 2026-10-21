# Quantum Machine Learning for Healthcare
## Breast Cancer Diagnosis Using Classical and Quantum-Inspired Machine Learning

This project explores the application of Quantum Machine Learning (QML) techniques for healthcare diagnostics by comparing classical machine learning algorithms with quantum-inspired approaches for breast cancer diagnosis.

Using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset, the project evaluates predictive performance, quantum resource utilization, and computational efficiency to investigate the potential of quantum computing in healthcare applications.

---

## Overview

Breast cancer is one of the most prevalent forms of cancer worldwide, and early diagnosis significantly improves patient outcomes. This project develops a complete machine learning pipeline that combines traditional machine learning and quantum-inspired algorithms to classify tumors as benign or malignant.

The study focuses on:

- Data preprocessing and feature engineering
- Classical machine learning classification
- Quantum-inspired machine learning models
- Performance evaluation and comparison
- Quantum resource analysis

---

## Dataset

### Wisconsin Diagnostic Breast Cancer (WDBC)

Dataset Characteristics:

- 569 patient samples
- 30 diagnostic features
- Binary classification problem:
  - Benign
  - Malignant

Source:

- UCI Machine Learning Repository

---

## Implemented Models

### Classical Machine Learning Models

#### Support Vector Machine (RBF-SVM)

- Non-linear classification model
- Strong baseline for healthcare datasets

#### Random Forest Classifier

- Ensemble learning approach
- Robust against overfitting
- Strong performance on tabular data

---

### Quantum-Inspired Models

#### Variational Quantum Classifier (VQC)

Features:

- Quantum-inspired feature representation
- Variational optimization
- 4-qubit architecture
- Circuit-depth analysis

#### Quantum Kernel Support Vector Machine (QKSVM)

Features:

- Quantum kernel mapping
- ZZFeatureMap encoding
- Kernel-based classification
- Quantum resource evaluation

---

## Technologies Used

- Python 3.11
- Qiskit
- Qiskit Machine Learning
- Scikit-Learn
- NumPy
- Pandas
- Matplotlib
- Seaborn

---

## Project Structure

```text
QML-Healthcare-Project/

├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── metrics.py
│   ├── classical_models.py
│   ├── quantum_vqc.py
│   ├── quantum_kernel_svm.py
│   ├── resource_analysis.py
│   └── experiment_runner.py
│
├── requirements.txt
├── README.md
│
├── images/
│   ├── pca_plot.png
│   ├── accuracy_comparison.png
│   ├── roc_auc_comparison.png
│   ├── vqc_circuit.png
│   └── feature_map.png
│
└── report/
    └── IEEE_Report.tex
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/Tejasrinayudu123/QML-Healthcare-Project-.git
cd QML-Healthcare-Project-
```

### Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

```bash
python -m src.experiment_runner
```

The pipeline will:

1. Load the WDBC dataset
2. Preprocess the data
3. Train classical models
4. Train quantum-inspired models
5. Evaluate model performance
6. Perform resource analysis

---

# Experimental Results

## Model Performance Comparison

| Model | Accuracy | F1 Score | ROC-AUC |
|---------|---------|---------|---------|
| Quantum-Inspired VQC | **93.86%** | **95.36%** | **98.61%** |
| QKSVM | 57.02% | 70.66% | 47.78% |

---

## Quantum Resource Analysis

| Model | Qubits | Circuit Depth | Shots |
|---------|---------|---------|---------|
| Quantum-Inspired VQC | 4 | 12 | 1024 |
| QKSVM | 4 | 31 | Simulator |

---

## Key Findings

### Quantum-Inspired VQC

- Achieved the highest classification performance.
- ROC-AUC of 0.9861 indicates excellent predictive capability.
- Used a relatively shallow quantum circuit (depth 12).
- Demonstrated strong effectiveness for healthcare classification tasks.

### Quantum Kernel SVM (QKSVM)

- Lower performance under the current kernel configuration.
- Required deeper quantum circuits (depth 31).
- Performance suggests sensitivity to feature encoding and kernel design.

### Overall Observations

- Variational quantum approaches significantly outperformed the tested quantum kernel approach.
- Circuit depth directly impacted computational complexity.
- Quantum-inspired models show potential for future healthcare analytics applications.

---

## Research Contribution

This project contributes:

- A reproducible Quantum Machine Learning workflow for healthcare applications.
- Comparative analysis of classical and quantum-inspired models.
- Evaluation of quantum resource requirements.
- Experimental assessment of breast cancer diagnosis using QML techniques.

---

## Limitations

- Experiments were conducted using simulated quantum environments.
- Limited qubit availability.
- Single healthcare dataset.
- No execution on physical quantum hardware.
- Limited hyperparameter tuning.

---

## Future Work

Potential future extensions include:

- Hybrid Quantum-Classical Neural Networks
- Execution on IBM Quantum Hardware
- Electronic Health Record (EHR) Analysis
- Medical Image Classification
- Genomic Disease Prediction
- Noise-Aware Quantum Training
- Advanced Quantum Feature Maps

---

## IEEE Research Report

A complete IEEE-format short research paper accompanies this project and includes:

- Background and Motivation
- Related Work
- Methodology
- Dataset Description
- Classical Baselines
- Quantum Machine Learning Approaches
- Experimental Results
- Resource Analysis
- Limitations
- Future Research Directions

---

## Author

**Tejasri Nayudu**

Master of Science in Computer Science  
Middle Tennessee State University

### Research Interests

- Quantum Machine Learning
- Healthcare Artificial Intelligence
- Machine Learning
- Backend Development
- Applied Artificial Intelligence

---
