This project implements a comprehensive machine learning pipeline for breast cancer diagnosis using both classical ML models and quantum machine learning (QML) techniques.
It compares traditional algorithms with quantum‑enhanced models to evaluate improvements in accuracy, robustness, and computational efficiency.

The goal is to explore how quantum computing can support healthcare diagnostics by improving pattern recognition in complex medical datasets.

✨ Key Features
Classical Machine Learning
Support Vector Machine (RBF‑SVM)

Random Forest Classifier

Logistic Regression

Standard preprocessing (scaling, PCA, train/test split)

Quantum Machine Learning
Variational Quantum Classifier (VQC)

Quantum Kernel SVM (QKSVM)

SPSA optimizer for quantum circuits

Qiskit‑based quantum circuit construction

Evaluation Metrics
Accuracy

Precision

Recall

F1‑Score

ROC‑AUC

Confusion Matrix

Project Capabilities
End‑to‑end ML/QML pipeline

Modular architecture

Easy to extend with new models

Clean separation of classical vs quantum workflows

Reproducible experiment runner

📁 Project Structure
Code
COMPREHENSIVE_DBMS/
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── classical_models.py
│   ├── quantum_vqc.py
│   ├── quantum_kernel_svm.py
│   ├── experiment_runner.py
│   └── __init__.py
│
├── requirements.txt
├── README.md
└── venv/  (ignored)
⚙️ Installation & Setup
1. Clone the repository
Code
git clone https://github.com/Tejasrinayudu123/QML-Healthcare-Project-.git
cd QML-Healthcare-Project-
2. Create a virtual environment
Code
python -m venv venv
venv\Scripts\activate
3. Install dependencies
Code
pip install -r requirements.txt
🚀 Running the Project
Use the experiment runner module:

Code
python -m src.experiment_runner
This will automatically:

Load the dataset

Preprocess the data

Train classical models

Train quantum models

Print evaluation metrics

Compare classical vs quantum performance

📊 Results (To Be Updated)
Once you run the experiment, paste the output here.
I will help you convert it into:

Professional tables

Graphs (ROC, PCA, circuit diagrams)

Final report format

🧠 Technologies Used
Python 3.11

Qiskit

scikit‑learn

NumPy / Pandas

Matplotlib / Seaborn

Quantum Circuit Optimization (SPSA)

🎯 Purpose of the Project
This project demonstrates how quantum computing can be applied to real‑world healthcare problems.
It serves as a strong portfolio project for:

Machine Learning

Quantum Computing

Data Science

Research & Academia

Healthcare AI

📌 Future Improvements
Add more quantum ansatz architectures

Implement QAOA‑based classification

Add hyperparameter tuning

Deploy as a web API

Add GPU acceleration for classical models

👤 Author
Tejasri Nayudu  
Master’s in Computer Science
Middle Tennessee State University
Backend Developer | ML/QML Enthusiast
