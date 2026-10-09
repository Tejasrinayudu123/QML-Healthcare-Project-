from .data_loader import load_wdbc
from .preprocessing import prepare_data
from .classical_models import run_classical_baselines
from .quantum_vqc import run_vqc
from .quantum_kernel_svm import run_qksvm

def main():
    X, y = load_wdbc()

    X_train_c, X_test_c, y_train, y_test, _, _ = prepare_data(X, y)

    n_qubits = 4
    X_train_q, X_test_q, y_train_q, y_test_q, _, _ = prepare_data(
        X, y, n_components=n_qubits
    )

    classical_reports = run_classical_baselines(X_train_c, y_train, X_test_c, y_test)
    vqc_report = run_vqc(X_train_q, y_train_q, X_test_q, y_test_q, n_qubits=n_qubits)
    qksvm_report = run_qksvm(X_train_q, y_train_q, X_test_q, y_test_q, n_qubits=n_qubits)

    print("=== Advanced Classical vs Quantum Comparison ===")
    for r in classical_reports + [vqc_report, qksvm_report]:
        print(f"Model: {r.name}")
        print(f"  Accuracy: {r.accuracy:.4f}")
        print(f"  F1: {r.f1:.4f}")
        print(f"  ROC-AUC: {r.roc_auc:.4f}")
        print(f"  Train time (s): {r.train_time_sec:.2f}")
        print(f"  Infer time (s): {r.infer_time_sec:.4f}")
        print(f"  Qubits: {r.n_qubits}")
        print(f"  Circuit depth: {r.circuit_depth}")
        print(f"  Shots: {r.n_shots}")
        print(f"  Notes: {r.notes}")
        print()

if __name__ == "__main__":
    main()
