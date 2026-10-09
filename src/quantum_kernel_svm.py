from qiskit_aer import AerSimulator
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit.circuit.library import ZZFeatureMap
from sklearn.svm import SVC
from .metrics import evaluate_model

def run_qksvm(X_train, y_train, X_test, y_test, n_qubits=4):
    feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2)
    backend = AerSimulator(method="statevector")

    qkernel = QuantumKernel(
        feature_map=feature_map,
        quantum_instance=backend,
    )

    K_train = qkernel.evaluate(X_train)
    K_test = qkernel.evaluate(X_test, X_train)

    svm = SVC(kernel="precomputed", C=1.0)

    report = evaluate_model(
        svm, K_train, y_train, K_test, y_test,
        n_qubits=n_qubits,
        circuit_depth=feature_map.decompose().depth(),
        n_shots=0,
        name=f"QKSVM ({n_qubits} qubits)",
        notes="Quantum kernel SVM with ZZFeatureMap."
    )
    return report
